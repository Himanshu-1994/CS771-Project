import os
from os.path import expanduser, isdir, isfile, join

import numpy as np
from itertools import groupby

from PIL import Image

from torchvision import transforms
import torch

from utils import get_from_repository

class PCDataset(object):
    def __init__(self, split, image_size = 400, negative_prob = 0):
        super().__init__()

        self.image_size = image_size
        self.negative_prob = negative_prob

        get_from_repository('PhraseCut', ['PhraseCut.tar'], integrity_check=lambda local_dir: all([
            isdir(join(local_dir, 'VGPhraseCut_v0')),
            isdir(join(local_dir, 'VGPhraseCut_v0', 'images')),
            isfile(join(local_dir, 'VGPhraseCut_v0', 'refer_train.json')),
            len(os.listdir(join(local_dir, 'VGPhraseCut_v0', 'images'))) in {108250, 108249}
        ]))

        self.base_path = join(expanduser('~/datasets/PhraseCut/VGPhraseCut_v0/images/'))

        # The following import is from https://github.com/ChenyunWu/PhraseCutDataset.git
        # The repository provides API's to work with the PhraseCut dataset
        # RefVGLoader: loads the dataset from files. It uses PhraseHandler to handle the phrases.
        # split can be 'val', 'test', or 'train'
        from PhraseCutDataset.utils.refvg_loader import RefVGLoader
        self.refvg_loader = RefVGLoader(split = split)

        # any idea why these values?
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.normalize = transforms.Normalize(mean, std)

        # img_ids where the size in the annotations does not match actual size
        invalid_img_ids = set([150417, 285665, 498246, 61564, 285743, 498269, 
                               498010, 150516, 150344, 286093, 61530, 150333, 
                               286065, 285814, 498187, 285761, 498042])

        self.sample_ids = [(i, j) 
                           for i in self.refvg_loader.img_ids 
                           for j in range(len(self.refvg_loader.get_img_ref_data(i)['phrases']))
                           if i not in invalid_img_ids]

        # grouping the dataset by phrases and creating text based prompts
        samples_by_phrase = sorted([(self.refvg_loader.get_img_ref_data(i)['phrases'][j], (i, j)) for i, j in self.sample_ids])
        samples_by_phrase = groupby(samples_by_phrase, key = lambda x: x[0])
        samples_by_phrase = {prompt: [s[1] for s in prompt_sample_ids] for prompt, prompt_sample_ids in samples_by_phrase}
        self.samples_by_phrase = samples_by_phrase

        self.all_phrases = list(set(self.samples_by_phrase.keys()))

    def __len__(self):
        return len(self.sample_ids)

    def load_sample(self, i, j):
        img_ref_data = self.refvg_loader.get_img_ref_data(i)
        phrase = img_ref_data['phrases'][j]

        sly, slx = slice(0, None), slice(0, None)

        img = np.array(Image.open(join(self.base_path, str(img_ref_data['image_id']) + '.jpg')))[sly, slx]
        if img.ndim == 2:
            img = np.dstack([img] * 3)
        img = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).float()
        img = torch.nn.functional.interpolate(img, (self.image_size, self.image_size), mode='bilinear', align_corners=True)[0]
        img = self.normalize(img / 255.0)

        from skimage.draw import polygon2mask
        polys_phrase = img_ref_data['gt_Polygons'][j]
        masks = []
        for polys in polys_phrase:
            for poly in polys:
                poly = [p[::-1] for p in poly]
                masks += [polygon2mask((img_ref_data['height'], img_ref_data['width']), poly)]
        seg = np.stack(masks).max(0)
    
        seg = seg[sly, slx].astype('uint8')
        seg = torch.from_numpy(seg).view(1, 1, *seg.shape)
        seg = torch.nn.functional.interpolate(seg, (self.image_size, self.image_size), mode='nearest')[0,0]

        return img, seg, phrase

    def __getitem__(self, i):
        sample_i, j = self.sample_ids[i]
        img, seg, phrase = self.load_sample(sample_i, j)

        # negative samples -> the sample’s phrase is replaced by a 
        # different phrase with a probability q_neg.
        if self.negative_prob > 0:
            if torch.rand((1,)).item() < self.negative_prob:
                new_phrase = None
                while new_phrase is None or new_phrase == phrase:
                    idx = torch.randint(0, len(self.all_phrases), (1,)).item()
                    new_phrase = self.all_phrases[idx]
                phrase = new_phrase
                seg = torch.zeros_like(seg)

        vis_s = [phrase]
        seg = seg.unsqueeze(0).float()
        data_x = (img,) + tuple(vis_s)

        return data_x, (seg, torch.zeros(0), i)