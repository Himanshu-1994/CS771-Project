import transformers
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from data import PhraseCut
from torchmetrics.classification import BinaryJaccardIndex
from tqdm import tqdm as tqdm

"""
Binary Jaccard Index (i,j) = a/(a + b + c) where is the a is the number of (1,1), b is (1,0), c is (0,1) across data points i & j
"""

def mIOU(threshold = 0.5):
    return BinaryJaccardIndex(threshold = threshold)

def compute_fail_cases(batch_size = 32, num_workers = 4, thresh = 0.5):
    dataset = PhraseCut("test", image_size=352)
    test_loader = DataLoader(dataset, batch_size = batch_size, num_workers = num_workers)

    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

    miou_metric = mIOU()

    count = 1

    for idx, (img, mask) in tqdm(enumerate(test_loader)):
        imgs = [img[0][idx, :].permute(1,2,0) for idx in range(img[0].shape[0])]
        prompts = list(img[1])
        masks = [mask[0][idx, :].permute(1,2,0).squeeze(-1) for idx in range(mask[0].shape[0])]

        current_batch_size = len(imgs)
        
        inputs = processor(text=prompts, images=imgs, padding="max_length", return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)
        
        preds = outputs.logits.unsqueeze(1)

        # print(preds[0][0])
        # print(masks[0])
        # print(preds[0][0].shape)
        # print(masks[0].shape)
        # print(torch.unique(masks[0]))

        #print(prompts[0])

        for idx in range(current_batch_size):
            miou = miou_metric(preds[idx][0], masks[idx])

            print(miou)

            if(miou < thresh):
                fig = plt.figure(figsize = (8,8))

                gs1 = gridspec.GridSpec(1, 3)

                ax1 = plt.subplot(gs1[0])
                ax1.imshow(imgs[idx])

                ax2 = plt.subplot(gs1[1])
                ax2.imshow(torch.sigmoid(preds[idx][0]))

                ax3 = plt.subplot(gs1[2])
                ax3.imshow(masks[idx])

                ax1.axis('off')
                ax2.axis('off')
                ax3.axis('off')

                ax1.text(0.5,-0.1, "(a) Input Image", size=12, ha="center", 
                transform=ax1.transAxes)

                ax2.text(0.5,-0.1, f"(b) {prompts[idx]}", size=12, ha="center", 
                transform=ax2.transAxes)

                ax3.text(0.5,-0.1, "Ground Truth", size=12, ha="center", 
                transform=ax3.transAxes)
                
                plt.savefig(f"failure_cases/fail_{count}_{miou}.png", facecolor = 'w', dpi = 300)

                count += 1

if __name__ == "__main__":
    """
    For now, I'm resizing dataset images & masks to 352x352 to match pre-trained ClipSeg's configurations
    """

    """
    Use their fancy method to convert to binary mask (177, 250)
    Right now your metrics are very strict, figure out how to go easy
    """

    """
    TODO:
    1. Figure out Jaccard's internal threshold
    2. Figure out thresh for classifying as failures
    3. Use their method to convert to binary mask
    """
    compute_fail_cases()