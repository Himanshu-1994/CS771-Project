from torch.functional import Tensor

import torch
import inspect
import json
import yaml
import time
import sys

from general_utils import log

import numpy as np
from os.path import expanduser, join, isfile, realpath

from torch.utils.data import DataLoader

from metrics import FixedIntervalMetrics

from general_utils import load_model, log, score_config_from_cli_args, AttributeDict, get_attribute, filter_args

from tqdm import tqdm

def load_model(checkpoint_id, weights_file=None, strict=True, model_args='from_config', with_config=False, ignore_weights=False):

    config = json.load(open(join('logs', checkpoint_id, 'config.json')))

    if model_args != 'from_config' and type(model_args) != dict:
        raise ValueError('model_args must either be "from_config" or a dictionary of values')

    #model_cls = get_attribute(config['model'])

    model_cls = get_attribute("model.ClipPred")

    # load model
    if model_args == 'from_config':
        _, model_args, _ = filter_args(config, inspect.signature(model_cls).parameters)

    model = model_cls(**model_args)

    if weights_file is None:
        weights_file = realpath(join('logs', checkpoint_id, 'weights.pth'))
    else:
        weights_file = realpath(join('logs', checkpoint_id, weights_file))

    if isfile(weights_file) and not ignore_weights:
        weights = torch.load(weights_file, map_location = torch.device("cpu"))
        for _, w in weights.items():
            assert not torch.any(torch.isnan(w)), 'weights contain NaNs'
        model.load_state_dict(weights, strict=strict)
    else:
        if not ignore_weights:
            raise FileNotFoundError(f'model checkpoint {weights_file} was not found')

    if with_config:
        return model, config
    
    return model

def score(config, train_checkpoint_id, train_config):

    config = AttributeDict(config)

    print(config)

    # use training dataset and loss
    train_config = AttributeDict(json.load(open(f'logs/{train_checkpoint_id}/config.json')))

    #cp_str = f'_{config.iteration_cp}' if config.iteration_cp is not None else ''

    #model_cls = get_attribute(train_config['model'])

    model_cls = get_attribute("model.ClipPred")

    _, model_args, _ = filter_args(train_config, inspect.signature(model_cls).parameters)

    model_args = {**model_args, **{k: config[k] for k in ['process_cond', 'fix_shift'] if k in config}}

    #strict_models = {'ConditionBase4', 'PFENetWrapper'}

    model = load_model(train_checkpoint_id, strict=False, model_args=model_args, 
                        weights_file='weights.pth')
                           

    model.eval()
    model.cuda()

    metric_args = dict()

    # if 'threshold' in config:
    #     if config.metric.split('.')[-1] == 'SkLearnMetrics':
    #         metric_args['threshold'] = config.threshold

    # if 'resize_to' in config:
    #     metric_args['resize_to'] = config.resize_to

    if 'sigmoid' in config:
        metric_args['sigmoid'] = config.sigmoid    

    # if 'custom_threshold' in config:
    #     metric_args['custom_threshold'] = config.custom_threshold     

    from data import PhraseCut

    # only_visual = config.only_visual is not None and config.only_visual
    # with_visual = config.with_visual is not None and config.with_visual

    # dataset = PhraseCut('test', 
    #                     image_size=train_config.image_size,
    #                     mask=config.mask, 
    #                     with_visual=with_visual, only_visual=only_visual, aug_crop=False, 
    #                     aug_color=False)

    dataset = PhraseCut('test', 
                        image_size=config.image_size,
                        negative_prob = 0)

    loader = DataLoader(dataset, batch_size=config.batch_size, num_workers=2, shuffle=False, drop_last=False)
    metric = get_attribute(config.metric)(resize_pred=True, **metric_args)

    shift = config.shift if 'shift' in config else 0


    with torch.no_grad():

        i, losses = 0, []
        for i_all, (data_x, data_y) in tqdm(enumerate(loader)):
            data_x = [v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v for v in data_x]
            data_y = [v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v for v in data_y]

            pred = model(data_x[0], data_x[1])
            metric.add([pred + shift], data_y)

            i += 1
            if config.max_iterations and i >= config.max_iterations:
                break                

    key_prefix = config['name'] if 'name' in config else 'phrasecut'      
    return {key_prefix: metric.scores()}
    #return {key_prefix: {k: v for k, v in zip(metric.names(), metric.value())}}

def main():
    config, train_checkpoint_id = score_config_from_cli_args()

    metrics = score(config, train_checkpoint_id, None)

    for dataset in metrics.keys():
        for k in metrics[dataset]:
            if type(metrics[dataset][k]) in {float, int}:
                print(dataset, f'{k:<16} {metrics[dataset][k]:.3f}')

if __name__ == '__main__':
    main()