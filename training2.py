import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchvision
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda.amp import autocast, GradScaler

import math
import pandas as pd
import numpy as np
import random
from contextlib import nullcontext
from functools import partial
import os
from os.path import isfile, join
import inspect
import sys
import argparse
from general_utils import TrainingLogger, training_config_from_cli_args, get_attribute, filter_args, log

import data

from model import ClipPred

def cosine_warmup_lr(i, warmup=10, max_iter=90):
    """ Cosine LR with Warmup """
    if i < warmup:
        return (i+1)/(warmup+1)
    else:
        return 0.5 + 0.5*math.cos(math.pi*(((i-warmup)/(max_iter- warmup))))

def validate(model, dataset, config,device):
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)

    metric_class, use_metric = config.val_metric_class, config.use_val_metric
    #loss_fn = get_attribute(config.loss)
    #loss_fn = F.binary_cross_entropy_with_logits

    model.eval()
    model.to(device)

    if metric_class is not None:
        metric = get_attribute(metric_class)()

    with torch.no_grad():

        i, losses = 0, []
        for data_x, data_y in data_loader:

            data_x = [x.to(device) if isinstance(x, torch.Tensor) else x for x in data_x]
            data_y = [x.to(device) if isinstance(x, torch.Tensor) else x for x in data_y]

            prompts = model.sample_prompts(data_x[1], prompt_list=('a photo of a {}',))
            with autocast():
                pred  = model(data_x[0], prompts)

            if metric_class is not None:
                metric.add([pred], data_y)

            # pred = model(data_x[0], prompts)
            # loss = loss_fn(pred[0], data_y[0])
            loss = F.binary_cross_entropy_with_logits(pred, data_y[0])
            losses += [float(loss)]

            i += 1

            #if config.val_max_iterations is not None and i > config.val_max_iterations:
                #break

    if use_metric is None:
        return np.mean(losses), {}, False
    else:
        metric_scores = {m: s for m, s in zip(metric.names(), metric.value())} if metric is not None else {}
        return np.mean(losses), metric_scores, True

def main(config):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #loading hyperparams from yaml file
    #config = training_config_from_cli_args()

    #val_interval is set to null
    #val_interval, best_val_loss, best_val_score = config.val_interval, float('inf'), float('-inf')
    best_val_loss, best_val_score = float('inf'), float('-inf')

    #returns model class (ClipDensePredT in default .yaml case)
    #model_cls = get_attribute(config.model)
    #inspect.signature(func).parameters returns ordered dictionary with signature of function
    #this function uses hyperparams in .yaml file to pass on to constructor of model_cls
    #_, model_args, _ = filter_args(config, inspect.signature(model_cls).parameters)

    #instantiating model using params above and moving to gpu
    #model = model_cls(**model_args).cuda()
    model = ClipPred(reduce_dim=args.reduce_dim,device=device)
    model.to(device)

    #Similar story with the dataset
    #dataset_cls = get_attribute(config.dataset)
    #_, dataset_args, _ = filter_args(config, inspect.signature(dataset_cls).parameters)

    dataset = data.PhraseCut("train", image_size = args.img_size, negative_prob = args.negative_prob)
    #dataset = dataset_cls(**dataset_args)

    log.info(f'Train dataset {dataset.__class__.__name__} (length: {len(dataset)})')
    
    # optimizer
    opt = torch.optim.AdamW(
                model.parameters(),
                config.lr,
            )
                #weight_decay=args.weight_decay,

    if config.lr_scheduler == 'cosine':
        assert config.T_max is not None and config.eta_min is not None
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, config.T_max, config.eta_min)
    elif config.lr_scheduler == 'warmup_cosine':        
        lr_scheduler = LambdaLR(opt, partial(cosine_warmup_lr, max_iter=(config.max_iterations), warmup=config.warmup))
    else:
        lr_scheduler = None

    #simple to understand
    batch_size, max_iterations = config.batch_size, config.max_iterations

    #loss_fn = get_attribute(config.loss)

    #loss_fn = torch.nn.functional.binary_cross_entropy_with_logits()

    #For automatic mixed precision
    if config.amp:
        log.info('Using AMP')
        autocast_fn = autocast
        scaler = GradScaler()
    else:
        autocast_fn, scaler = nullcontext, None

    save_only_trainable = True
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=args.num_workers)

    train_len = len(data_loader)
    val_interval = train_len
    
    checkpoint_iterations = [(i+1)*1000 for i in range(20)]

    #val_interval = None
    if val_interval is not None:
        #dataset_val_args = {k[4:]: v for k,v in config.items() if k.startswith('val_') and k != 'val_interval'}
        #trying to inject hyperparams in .yaml file into constructor for dataset
        #_, dataset_val_args, _ = filter_args(dataset_val_args, inspect.signature(dataset_cls).parameters)
        #print('val args', {**dataset_args, **{'split': 'val', 'aug': 0}, **dataset_val_args})

        #dataset_val = dataset_cls(**{**dataset_args, **{'split': 'val', 'aug': 0}, **dataset_val_args})
        dataset_val = data.PhraseCut("val", image_size = args.img_size)

    # disable config when hyperparam. opt. to avoid writing logs.
    #tracker_config = config if not config.hyperparameter_optimization else None
    tracker_config = vars(config)

    with TrainingLogger(log_dir=config.name, model=model, config=tracker_config) as logger:

        i = 0
        while True:
            for data_x, data_y in data_loader:

                # between caption and output feature.
                # 1. Sample random captions
                # 2. Check alignment with CLIP

                # randomly mix text and visual support conditionals

                # now this if condition will evaluate to False for now (only text prompts, no visual support for now)
                if config.mix:

                    assert config.mask.startswith('text_and')

                    with autocast_fn():
                        # data_x[1] = text label
                        prompts = model.sample_prompts(data_x[1])

                        # model.clip_model()

                        text_cond = model.compute_conditional(prompts)
                        if model.__class__.__name__ == 'CLIPDensePredTMasked':
                            # when mask=='separate'
                            visual_s_cond, _, _ = model.visual_forward_masked(data_x[2].to(device), data_x[3].to(device))
                        else:
                            # data_x[2] = visual prompt
                            visual_s_cond, _, _ = model.visual_forward(data_x[2].to(device))

                    max_txt = config.mix_text_max if config.mix_text_max is not None else 1
                    batch_size = text_cond.shape[0]

                    # sample weights for each element in batch
                    text_weights = torch.distributions.Uniform(config.mix_text_min, max_txt).sample((batch_size,))[:, None]
                    text_weights = text_weights.to(device)

                    if dataset.__class__.__name__ == 'PhraseCut':
                        # give full weight to text where support_image is invalid
                        visual_is_valid = data_x[4] if model.__class__.__name__ == 'CLIPDensePredTMasked' else data_x[3]
                        text_weights = torch.max(text_weights[:,0], 1 - visual_is_valid.float().to(device)).unsqueeze(1)

                    cond = text_cond * text_weights + visual_s_cond * (1 - text_weights)

                else:
                    # no mix
                    
                    # we shouldnt need this if condition
                    if model.__class__.__name__ == 'CLIPDensePredTMasked':
                        # compute conditional vector using CLIP masking
                        with autocast_fn():
                            assert config.mask == 'separate'
                            cond, _, _ = model.visual_forward_masked(data_x[1].to(device), data_x[2].to(device))
                    else:
                        #conditional vector is just the text prompt passed through text transformer
                        cond = data_x[1]
                        if isinstance(cond, torch.Tensor):
                            cond = cond.to(device)

                with autocast_fn():
                    visual_q = None

                    pred = model(data_x[0].to(device), cond)

                    loss = F.binary_cross_entropy_with_logits(pred, data_y[0].to(device))

                    if torch.isnan(loss) or torch.isinf(loss):
                        # skip if loss is nan
                        log.warning('Training stopped due to inf/nan loss.')
                        sys.exit(-1)

                    extra_loss = 0
                    loss += extra_loss

                opt.zero_grad()

                if scaler is None:
                    loss.backward()
                    opt.step()
                else:
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()

                if lr_scheduler is not None:
                    lr_scheduler.step()
                    if i % 2000 == 0:
                        current_lr = [g['lr'] for g in opt.param_groups][0]
                        log.info(f'current lr: {current_lr:.5f} ({len(opt.param_groups)} parameter groups)')

                logger.iter(i=i, loss=loss)                    
                i += 1

                if i >= max_iterations:

                    if not isfile(join(logger.base_path, 'weights.pth')):
                        # only write if no weights were already written
                        logger.save_weights(only_trainable=save_only_trainable)
                    
                    sys.exit(0)

                #checkpoint_iterations = [i+1 for i in range(20)]*1000
                if checkpoint_iterations is not None and i in checkpoint_iterations:
                #if config.checkpoint_iterations is not None and i in config.checkpoint_iterations:
                    logger.save_weights(only_trainable=save_only_trainable, weight_file=f'weights_{i}.pth')

                
                if val_interval is not None and i % val_interval == val_interval - 1:

                    val_loss, val_scores, maximize = validate(model, dataset_val, config, device)
                    
                    if len(val_scores) > 0:

                        score_str = f', scores: ' + ', '.join(f'{k}: {v}' for k, v in val_scores.items())
                        
                        if maximize and val_scores[config.use_val_metric] > best_val_score:
                            logger.save_weights(only_trainable=save_only_trainable)
                            best_val_score = val_scores[config.use_val_metric]

                        elif not maximize and val_scores[config.use_val_metric] < best_val_score:
                            logger.save_weights(only_trainable=save_only_trainable)
                            best_val_score = val_scores[config.use_val_metric]

                    else:
                        score_str = ''
                        # if no score is used, fall back to loss
                        if val_loss < best_val_loss:
                            logger.save_weights(only_trainable=save_only_trainable)
                            best_val_loss = val_loss
                    
                    log.info(f'Validation loss: {val_loss}' + score_str)
                    logger.iter(i=i, val_loss=val_loss, extra_loss=float(extra_loss), **val_scores)
                    model.train()

            print('epoch complete')

def argument_parser():
  parser = argparse.ArgumentParser()

  parser.add_argument("--name",default="pc",type=str,help="Name")

  parser.add_argument("--batch-size",default=64,type=int,help="Batch Size for Training",dest="batch_size")
  parser.add_argument("--max-iterations",default=20000,type=int,help="Max Iterations",dest="max_iterations")
  parser.add_argument("-ckpt","--checkpoint-iterations",default=1000,type=int,help="Checkpoint",dest="checkpoint_iterations")
  parser.add_argument("--image-size",default=352,type=int,help="Internal embedding size",dest="img_size")

  parser.add_argument("--amp",default=True,type=bool,help="Automatic Mixed Precision")
  parser.add_argument("--mix",default=False,type=bool,help="Image and Text Prompts")


  parser.add_argument("--num_workers",default=4,type=int,help="Number of workers in DataLoader")
  parser.add_argument("--reduce-dim",default=64,type=int,help="Internal embedding size",dest="reduce_dim")

  parser.add_argument("-lr", "--learning-rate", default=0.001, type=float,help="initial learning rate",dest="lr")

  parser.add_argument("--model",default="models.clipseg.ClipPred",type=str,help="Model")
  parser.add_argument("--mask",default="text",type=str,help="mask")

  parser.add_argument("-lrs","--lr-scheduler",default="cosine",type=str,help="LR Scheduler",dest="lr_scheduler")
  parser.add_argument("--T-max",default=20000,type=int,help="Tmax",dest="T_max")
  parser.add_argument("--eta-min",default=0.0001,type=float,help="Eta Min",dest="eta_min")
  parser.add_argument("--negative-prob",default=0.2,type=float,help="Negative Sampling",dest="negative_prob")
  
  parser.add_argument("--use-val-metric",default=None,type=bool,help="Use Validation Metric",dest="use_val_metric")
  parser.add_argument("--val-metric-class",default=None,type=str,help="Validation Metric",dest="val_metric_class")

  parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
  parser.add_argument(
      "--wd",
      "--weight-decay",
      default=1e-4,
      type=float,
      metavar="W",
      help="weight decay (default: 1e-4)",
      dest="weight_decay",
  )



  return parser.parse_args()

if __name__ == "__main__":
    args = argument_parser()
    main(args)
