import os
import yaml
import sys
import torch
import json
from os.path import join

class Logger(object):
    def __getattr__(self, k):
        return print

log = Logger()

class AttributeDict(dict):
    """ 
    An extended dictionary that allows access to elements as atttributes and counts 
    these accesses. This way, we know if some attributes were never used. 
    """

    def __init__(self, *args, **kwargs):
        from collections import Counter
        super().__init__(*args, **kwargs)
        self.__dict__['counter'] = Counter()

    def __getitem__(self, k):
        self.__dict__['counter'][k] += 1
        return super().__getitem__(k)

    def __getattr__(self, k):
        self.__dict__['counter'][k] += 1
        return super().get(k)

    def __setattr__(self, k, v):
        return super().__setitem__(k, v)

    def __delattr__(self, k, v):
        return super().__delitem__(k, v)    

    def unused_keys(self, exceptions=()):
        return [k for k in super().keys() if self.__dict__['counter'][k] == 0 and k not in exceptions]

    def assume_no_unused_keys(self, exceptions=()):
        if len(self.unused_keys(exceptions=exceptions)) > 0:
            log.warning('Unused keys:', self.unused_keys(exceptions=exceptions))

class TrainingLogger(object):

    def __init__(self, model, log_dir, config=None, *args):
        super().__init__()
        self.model = model
        self.base_path = join(f'logs/{log_dir}') if log_dir is not None else None

        os.makedirs('logs/', exist_ok=True)
        os.makedirs(self.base_path, exist_ok=True)

        if config is not None:
            json.dump(config, open(join(self.base_path, 'config.json'), 'w'))

    def iter(self, i, **kwargs):
        if i % 100 == 0 and 'loss' in kwargs:
            loss = kwargs['loss']
            print(f'iteration {i}: loss {loss:.4f}')

    def save_weights(self, only_trainable=False, weight_file='weights.pth'):
        if self.model is None:
            raise AttributeError('You need to provide a model reference when initializing TrainingTracker to save weights.')

        weights_path = join(self.base_path, weight_file)

        weight_dict = self.model.state_dict()

        if only_trainable:
            weight_dict = {n: weight_dict[n] for n, p in self.model.named_parameters() if p.requires_grad}
        
        torch.save(weight_dict, weights_path)
        log.info(f'Saved weights to {weights_path}')

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        """ automatically stop processes if used in a context manager """
        pass


def training_config_from_cli_args():
    experiment_name = sys.argv[1]
    experiment_id = int(sys.argv[2])

    yaml_config = yaml.load(open(f'{experiment_name}'), Loader=yaml.SafeLoader)

    config = yaml_config['configuration']
    config = {**config, **yaml_config['individual_configurations'][experiment_id]}
    config = AttributeDict(config)
    return config

def get_attribute(name):
    import importlib

    if name is None:
        raise ValueError('The provided attribute is None')
    
    name_split = name.split('.')
    mod = importlib.import_module('.'.join(name_split[:-1]))
    return getattr(mod, name_split[-1])

def filter_args(input_args, default_args):

    updated_args = {k: input_args[k] if k in input_args else v for k, v in default_args.items()}
    used_args = {k: v for k, v in input_args.items() if k in default_args}
    unused_args = {k: v for k, v in input_args.items() if k not in default_args}

    return AttributeDict(updated_args), AttributeDict(used_args), AttributeDict(unused_args)