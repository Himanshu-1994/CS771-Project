import os
import yaml
import sys
import torch
import json
import inspect
from os.path import join
from os.path import join, realpath, isfile

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

        self.train_log = join(self.base_path,'train.log')
        self.val_log = join(self.base_path,'val.log')

        if config is not None:
            json.dump(config, open(join(self.base_path, 'config.json'), 'w'))

    def iter(self, i, **kwargs):
        if i % 100 == 0 and 'loss' in kwargs:
            loss = kwargs['loss']
            print(f'iteration {i}: loss {loss:.4f}')
            with open(self.train_log,'a') as file:
              file.write(f'iteration, {i}, loss, {loss:.4f}\n')
              file.close()

    def iter_val(self, i, val_loss, **kwargs):
      
      with open(self.val_log,'a') as file:
        print(f'iteration {i}: loss {val_loss:.4f}')
        file.write(f'iteration, {i}, loss, {val_loss:.4f}\n')
        file.close()
      

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


"""New util functions added"""

def load_model(checkpoint_id, weights_file=None, strict=True, model_args='from_config', with_config=False):

    config = json.load(open(join('logs', checkpoint_id, 'config.json')))

    if model_args != 'from_config' and type(model_args) != dict:
        raise ValueError('model_args must either be "from_config" or a dictionary of values')

    model_cls = get_attribute(config['model'])

    # load model
    if model_args == 'from_config':
        _, model_args, _ = filter_args(config, inspect.signature(model_cls).parameters)

    model = model_cls(**model_args)

    if weights_file is None:
        weights_file = realpath(join('logs', checkpoint_id, 'weights.pth'))
    else:
        weights_file = realpath(join('logs', checkpoint_id, weights_file))

    if isfile(weights_file):
        weights = torch.load(weights_file)
        for _, w in weights.items():
            assert not torch.any(torch.isnan(w)), 'weights contain NaNs'
        model.load_state_dict(weights, strict=strict)
    else:
        raise FileNotFoundError(f'model checkpoint {weights_file} was not found')

    if with_config:
        return model, config
    
    return model

def score_config_from_cli_args():
    experiment_name = sys.argv[1]
    #experiment_id = int(sys.argv[2])

    train_config = AttributeDict(json.load(open(f'logs/{experiment_name}/config.json')))

    #yaml_config = yaml.load(open(f'experiments/{experiment_name}'), Loader=yaml.SafeLoader)

    # config = yaml_config['test_configuration_common']

    # if type(yaml_config['test_configuration']) == list:
    #     test_id = int(sys.argv[3])
    #     config = {**config, **yaml_config['test_configuration'][test_id]}
    # else:
    #     config = {**config, **yaml_config['test_configuration']}

    # if 'test_configuration' in yaml_config['individual_configurations'][experiment_id]:
    #     config = {**config, **yaml_config['individual_configurations'][experiment_id]['test_configuration']}

    train_config = {"normalize": True, "image_size": train_config.img_size, "batch_size": train_config.batch_size, "name": train_config.name, "metric": "metrics.FixedIntervalMetrics", "split":"test", "mask":"text", "label_support": True, "sigmoid": True}

    #train_config = AttributeDict(train_config)

    train_checkpoint_id = train_config["name"]

    #config = AttributeDict(config)

    return train_config, train_checkpoint_id