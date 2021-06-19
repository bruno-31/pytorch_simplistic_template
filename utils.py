import os
import torch
import shutil
import random
import argparse
from datetime import datetime

import numpy as np

def resume_training(ckpt_path, arsg, device):
    """Check if experiment exists already. If yes, load arguments and starting epoch. If not create dir
    """
    if os.path.exists(ckpt_path):
        print(f'ckpt found @{ckpt_path}')
        checkpoint = torch.load(ckpt_path, map_location=device)
        args = checkpoint['args'] #use parameter old experiment
        starting_epoch = checkpoint['epoch']
        print(f"resuming training @ epoch {starting_epoch})")
    else:
        print(f'no ckpt found @{ckpt_path}, creating directory')
        os.makedirs(os.path.dirname(ckpt_path))
        starting_epoch = 0
    return starting_epoch


def save_checkpoint(state, is_best, checkpoint):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'
    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    filepath = os.path.join(checkpoint, 'last.pth.tar')
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    else:
        print("Checkpoint Directory exists! ")
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))


def load_checkpoint(checkpoint_path, model, optimizer=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.
    Args:
        checkpoint_path: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint_path):
        raise("File doesn't exist {}".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location=model.device)
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint_path


def num_parameters(model):
    return sum(p.numel() for p in model.parameters())


def generate_key(date=True):
    if date:
        timestampStr = datetime.now().strftime("%d%b%H:%M:%S")
        return timestampStr
    return '{}'.format(random.randint(0, 100000))


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def show_gpu_mem():
    mem = torch.cuda.memory_allocated() * 1e-6
    max_mem = torch.cuda.max_memory_allocated() * 1e-6
    return mem, max_mem


def seed_worker(worker_id):
    ''' Seed Dataloader workers for reproducibility
    '''
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)