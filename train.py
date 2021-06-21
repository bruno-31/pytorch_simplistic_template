import time
import random
import math
import pprint
from tqdm import tqdm
import argparse
import os
from utils import resume_training,num_parameters,str2bool,save_checkpoint, generate_key,\
    show_gpu_mem, seed_worker, load_checkpoint, get_lr

from metrics import PSNR,L2
from dataloaders import SimpleDataloader

import numpy as np
import torch
import torchvision


def main(args, *vargs, **kwargs):
    # fix rnd seed
    if args.deterministic:
        torch.manual_seed(0); np.random.seed(0); random.seed(0)
        generator = torch.Generator(); generator.manual_seed(0)  # seed dataloaders workers
        torch.use_deterministic_algorithms(True)

    # prepare torch devices
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # prepare dataloaders dict {'train': .. 'test':}
    loaders_dict = SimpleDataloader.basicImageDataloader(args.train_path,
                                                         args.test_path,
                                                         train_batch=args.train_batch,
                                                         test_batch=args.test_batch,
                                                         worker_init_fn=seed_worker if args.deterministic else None,
                                                         generator=generator if args.deterministic else None,
                                                         num_iters=args.num_iters, num_workers=args.num_workers)

    # prepare experiment's logging files and checkpoint
    experiment = args.experiment if args.experiment is not None else generate_key()
    experiment_dir = os.path.join(args.logdir, experiment)
    os.makedirs(experiment_dir, exist_ok=True)
    starting_epoch, best_value_criterion, args = resume_training(ckpt_path=experiment_dir, args=args)
    if args.tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(experiment_dir)

    # prepare model and restore weights if needed
    if args.model == 'ResUnet':
        from models.ResUnet import ResUNet as Net
        model = Net(in_nc=3, out_nc=3, nb=1, nc=[32,32,32,32], **vars(args)).to(device)
    else:
        raise NotImplementedError('Unknown model architecture')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_decay)
    not starting_epoch or load_checkpoint(os.path.join(experiment_dir, 'last.pth.tar'), model, optimizer) # loading weights if possible

    # multi gpu
    if args.n_gpu >1:
        model = torch.nn.DataParallel(model)

    # prepare metrics and loss criterion
    loss_fn = L2(boundary_ignore=0)
    metrics_fn = {'l2':L2(), 'Psnr':PSNR(boundary_ignore=10)}
    criterion = 'Psnr' # key metric for saving best

    # ----------------------  perform training ----------------------

    print(f'Device : {device} \nModel \'{args.model}\', Trainable params : {num_parameters(model)} \nSaving @ {experiment_dir} '
          f'\nCriterion best {best_value_criterion:0.3f} \nParameters :')
    pprint.pprint(vars(args)); print('\n')

    for epoch in range(starting_epoch, args.num_epochs):

        for phase in ['test'] if args.test_only else ['train','test']:

            model.train() if phase == 'train' else model.eval()
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
                is_best = False
                if epoch % args.test_ev != 0 or epoch == 0:
                    continue

            # init statistics
            statistics_dict = {metric: 0. for metric in metrics_fn.keys()}  # initialize statistics
            tic = time.time()

            # Iterate over data.
            num_iters = 0
            for hr in tqdm(loaders_dict[phase], disable=not args.tqdm):
                hr = hr.to(device)
                lr = hr
                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    pred = model(lr)
                    loss = loss_fn(pred, hr)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        optimizer.zero_grad() # zero the parameter gradients
                        loss.backward()
                        optimizer.step()
                        scheduler.step()

                # update statistics
                vars_dict = {'pred': pred, 'gt': hr}
                statistics_dict = {key: value + metrics_fn[key](**vars_dict) for (key, value) in statistics_dict.items()} # update all metrics
                num_iters += 1

            # average statistics
            tac = time.time()
            statistics_dict = {key: value / num_iters for (key, value) in statistics_dict.items()}

            # console logging
            epoch_log = f'{phase} {epoch:03d} | {(tac-tic):0.2f}s | {(tac-tic)/num_iters:0.2f}s/iter | lr:{get_lr(optimizer):0.2e}' # display basic info
            epoch_log += "".join([f' | {key}:{value:0.3f}' for key,value in statistics_dict.items()]) # display all metrics
            epoch_log += f' | gpu {show_gpu_mem()[1]:0.1f} Mb' if device!=torch.device('cpu') else '' # display gpu usage
            tqdm.write(epoch_log)

            # tensorboard logging
            if args.tensorboard:
                [writer.add_scalar(f'{key}/{phase}', value, epoch) for (key, value) in statistics_dict.items()]
                for name,image in {'hr':hr,'lr':lr}.items():
                    grid = torchvision.utils.make_grid(hr, nrow=int(math.sqrt(image.shape[0])))
                    writer.add_image(f'{name}/{phase}', grid, epoch)
                writer.flush()

            # saving best model wrt. chosen criterion
            if phase == 'test' and statistics_dict[criterion] > best_value_criterion:
                best_value_criterion = statistics_dict[criterion] ; is_best = True

        # saving checkpoint
        if not args.test_only:
            save_checkpoint({'state_dict': model.state_dict(),
                             'optim_dict':optimizer.state_dict(),
                             'epoch': epoch,
                             'args': args,
                             'best_value': best_value_criterion}, is_best, experiment_dir)

    if args.tensorboard:
        writer.close()


if __name__ == '__main__':
    # settings
    parser = argparse.ArgumentParser()
    # IO parameters
    parser.add_argument("--train_path", type=str, help="Path to the dir containing the training dataset.")
    parser.add_argument("--test_path", type=str, help="Path to the dir containing the testing dataset.")
    parser.add_argument("--logdir", type=str, help="dir to save experiments", default='experiments_logs')
    parser.add_argument("--experiment", type=str, help="The name of the experiment to be saved.", default=None)
    parser.add_argument("--tensorboard", type=str2bool, help="The name of the experiment to be saved.", default=False)
    # optimization parameters
    parser.add_argument("--train_batch", type=int, help="Training batch size", default=1)
    parser.add_argument("--test_batch", type=int, help="Training batch size", default=1)
    parser.add_argument("--lr", type=float, dest="lr", help="Learning rate", default=2e-4)
    parser.add_argument("--lr_step", type=int, dest="lr_step", help="Learning rate decrease step", default=40000)
    parser.add_argument("--lr_decay", type=float, help="Learning rate decay (on step)", default=0.35)
    parser.add_argument("--num_epochs", type=int, help="Total number of epochs to train", default=250)
    parser.add_argument("--num_iters", type=int, help="Num iters per epoch, (set 0 for one pass) ", default=0)
    parser.add_argument("--test_ev", type=int, help="Test every x epochs", default=10)
    parser.add_argument("--test_only", type=str2bool, help="Only testing", default=False)
    # model parameters
    parser.add_argument("--model", type=str, help="The name of the experiment to be saved.", default='ResUnet')
    # misc
    parser.add_argument("--n_gpu", type=int, help="Learning rate decrease step", default=1)
    parser.add_argument("--num_workers", type=int, help="Learning rate decrease step", default=1)
    parser.add_argument("--deterministic", type=str2bool, help="Learning rate decrease step", default=False)
    parser.add_argument("--tqdm", type=str2bool, help="tqdm", default=False)
    args = parser.parse_args()

    main(args)