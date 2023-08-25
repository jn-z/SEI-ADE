## Utilities
from __future__ import print_function
import argparse
import random
import time
import os
import logging
from timeit import default_timer as timer
from network.train_net import TrainModule
## Libraries
import numpy as np

## Torch
import torch
import torch.nn as nn
from torch.utils import data
import torch.nn.functional as F
import torch.optim as optim
import pdb
## Custrom Imports
from src.logger_v1 import setup_logs
from src.data_reader.dataset import RawDatasetSpkClass,TorchDataSet
from src.training_v1 import train_spk, snapshot
from src.validation_v1 import validation_spk
from src.prediction_v1 import prediction_spk
#from src.prediction_v1_test_result import prediction_spk
from src.model.model import CDCK2, SpkClassifier
from src.model.model_decom import FDCN, FDRN
from src.model.MaskNet import MaskExpNet
############ Control Center and Hyperparameter ###############
run_name_cpc = "cdc"
run_name_spk = "resnet"
run_name_decom = "decom"
run_name_mask = "mask"
class ScheduledOptim(object):
    """A simple wrapper class for learning rate scheduling"""

    def __init__(self, optimizer, n_warmup_steps):
        self.optimizer = optimizer
        self.d_model = 128 
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0 
        self.delta = 1

    def state_dict(self):
        self.optimizer.state_dict()

    def step(self):
        """Step by the inner optimizer"""
        self.optimizer.step()

    def zero_grad(self):
        """Zero out the gradients by the inner optimizer"""
        self.optimizer.zero_grad()

    def increase_delta(self):
        self.delta *= 2

    def update_learning_rate(self):
        """Learning rate scheduling per step"""

        self.n_current_steps += self.delta
        new_lr = np.power(self.d_model, -0.5) * np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        return new_lr

def main():
    ## Settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--raw-hdf5', required=False)
    parser.add_argument('--validation-raw', required=False)
    parser.add_argument('--eval-raw', required=True)
    parser.add_argument('--train-list', required=False)
    parser.add_argument('--validation-list', required=False)
    parser.add_argument('--eval-list')
    parser.add_argument('--all-sets')
    parser.add_argument('--index-test-file')
    parser.add_argument('--index-file')
    parser.add_argument('--logging-dir', required=True,
                        help='model save directory')
    parser.add_argument('--model-path')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--n-warmup-steps', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=32,
                        help='batch size')
    parser.add_argument('--audio-window', type=int, default=1280,
                        help='window length to sample from each utterance')
    parser.add_argument('--frame-window', type=int, default=1)
    parser.add_argument('--spk-num', type=int, default=10)
    parser.add_argument('--timestep', type=int, default=15)
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--margin-posneg', default=0.2)
    parser.add_argument('--marginType',default='softmax',type=str)
    parser.add_argument('--cemargin', default=0.0, type=float)
    parser.add_argument('--cescale', default=0.0, type=float)
    parser.add_argument('--is-ce', default=1, type=int)
    parser.add_argument('--loss-type',default='batch_hard',type=str)
    parser.add_argument('--cvector-dim', default=512, type=int)
    parser.add_argument('--fea-dim', default=256, type=int)
    parser.add_argument('--weibull_tail', default=23, type=int, help='Classes used in testing with 20')
    parser.add_argument('--weibull_alpha', default=3, type=int, help='Classes used in testing with 3')
    parser.add_argument('--weibull-threshold', default=0, type=float, help='Classes used in testing 0.063')
    parser.add_argument('--alpha-parameter', default=1.0)
    parser.add_argument('--alpha-resnet', default=1.0)
    parser.add_argument('--alpha-rev-cpc', default=0.1)
    parser.add_argument('--alpha-cpc', default=1.0)
    parser.add_argument('--alpha', default=2.0)
    parser.add_argument('--mask-alpha', type=float, help='weight for explainabilty mask loss', default=0.03)
    parser.add_argument('--mask-input-channel', default=1, type=int)
    parser.add_argument('--result-list')
    args = parser.parse_args()
    fullpath = args.result_list + "/" + "map.list"
    # print("fullpath=",fullpath)
    with open(fullpath, "r") as F:
        line = F.readlines()
        args.spk_num = len(line)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    print('use_cuda is', use_cuda)
    global_timer = timer() # global timer
    logger = setup_logs(args.logging_dir, run_name_spk) # setup logs
    device = torch.device("cuda" if use_cuda else "cpu")
    decom_model = FDCN(1, 3, 3, 3)
    checkpoint_decom = torch.load(os.path.join(args.logging_dir, run_name_decom + '-model_best.pth'))
    #spk_model = torch.nn.DataParallel(spk_model).cuda()  # data parallelism over GPUs
    decom_model.load_state_dict(checkpoint_decom['state_dict'])
    decom_model = decom_model.to(device)
    for param in decom_model.parameters():
        param.requires_grad = False
    mask_model = MaskExpNet(args).to(device)
    checkpoint_mask = torch.load(os.path.join(args.logging_dir, run_name_mask + '-model_best.pth'))
    # spk_model = torch.nn.DataParallel(spk_model).cuda()  # data parallelism over GPUs
    mask_model.load_state_dict(checkpoint_mask['state_dict'])
    mask_model = mask_model.to(device)
    for param in mask_model.parameters():
        param.requires_grad = False

    cdc_model = CDCK2(args.timestep, args.batch_size, args.audio_window, args.alpha_parameter)
    #checkpoint = torch.load(args.model_path, map_location=lambda storage, loc: storage) # load everything onto CPU
    checkpoint_cpc = torch.load(os.path.join(args.logging_dir, run_name_cpc + '-model_best.pth'))
    cdc_model.load_state_dict(checkpoint_cpc['state_dict'])
    cdc_model = cdc_model.to(device)
    for param in cdc_model.parameters():
        param.requires_grad = False
    #spk_model = SpkClassifier(args.spk_num).to(device)
    spk_model = TrainModule(args).to(device)
    ## Loading the dataset
    params = {'num_workers': 0,
              'pin_memory': False} if use_cuda else {}
    # nanxin optimizer
    optimizer = ScheduledOptim(
        optim.Adam(
            filter(lambda p: p.requires_grad, spk_model.parameters()),
            betas=(0.9, 0.98), eps=1e-09, weight_decay=1e-4, amsgrad=True),
        args.n_warmup_steps)

    model_params = sum(p.numel() for p in spk_model.parameters() if p.requires_grad)
    logger.info('### Model summary below###\n {}\n'.format(str(spk_model)))
    logger.info('===> Model total parameter: {}\n'.format(model_params))
    logger.info('===> loading train, validation and eval dataset')
    #training_set   = RawDatasetSpkClass(args.raw_hdf5, args.train_list, args.index_file, args.audio_window, args.frame_window)
    #validation_set = RawDatasetSpkClass(args.validation_raw, args.validation_list, args.index_file, args.audio_window, args.frame_window)
    eval_set = RawDatasetSpkClass(args.eval_raw,args.all_sets, args.eval_list, args.index_file, args.audio_window, args.frame_window)
    #train_loader = data.DataLoader(training_set, batch_size=args.batch_size, shuffle=True, **params) # set shuffle to True
    #validation_loader = data.DataLoader(validation_set, batch_size=args.batch_size, shuffle=False, **params) # set shuffle to False
    eval_loader = data.DataLoader(eval_set, batch_size=args.batch_size, shuffle=False, **params) # set shuffle to False
    #主数据加载test5
    logger.info('===> loading best model for prediction')
    checkpoint = torch.load(os.path.join(args.logging_dir, run_name_spk + '-model_best.pth'))
    spk_model = torch.nn.DataParallel(spk_model).cuda()  # data parallelism over GPUs
    spk_model.load_state_dict(checkpoint['state_dict'])
    prediction_spk(args, decom_model, cdc_model,mask_model, spk_model, device, eval_loader, args.batch_size, args.frame_window)
    ## end 
    end_global_timer = timer()
    logger.info("################## Success #########################")
    logger.info("Total elapsed time: %s" % (end_global_timer - global_timer))

if __name__ == '__main__':
    main()
