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
from src.model.model import CDCK2,CDCK5, SpkClassifier
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
    parser.add_argument('--raw-hdf5', required=True)
    parser.add_argument('--validation-raw', required=True)
    parser.add_argument('--eval-raw', required=True)
    parser.add_argument('--train-list', required=True)
    parser.add_argument('--validation-list', required=True)
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
    parser.add_argument('--keep-res', action='store_true', default=False,
                        help='origin size')
    parser.add_argument('--not-rand-crop', action='store_true', default=True)
    parser.add_argument('--no-color-aug', action='store_true', default=False)
    parser.add_argument('--flip', default=0.5, type=float)
    parser.add_argument('--scale-shift-values', default=1.0, type=float)
    parser.add_argument('--color-aug-value', default=0.5, type=int)
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
    parser.add_argument('--alpha-parameter', default=1.0, type=float)
    parser.add_argument('--alpha-resnet', default=1.0)
    parser.add_argument('--alpha-rev-cpc', default=0.1, type=float)
    parser.add_argument('--alpha-cpc', default=1.0)
    parser.add_argument('--alpha', default=2.0,type=float)
    parser.add_argument('--weibull_tail', default=23, type=int, help='Classes used in testing with 20')
    parser.add_argument('--weibull_alpha', default=3, type=int, help='Classes used in testing with 3')
    parser.add_argument('--weibull-threshold', default=0, type=float, help='Classes used in testing 0.063')
    parser.add_argument('--mask-alpha', type=float, default=0.01, help='weight for explainabilty mask loss')
    parser.add_argument('--mask-input-channel', default=1, type=int)
    parser.add_argument('--result-list')
    args = parser.parse_args()
    fullpath = args.result_list + "/" + "map.list"
    # print("fullpath=",fullpath)
    with open(fullpath, "r") as F:
        line = F.readlines()
        args.spk_num = len(line)
    #print(line)
    #print(args.spk_num)
    
    if (not os.path.exists(args.logging_dir)):
        os.makedirs(args.logging_dir)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    print('use_cuda is', use_cuda)
    global_timer = timer() # global timer
    logger = setup_logs(args.logging_dir, run_name_spk) # setup logs
    device = torch.device("cuda" if use_cuda else "cpu")
    cdc_model = CDCK2(args.timestep, args.batch_size, args.audio_window, args.alpha_parameter)#.to(device)
    #checkpoint = torch.load(args.model_path, map_location=lambda storage, loc: storage) # load everything onto CPU
    #checkpoint = checkpoint.to(device)
    #cdc_model.load_state_dict(checkpoint['state_dict'])
    cdc_model = cdc_model.to(device)
    decom_model = FDCN(1,3,3,3)
    mask_model = MaskExpNet(args).to(device)
    #decom_checkpoint = torch.load(args.decom_model_path, map_location=lambda storage, loc: storage)  # load everything onto CPU
    #decom_model.load_state_dict(decom_checkpoint['state_dict'])
    decom_model = decom_model.to(device)
    #spk_model = SpkClassifier(args.spk_num).to(device)
    spk_model = TrainModule(args).to(device)
    ## Loading the dataset
    params = {'num_workers': 0,
              'pin_memory': False} if use_cuda else {}

    logger.info('===> loading train, validation and eval dataset')
    training_set   = RawDatasetSpkClass(args.raw_hdf5, args.all_sets, args.train_list, args.index_file, args.audio_window, args.frame_window)
    validation_set = RawDatasetSpkClass(args.validation_raw, args.all_sets,args.validation_list, args.index_file, args.audio_window, args.frame_window)
    eval_set = RawDatasetSpkClass(args.eval_raw, args.all_sets, args.eval_list, args.index_test_file, args.audio_window, args.frame_window)
    train_loader = data.DataLoader(training_set, batch_size=args.batch_size, shuffle=True, **params) # set shuffle to True
    validation_loader = data.DataLoader(validation_set, batch_size=args.batch_size, shuffle=False, **params) # set shuffle to False
    eval_loader = data.DataLoader(eval_set, batch_size=args.batch_size, shuffle=False, **params) # set shuffle to False
    # nanxin optimizer
    optimizer_decom = ScheduledOptim(
        optim.Adam(
            filter(lambda p: p.requires_grad, decom_model.parameters()),
            betas=(0.9, 0.98), eps=1e-09, weight_decay=1e-4, amsgrad=True), args.n_warmup_steps)
    optimizer_mask = ScheduledOptim(
        optim.Adam(
            filter(lambda p: p.requires_grad, mask_model.parameters()),
            betas=(0.9, 0.98), eps=1e-09, weight_decay=1e-4, amsgrad=True), args.n_warmup_steps)

    optimizer_cpc = ScheduledOptim(
        optim.Adam(
            filter(lambda p: p.requires_grad, cdc_model.parameters()),
            betas=(0.9, 0.98), eps=1e-09, weight_decay=1e-4, amsgrad=True), args.n_warmup_steps)

    optimizer = ScheduledOptim(
        optim.Adam(
            filter(lambda p: p.requires_grad, spk_model.parameters()), 
            betas=(0.9, 0.98), eps=1e-09, weight_decay=1e-4, amsgrad=True),args.n_warmup_steps)
    spk_model = torch.nn.DataParallel(spk_model).cuda()
    model_params = sum(p.numel() for p in spk_model.parameters() if p.requires_grad)
    logger.info('### Model summary below###\n {}\n'.format(str(spk_model)))
    logger.info('===> Model total parameter: {}\n'.format(model_params))
    ## Start training
    best_ss_acc = np.inf
    best_ce_acc = 0
    best_loss = np.inf
    best_epoch = -1 
    for epoch in range(1, args.epochs + 1):
        epoch_timer = timer()

        # Train and validate
        train_spk(args, cdc_model, decom_model, mask_model, spk_model, device, train_loader, optimizer_decom, optimizer_mask, optimizer_cpc, optimizer,
                  epoch, args.batch_size, args.frame_window)
        val_acc, ss_val_acc, val_loss = validation_spk(args, cdc_model, decom_model, mask_model, spk_model, device, validation_loader, args.batch_size, args.frame_window)
        ss_acc_range = best_ss_acc - ss_val_acc
        ce_acc_range = best_ce_acc - val_acc
        # Save
        if (val_acc >= best_ce_acc and ss_val_acc <= 2.0*best_ss_acc) :
                #or (ss_val_acc > best_ss_acc and val_acc <= 1.05*best_ce_acc) or \
                #      (ss_val_acc >= 0.95*best_ss_acc and val_acc <= best_ce_acc):
            best_ss_acc = min(ss_val_acc, best_ss_acc)
            best_ce_acc = max(val_acc, best_ce_acc)
        #if val_loss < best_loss:
        #    best_loss = min(val_loss, best_loss)
            snapshot(args.logging_dir, run_name_decom, {
                'epoch': epoch + 1,
                'validation_acc': val_acc, 
                'state_dict': decom_model.state_dict(),
                'validation_loss': val_loss,
                'optimizer': optimizer_decom.state_dict(),
            })
            snapshot(args.logging_dir, run_name_cpc, {
                'epoch': epoch + 1,
                'validation_acc': val_acc,
                'state_dict': cdc_model.state_dict(),
                'validation_loss': val_loss,
                'optimizer': optimizer_cpc.state_dict(),
            })
            snapshot(args.logging_dir, run_name_mask, {
                    'epoch': epoch + 1,
                    'validation_acc': val_acc,
                    'state_dict': mask_model.state_dict(),
                    'validation_loss': val_loss,
                    'optimizer': optimizer_mask.state_dict(),
            })
            snapshot(args.logging_dir, run_name_spk, {
                'epoch': epoch + 1,
                'validation_acc': val_acc,
                'state_dict': spk_model.state_dict(),
                'validation_loss': val_loss,
                'optimizer': optimizer.state_dict(),
            })
            best_epoch = epoch + 1

        elif epoch - best_epoch > 2:
            optimizer.increase_delta()
            best_epoch = epoch + 1
        
        end_epoch_timer = timer()
    logger.info("#### End epoch {}/{}, elapsed time: {}".format(epoch, args.epochs, end_epoch_timer - epoch_timer))
        ## prediction
    logger.info('===> loading best model for prediction')
    checkpoint_spk = torch.load(os.path.join(args.logging_dir, run_name_spk + '-model_best.pth'))
    checkpoint_decom = torch.load(os.path.join(args.logging_dir, run_name_decom + '-model_best.pth'))
    checkpoint_cpc = torch.load(os.path.join(args.logging_dir, run_name_cpc + '-model_best.pth'))
        # pdb.set_trace()
    spk_model.load_state_dict(checkpoint_spk['state_dict'])
    decom_model.load_state_dict(checkpoint_decom['state_dict'])
    cdc_model.load_state_dict(checkpoint_cpc['state_dict'])
    prediction_spk(args, decom_model, cdc_model, spk_model, device, eval_loader, args.batch_size, args.frame_window)


    ## end
    end_global_timer = timer()
    logger.info("################## Success #########################")
    logger.info("Total elapsed time: %s" % (end_global_timer - global_timer))

if __name__ == '__main__':
    main()
