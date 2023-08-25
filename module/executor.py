import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level = logging.DEBUG,
                    format = '%(asctime)s[%(levelname)s] %(name)s -%(message)s',
                    )
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils
import logging
import os
import re
from torch.utils.checkpoint import checkpoint_sequential
import time
import torch.nn.functional as F
from .utils import LRDecayOptimizer, CELossMetric, VELossMetric, CombineMetric, AccMetric, \
                    Speedometer, BatchEndParam
import numpy as np
from struct import pack,calcsize
import os
import ctypes
import torch.distributed as dist
from .utils import BMUFAdam, BMUFSGD
import operator
from module.utils import all_gather_info
from collections import defaultdict
import math
# import horovod.torch as hvd
# from horovod.torch.mpi_ops import allreduce, allreduce_async, allreduce_, allreduce_async_, broadcast_
# from horovod.torch.mpi_ops import poll, synchronize
import time
# from .dataset_bht_new import DataSet as build_data
from module.sampler import train_data_sampler
from .dataset import DataSet as build_data_dev
# from threading import Thread, Lock
# from queue import Queue

from multiprocessing import Process, Queue, Lock, Value

from tensorboardX import SummaryWriter
import pdb
from .triplet_loss import calc_cdist
from .triplet_loss import batch_hard,batch_all

class Executor(object):
    def __init__(self, args):
        self.args = args
        self.num_thread = 4
        self.exist_id = '0000000'
        self.ncount = Value('i', 0)
        self.mutex = Lock()
        pass

    def save_checkpoint(self, model, optimizer, filename):
        torch.save({'model':model.state_dict(), 'optimizer':None},filename)

    def load_checkpoint(self, model, optimizer, filename):

        checkpoint = torch.load(filename)
        if model is not None:
            model_dict = model.state_dict()
            cnn_key_list = [key for key in checkpoint['model'].keys() if 'resnet' in key]
            for key in cnn_key_list:
                model_dict[key] = checkpoint['model'][key]
            cnn_key_list = [key for key in checkpoint['model'].keys() if 'xvector' in key]
            for key in cnn_key_list:
                model_dict[key] = checkpoint['model'][key]
                
            model.load_state_dict(model_dict)

    def _create_model_dir(self):
        if not os.path.exists(self.args.model_dir):
            try:
                os.makedirs(self.args.model_dir)
            except:
                logging.warn("model dir has already been made by other threads")

    def _tain_summary_init(self):
        ## for summary
        if self.args.summary:
            summary_dir = './summary/' + "rank{}".format(self.args.rank)
            if os.path.exists(summary_dir):
                import shutil
                shutil.rmtree(summary_dir, ignore_errors=False)
            self.summary_writer = SummaryWriter(summary_dir)
        else:
            pass

    def _get_train_param(self, train_module):
        train_params = []
        for name, param in train_module.named_parameters():
            if param.requires_grad:
                train_params.append(param)
            else:
                pass
        return train_params

    def _creat_optimizer(self, train_params):
        bm_lr = 1.0
        bm_const = 1.25
        bm_mom = (1.0-bm_lr/(self.args.distributed_world_size*bm_const))
        if self.args.use_bmuf and self.args.rank==0 :
            logging.info("Update with bmuf algorithm, bm_lr is %f, bm_mom is %f"%(bm_lr, bm_mom))
        else:
            pass

        optimizer = BMUFSGD(train_params, lr = self.args.lr_rate, weight_decay=1e-5, bm_lr=bm_lr, bm_mom=bm_mom)
        return optimizer

    def _load_checkpoint(self, train_module):
        if self.args.check_point is not None:
            if self.args.rank == 0:
                self.load_checkpoint(model=train_module, optimizer=None, filename=self.args.check_point)
                logging.info("Load Check Point {}".format(self.args.check_point))
            for name, param in train_module.named_parameters():
                dist.broadcast(param.data, src=0)
                # broadcast_(param.data, root_rank=0)

    def _lr_scale(self, epoch, optimizer):
        if epoch >= self.args.decay_epoch:
            curr_lr = self.args.lr_rate * (self.args.decay_ratio**(epoch-self.args.decay_epoch+1))
        else:
            curr_lr = self.args.lr_rate
        optimizer._set_lr(curr_lr)

        if self.args.rank == 0:
            logging.info("set optimizer lrate to {}".format(curr_lr))

    
    def _adjust_learning_rate(self,optimizer, epoch, lr, schedule, gamma=0.1):
        #lr = self._adjust_learning_rate(self.optimizer, epoch, lr, schedule, gamma=0.1)
        if epoch in schedule:
            curr_lr = lr * gamma
        else:
            curr_lr = lr
        optimizer._set_lr(curr_lr)
        
        if self.args.rank == 0:
            logging.info("set optimizer lrate to {}".format(lr))
        #return lr
        return curr_lr
    def fit(self, train_module):
        all_gather_info("task local device ids is {}".format(self.args.rank), self.args.gpu)
        self._create_model_dir()
        self._tain_summary_init()

        self.train_module = train_module.cuda(self.args.gpu)
        self.train_params = self._get_train_param(self.train_module)
        self.optimizer = self._creat_optimizer(self.train_params)
        self._load_checkpoint(self.train_module)

        self.train_metric = CombineMetric([CELossMetric(), VELossMetric(), AccMetric()])
        self.callbacks = [Speedometer(frequent= self.args.display_freq)]

        #solve train stop problem
        self.npart = math.floor(self.args.rank/self.args.ranks_group)
        all_gather_info('rank %s reading lmdb file from %s'%(self.args.rank, self.args.train[self.npart]), self.args.gpu)
        
        # lr = self.args.lr_rate
        # schedule = self.args.lr_schedule
        for epoch in list(range(self.args.nepochs)):
            # print('111111111')
            self.trainset = train_data_sampler(lmdb_file=self.args.train[self.npart],label_key_file=self.args.train_key[self.npart],chunk_spk_num=self.args.chunk_spk_num,batch_spk_num=self.args.batch_p,spk_voices=self.args.batch_k,is_train=False)

            # print('22222')
            #print('##########rank is :',self.args.rank)
            self.train_epoch(epoch, self.trainset, mean_file=self.args.train_mean[self.npart], dstvar_file=self.args.train_variance[self.npart])
            # print('3333')
            self._lr_scale(epoch, self.optimizer)
            
            #lr = self._adjust_learning_rate(self.optimizer, epoch, lr, schedule, gamma=0.1)
    def train_epoch(self, nepoch, trainset, mean_file, dstvar_file):

        self.train_module.train()
        self.train_metric.reset()
        trainset.start()
        tic = time.time()

        for bid, batch_data in enumerate(trainset):
            try:

                src, src_mask, state_lab = batch_data
                batch_size, fea_frames, fea_dim = src.size()

                mean_value = np.loadtxt(mean_file, skiprows=0, dtype='float32')
                mean_value = torch.Tensor(mean_value)
                src = src - mean_value.repeat(batch_size, 1 , 1)

                dstvar_value = np.loadtxt(dstvar_file, skiprows=0, dtype='float32')
                dstvar_value = torch.Tensor(dstvar_value)
                src = src * dstvar_value.repeat(batch_size, 1, 1)
                # src = src * src_mask

                params = [p.cuda(self.args.gpu) for p in (src, state_lab)]
                # print('############self.npart:',self.npart)
                anchor_sv, predict, ce_loss = self.train_module(*params, fea_frames, self.npart) 
                
                ve_loss_data = {}
                ve_loss_data["dist"] = calc_cdist(anchor_sv,anchor_sv.detach())
                ve_loss_data["pids"] = torch.squeeze(state_lab.cuda(self.args.gpu))
                ve_loss_data["gpu"] = self.args.gpu
                if self.args.loss_type == 'batch_hard':
                    ve_loss = batch_hard(ve_loss_data["dist"],ve_loss_data["pids"],self.args.margin_posneg)
                elif self.args.loss_type == 'batch_all':
                    ve_loss,fraction_positive_triplets = batch_all(ve_loss_data["dist"],ve_loss_data["pids"],ve_loss_data["gpu"],self.args.margin_posneg)
                ce_loss = ce_loss.sum()
                #ve_loss = ve_loss.sum()
                
                if self.args.is_ce:
                    total_loss = (ce_loss / batch_size) +ve_loss
                else:
                    total_loss = ve_loss
                
                self.optimizer.zero_grad()
                total_loss.sum().backward()
                torch.nn.utils.clip_grad_norm_(self.train_params, 1.0)
                self.optimizer.step()
                forward_result = dict()
                forward_result['ce_loss'] = ce_loss.cpu()
                forward_result['ve_loss'] = ve_loss.cpu()
                forward_result['label'] = state_lab.cpu()
                forward_result['pred'] = predict.cpu()
                forward_result['batch'] = batch_size
                self.train_metric.update(forward_result)

                batch_param = BatchEndParam(epoch=nepoch, part=self.npart, nbatch= bid, nsample=batch_size, \
                    nframes=fea_frames, rank=self.args.rank, eval_metric=self.train_metric)
                for callback in self.callbacks:
                    callback(batch_param)

                if self.args.use_bmuf:
                    if (bid+1) % self.args.param_sync_freq == 0:
                        for param in self.train_params:
                            dist.all_reduce(param.data)
                            param.data /= float(self.args.distributed_world_size)
                        self.optimizer.bmuf_step()

                if bid % self.args.cvtest_freq == 0:
                    # self.devset = build_data_dev(lmdb_file=self.args.dev, lmdb_key=self.args.dev_key, \
                        # mean_file=self.args.train_mean[self.npart], variance_file=self.args.train_variance[self.npart], chunk=self.args.dev_chunk_size, \
                        # batch=self.args.batch_p*self.args.batch_k, rank=0, world=1, \
                        # fea_dim=self.args.fea_dim, start_label=self.args.dev_start_labels, speakers=self.args.dev_speakers,  margin=self.args.margin, \
                        # num_sort=self.args.num_sort, pad=self.args.padding, shuffle=False, tf_mask=0, num_thread=self.args.thread, dev=True)
                    # self.eval_epoch(nepoch, bid, self.devset)
                    if self.args.rank == 0: 
                        modelfile = '%s/model%d_%d.model'%(self.args.model_dir, nepoch, bid)
                        self.save_checkpoint(self.train_module, None, modelfile)
                    
                    self.train_module.train()

            except RuntimeError as e:
                if 'out of memory' in str(e):
                    all_gather_info('rank:%s, run out of memory, skipping this batch!!'%self.args.rank, self.args.gpu)
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                    continue
                elif 'illegal memory access' in str(e):
                    all_gather_info('rank:%s, encounter an illegal memory access, skipping this batch!!'%self.args.rank, self.args.gpu)
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                    continue
                else:
                    all_gather_info('rank:%s, %s'%(self.args.rank, str(e)), self.args.gpu)
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                    continue

            if self.args.warmup:
                if bid == (self.args.warmup_bacthnum+1):
                    break

        if self.args.rank == 0:
            if self.args.warmup:
                logging.info('warmup initail model end, cost %d seconds.', time.time()-tic)
            else:
                logging.info('TRAIN epoch%d, cost %d seconds, %s', nepoch, time.time()-tic, self.train_metric.report())

    def eval_classify_spk(self, nepoch, nbid, devset):

        self.train_module.eval()
        devset.reset()
        tic = time.time()

        with torch.no_grad():
            dict_of_speakerVector = defaultdict(list)
            dict_of_speakerVector_mean = defaultdict(list)
            dict_of_speakerVector_mean_enroll = defaultdict(list)
            dict_of_speakerVector_mean_enrollM = defaultdict(list)

            for bid, batch_data in enumerate(devset):
                #bid_start_time = time.time()
                src          = torch.from_numpy(batch_data['src'])
                state_lab    = torch.from_numpy(batch_data['state_lab']).long()
                src_mask     = torch.from_numpy(batch_data['src_mask'])
                large_margin = torch.from_numpy(batch_data['large_margin'])
                wave_name    = batch_data['wav_names']
                key_name     = batch_data['key_names']
                #bid_end_time = time.time()
                #logging.info('dev nepoch %s bid %s rank %s get data cost %ss : %s' \
                #    % (nepoch, nbid, self.args.rank, (bid_end_time-bid_start_time), key_name))

                batch_size, fea_frames, fea_dim = src.size()
                #continue
                # import pdb
                # pdb.set_trace()
                #for ibatch in list(range(batch_size)):
                #    logging.info('dev nepoch %s bid %s rank %s get npart %s data : %s' % (nepoch, nbid, self.args.rank, self.npart, key_name[ibatch]))
                
                params = [p.cuda(self.args.gpu) for p in (src, state_lab)]
                anchor_sv, predict, ce_loss = self.train_module(*params, fea_frames, 0)
                
                temp_zip = zip(wave_name, anchor_sv)
                for k, v in temp_zip:
                    #wav_name = re.split(r'[\[\%\]\.]', k)[0]
                    if re.search(r'reverb',k):
                        continue
                    wav_name =  re.split(r'[\[\/\]\.]', k)[-3]
                    wav_name = wav_name.replace('0##','')
                    value = v.data.cpu().numpy()
                    value = value.reshape(-1, 1)
                    dict_of_speakerVector[wav_name].append(value)
    def eval_epoch(self, nepoch, nbid, devset):

        self.train_module.eval()
        devset.reset()
        tic = time.time()

        with torch.no_grad():
            dict_of_speakerVector = defaultdict(list)
            dict_of_speakerVector_mean = defaultdict(list)
            dict_of_speakerVector_mean_enroll = defaultdict(list)
            dict_of_speakerVector_mean_enrollM = defaultdict(list)

            for bid, batch_data in enumerate(devset):
                #bid_start_time = time.time()
                src          = torch.from_numpy(batch_data['src'])
                state_lab    = torch.from_numpy(batch_data['state_lab']).long()
                src_mask     = torch.from_numpy(batch_data['src_mask'])
                large_margin = torch.from_numpy(batch_data['large_margin'])
                wave_name    = batch_data['wav_names']
                key_name     = batch_data['key_names']
                #bid_end_time = time.time()
                #logging.info('dev nepoch %s bid %s rank %s get data cost %ss : %s' \
                #    % (nepoch, nbid, self.args.rank, (bid_end_time-bid_start_time), key_name))

                batch_size, fea_frames, fea_dim = src.size()
                #continue
                # import pdb
                # pdb.set_trace()
                #for ibatch in list(range(batch_size)):
                #    logging.info('dev nepoch %s bid %s rank %s get npart %s data : %s' % (nepoch, nbid, self.args.rank, self.npart, key_name[ibatch]))
                
                params = [p.cuda(self.args.gpu) for p in (src, state_lab)]
                anchor_sv, predict, ce_loss = self.train_module(*params, fea_frames, 0)
                
                temp_zip = zip(wave_name, anchor_sv)
                for k, v in temp_zip:
                    #wav_name = re.split(r'[\[\%\]\.]', k)[0]
                    if re.search(r'reverb',k):
                        continue
                    wav_name =  re.split(r'[\[\/\]\.]', k)[-3]
                    wav_name = wav_name.replace('0##','')
                    value = v.data.cpu().numpy()
                    value = value.reshape(-1, 1)
                    dict_of_speakerVector[wav_name].append(value)

        for k in dict_of_speakerVector:
            count = 0
            vector_sum = np.zeros([512, 1])
            for item in dict_of_speakerVector[k]:
                vector_sum += item
                count += 1
            dict_of_speakerVector_mean[k] = vector_sum / count

        # enroll vectors
        # for k in dict_of_speakerVector_mean:
        #     feaname = re.split(r'[\_]', k)
        #     wav_name = feaname[0]
        #     key_flag = feaname[1]
        #     value = dict_of_speakerVector_mean[k]
        #     if operator.eq(key_flag, "enroll"):
        #         dict_of_speakerVector_mean_enroll[wav_name].append(value)

        # for k in dict_of_speakerVector_mean_enroll:
        #     count = 0
        #     vector_sum = np.zeros([512, 1])
        #     for item in dict_of_speakerVector_mean_enroll[k]:
        #         vector_sum += item
        #         count += 1
        #     dict_of_speakerVector_mean_enrollM[k] = vector_sum / count

        true_score = []
        false_score = []

        with open(self.args.test_list, 'r') as file_list:
            for line in file_list:
                line = line.strip()
                if not line:
                    continue
                splited_line = line.split()
                enroll = splited_line[0]
                test = splited_line[1]
                key_flag = splited_line[2]

                if enroll in dict_of_speakerVector_mean.keys() and test in dict_of_speakerVector_mean.keys():
                    vector1 = dict_of_speakerVector_mean[enroll].reshape(-1)
                    vector2 = dict_of_speakerVector_mean[test].reshape(-1)
                    score = np.dot(vector1,vector2)/(np.linalg.norm(vector1)*(np.linalg.norm(vector2))) 
                    if operator.eq(key_flag, "True"):
                        true_score.append(score)
                    else:
                        false_score.append(score)
        file_list.close()

        #import pdb
        #pdb.set_trace()
        EER_list = []
        Max_score = max(max(true_score), max(false_score))
        Min_score = min(min(true_score), min(false_score))
        step = 0.001
        for threshold in np.arange(Min_score, Max_score, step):
            false_positive_list = [i for i in false_score if i > threshold]
            false_negtive_list = [i for i in true_score if i < threshold]

            false_positive = len(false_positive_list)
            false_negtive = len(false_negtive_list)
            target_nums = len(true_score)
            nontarget_nums = len(false_score)

            false_reject_rate = float(false_negtive) / target_nums
            false_alarm_rate = float(false_positive) / nontarget_nums
            if abs(false_reject_rate - false_alarm_rate) < 0.01:
                eer = (false_reject_rate + false_alarm_rate) / 2.0
                EER_list.append(eer)

        try:
            EER = min(EER_list)
        except:
            EER = 0.5
        toc = time.time()
        if self.args.rank == 0:
            logging.info('DEV epoch %d bid %d cost %d seconds, EER=%f', nepoch, nbid, toc-tic, EER)