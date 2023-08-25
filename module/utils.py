from collections import namedtuple
import logging
import time
import numpy
import numpy as np
import torch
from torch.optim import Optimizer
import pickle
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level = logging.DEBUG,
                    format = '%(asctime)s[%(levelname)s] %(name)s -%(message)s',
                    )
import math
# import horovod.torch as hvd
# from horovod.torch.mpi_ops import allreduce, allreduce_async, allreduce_, allreduce_async_
# from horovod.torch.mpi_ops import poll, synchronize
import torch.distributed as dist

class Metric(object):
    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        pass

    def update(self, params):
        pass

    def states(self):
        return {}

    def report(self):
        states = self.states()
        results = ', '.join(['{}={:.5f}'.format(k,v) for k,v in states.items()])
        return results


class CombineMetric(Metric):
    def __init__(self, metrics):
        self.metrics = metrics
        super(CombineMetric, self).__init__('combine')


    def reset(self):
        for metric in self.metrics:
            metric.reset()

    def update(self, *args):
        for metric in self.metrics:
            metric.update(*args)

    def states(self):
        result = {}
        for metric in self.metrics:
            result.update(metric.states())
        return result


class CELossMetric(Metric):
    def __init__(self):
        self.num_samples = 0.
        self.sum_loss = 0.
        super(CELossMetric,self).__init__('ce-loss')

    def reset(self):
        self.num_samples= 0.
        self.sum_loss = 0.

    def update(self, params):
        samples = params['batch']
        loss = params['ce_loss']
        self.num_samples += samples
        self.sum_loss += loss.item()

    def states(self):
        return {self.name:self.sum_loss/self.num_samples}


class VELossMetric(Metric):
    def __init__(self):
        self.num_samples = 0.
        self.sum_loss = 0.
        super(VELossMetric,self).__init__('ve-loss')

    def reset(self):
        self.num_samples= 0.
        self.sum_loss = 0.

    def update(self, params):
        samples = params['batch']
        loss = params['ve_loss']
        self.num_samples += samples
        self.sum_loss += loss.item()

    def states(self):
        return {self.name:self.sum_loss/self.num_samples}


class AccMetric(Metric):
    def __init__(self):
        self.num_samples = 0.
        self.sum_acc = 0.
        super(AccMetric,self).__init__('acc')

    def reset(self):
        self.num_samples = 0.
        self.sum_acc = 0.

    def update(self, params):
        samples = params['batch']
        label = params['label']
        predict = params['pred']
        label = label.view(-1)
        correct = predict.eq(label).float()
        if not hasattr(correct, 'sum'):
            correct = correct.cpu()
        
        self.num_samples += samples
        self.sum_acc += correct.sum().item()
    def states(self):
        return {self.name:self.sum_acc/self.num_samples}


BatchEndParam = namedtuple('BatchEndParams',
                           ['epoch',
                            'part',
                            'nbatch',
                            'nsample',
                            'nframes',
                            'rank',
                            'eval_metric'])

class Speedometer(object):
    """Calculate and log training speed periodically.

    Parameters
    ----------
    batch_size: int
        batch_size of data.
    frequent: int
        How many batches between calculations.
        Defaults to calculating & logging every 50 batches.
    """
    def __init__(self, frequent=50):
        self.frequent = frequent
        self.init = False
        self.tic = 0
        self.last_count = 0
        self.nsamples = 0
        self.nframes = 0

        try:
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
        except:
            self.world_size = 1
            self.rank = 0

    def __call__(self, param):
        """Callback to Show speed."""
        count = param.nbatch
        if self.last_count > count:
            self.init = False
        self.last_count = count
        self.nsamples += param.nsample
        self.nframes += param.nframes

        if self.init:    
            if count % self.frequent == 0:
                try:
                    speed = self.nsamples / (time.time() - self.tic)
                    frame_speed = self.nframes / (time.time() - self.tic)
                    if param.eval_metric is not None:
                        if self.rank == 0:
                            logger.info('Rank[%d] Epoch[%d] Part[%d] Batch [%d] Speed: %.2f samples/sec, f-Speed: %d frames/sec, %s',
                                        param.rank, param.epoch, param.part, count, speed, frame_speed, param.eval_metric.report())
                    else:
                        if self.rank == 0:
                            logger.info("Rank[%d] Iter[%d] Part[%d] Batch [%d] Speed: %.2f samples/sec",
                                        param.rank, param.epoch, param.part, count, speed)
                    self.tic = time.time()
                    self.nsamples=0
                    self.nframes =0
                    param.eval_metric.reset()
                except:
                    pass
        else:
            self.init = True
            self.tic = time.time()
            self.nsamples = 0
            self.nframes = 0

class MasterOptimizer(object):
    def __init__():
        pass


class LRDecayOptimizer(object):
    def __init__(self, optimizer, init_lr=1e-4, min_lr= 1e-6,
                 decay_start = 40000, decay_step=10000, decay_ratio = 0.6):
        self._optimizer = optimizer
        self._base_lr = init_lr
        self._min_lr = min_lr
        self._decay_start = decay_start
        self._decay_step = decay_step
        self._decay_ratio = decay_ratio
        self._count = 0
        self._update_num = 0
        self._set_lr(init_lr)

    def state_dict(self):
        state_dict = self._optimizer.state_dict()
        state_dict['update_num'] = self._update_num
        return state_dict

    def load_state_dict(self, state_dict):
        self._update_num = state_dict['update_num']
        self._optimizer.load_state_dict(state_dict)

    def _set_lr(self, new_lr):
        logging.info('set learning rate to %f', new_lr)
        self._base_lr = new_lr
        for group in self._optimizer.param_groups:
            group['lr'] = new_lr

    def zero_grad(self):
        self._optimizer.zero_grad()

    def _update_lr(self):
        while self._update_num - self._count > self._decay_step:
            self._count += self._decay_step
            if self._count > self._decay_start:
                new_lr = self._base_lr* self._decay_ratio
                if new_lr < self._min_lr:
                    new_lr = self._min_lr
                self._set_lr(new_lr)

    def step(self, closure= None):
        self._update_num +=1
        self._optimizer.step(closure)
        self._update_lr()


class AccPerformance(object):
    def __init__(self,vocab, logfile= None):
        self._ignores= (vocab.pad_id, vocab.bos_id, vocab.eos_id)
        self._vocab = vocab
        self._log_file = None
        if logfile is not None:
            self._log_file = open(logfile, 'w')
        self._label_num = 0
        self._error_num = 0

    def reset(self):
        self._label_num = 0
        self._error_num = 0

    def _edit_dist(self, label, rec):
        dist_mat = numpy.zeros((len(label) +1, len(rec)+1), dtype= 'int32')
        dist_mat[0,:] = range(len(rec)+1)
        dist_mat[:,0] = range(len(label)+1)
        for i in range(1, len(label)+1):
            for j in range(1, len(rec)+1):
                hit_score = dist_mat[i-1, j-1] +(label[i-1] != rec[j-1])
                ins_score = dist_mat[i, j-1] +1
                del_score = dist_mat[i-1, j]+1
                dist_mat[i,j] = min(hit_score, ins_score, del_score)
        return len(label), dist_mat[len(label), len(rec)]

    def update(self, label, hyp):
        label = [l for l in label if l not in self._ignores]
        hyp = [ h for h in hyp if h not in self._ignores]
        lab_len, err = self._edit_dist(label, hyp)
        self._label_num += lab_len
        self._error_num += err
        wer = float(err)/ lab_len
        lab_str = ' '.join([self._vocab.get_word(w) for w in label])
        hyp_str = ' '.join([self._vocab.get_word(w) for w in hyp])
        if self._log_file:
            self._log_file.write('%s|||%s|||%f\n'%(lab_str, hyp_str, wer))
            self._log_file.flush()

    def get_performance(self):
        wer = float(self._error_num)/ self._label_num
        return wer



class BMUFAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, bm_lr=1.0, bm_mom=1.0):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay,
                        bm_lr=bm_lr, bm_mom=bm_mom)
        super(BMUFAdam, self).__init__(params, defaults)

    def _set_lr(self, new_lr):
        logging.info('set learning rate to %f', new_lr)
        self._base_lr = new_lr
        for group in self.param_groups:
            group['lr'] = new_lr

    def bmuf_step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                if 'param_buffer' not in state:
                    state['param_buffer'] = torch.zeros_like(p.data)
                    state['param_buffer'].copy_(p.data)
                if 'delta_buffer' not in state:
                    state['delta_buffer'] = torch.zeros_like(p.data)
                param_buffer, delta_buffer = state['param_buffer'], state['delta_buffer']
                Gt = p.data-param_buffer
                delta_buffer.mul_(group['bm_mom']).add_(Gt.mul_(group['bm_lr']))
                param_buffer.add_(delta_buffer)
                p.data.copy_(param_buffer)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss


class BMUFSGD(Optimizer):
    def __init__(self, params, lr=1e-2, momentum=0.9, dampening=0,
                 weight_decay=0, nesterov=True, bm_lr=1.0, bm_mom=1.0):
        defaults = dict(lr=lr, weight_decay=weight_decay,
                        momentum = momentum,
                        nesterov=nesterov, dampening=dampening,
                        bm_lr=bm_lr, bm_mom=bm_mom)
        super(BMUFSGD, self).__init__(params, defaults)

    def _set_lr(self, new_lr):
        # logging.info('set learning rate to %f', new_lr)
        self._base_lr = new_lr
        for group in self.param_groups:
            group['lr'] = new_lr

    def bmuf_step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                if 'param_buffer' not in state:
                    state['param_buffer'] = torch.zeros_like(p.data)
                    state['param_buffer'].copy_(p.data)
                if 'delta_buffer' not in state:
                    state['delta_buffer'] = torch.zeros_like(p.data)
                param_buffer, delta_buffer = state['param_buffer'], state['delta_buffer']
                Gt = p.data-param_buffer
                delta_buffer.mul_(group['bm_mom']).add_(Gt.mul_(group['bm_lr']))
                param_buffer.add_(delta_buffer)
                p.data.copy_(param_buffer)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(-group['lr'], d_p)

        return loss

def all_gather_info(info, gpu, des_rank=0, max_len=3000):

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    enc = pickle.dumps(info)
    info_size = len(enc)
    enc_info_size = pickle.dumps(info_size)
    assert info_size+len(enc_info_size) <= max_len

    cur_info_tensor = torch.ByteTensor(max_len).cuda(gpu)
    cur_info_tensor[:len(enc_info_size)] = torch.ByteTensor([enc_info_size]).cuda(gpu)
    cur_info_tensor[len(enc_info_size):len(enc_info_size)+info_size] = torch.ByteTensor([enc]).cuda(gpu)
    gather_info_tensor = [torch.ByteTensor(max_len).cuda(gpu) for _ in range(world_size)]

    dist.all_gather(tensor_list=gather_info_tensor, tensor=cur_info_tensor)

    if rank == des_rank:
        for i in range(world_size):
            rank_info_size = pickle.loads(bytes(gather_info_tensor[i][:len(enc_info_size)]))
            rank_info = pickle.loads(bytes(gather_info_tensor[i][len(enc_info_size):len(enc_info_size)+rank_info_size]))
            logging.info(rank_info)