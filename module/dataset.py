import lmdb
from .datum_pb2 import Datum
import numpy as np
import random
import math
#import multiprocessing as mp
from threading import Thread
from queue import Queue
from collections import namedtuple, defaultdict

import logging
import os
import time
import warnings

class DataSet(object):
    def __init__(self, lmdb_file, lmdb_key, mean_file, variance_file, chunk, batch, rank, world, fea_dim, start_label, speakers, margin, num_sort, pad, shuffle, tf_mask, num_thread, dev):
        self._lmdb_file = lmdb_file
        self._lmdb_key = lmdb_key
        self._mean_file = np.loadtxt(mean_file, skiprows=0, dtype='float32')
        self._variance_file = np.loadtxt(variance_file, skiprows=0, dtype='float32')
        self._num_thread = num_thread
        self._chunk = chunk
        self._batch = batch
        self._num_sort = num_sort
        self._pad = pad
        self._fea_dim = fea_dim
        self._start_label = start_label
        self._speakers = speakers
        self._margin = margin
        self._shuffle = shuffle
        self._tf_mask = tf_mask
        self._dev = dev

        self._lmdb_key = self._read_key(lmdb_key, rank, world)
        self._exit_code = '00000000000000000'
        self._num_exit_code = 0
        self._multi_task = list()

        #self._mean_file     = np.tile(self._mean_file, (3, 1))
        #self._variance_file = np.tile(self._variance_file, (3, 1))
        #np.savetxt("mean_file.txt", self._mean_file)
        #np.savetxt("variance_file.txt", self._variance_file)

    def reset(self):
        '''
        self._data_pool = mp.Queue(8)
        self._global_queue = mp.Queue(8)
        chunk_id = self._get_chunk()
        self._multi_task = [mp.Process(target=self._worker, daemon=True, \
                            args=(self._global_queue, self._lmdb_file, chunk_id[ii])) \
                            for ii in range(self._num_thread)]
        self._multi_task.append(mp.Process(target=self._package, daemon=True, \
                                args=(self._global_queue, self._data_pool)))
        '''
        self._data_pool = Queue(5)
        self._global_queue = Queue(5)
        chunk_id = self._get_chunk()
        self._multi_task = [Thread(target=self._worker, daemon=True, \
                            args=(self._global_queue, self._lmdb_file, chunk_id[ii])) \
                            for ii in range(self._num_thread)]
        self._multi_task.append(Thread(target=self._package, daemon=True, \
                                args=(self._global_queue, self._data_pool)))
        for task in self._multi_task:
            task.start()

    def _worker(self, queue, lmdb_file, lmdb_key):
        while True:
            if os.path.exists(lmdb_file):
                break
            else:
                print('Error-2: lmdb={} not exist, the read thread retries in 5 seconds !'.format(lmdb_file))
                time.sleep(5)

        lmdb_env = lmdb.open(lmdb_file, readonly=True, lock=False)
        lmdb_cursor = lmdb_env.begin().cursor()
        for chunk in lmdb_key:
            for key in chunk:
                while not lmdb_cursor.set_key(key):
                    print('Error-3: key={} not exist, the read thread retries in 5 seconds !'.format(key))
                    time.sleep(5)
 
                queue.put((lmdb_cursor.key(), lmdb_cursor.value()))
                #print('put key {}'.format(key))

        queue.put(self._exit_code)
        lmdb_env.close()

    def _tf_masking(self, length):
        F_max_mask = int(self._fea_dim/4)
        T_max_mask = int(length/10)

        F_mask = random.randint(0, F_max_mask)
        F0_mask = random.randint(0, self._fea_dim-F_mask)
        T_mask = random.randint(0, T_max_mask)
        T0_mask = random.randint(0, length-T_mask)
        data_tf_mask = np.ones((length, self._fea_dim), dtype='float32')
        data_tf_mask[T0_mask:T0_mask+T_mask, F0_mask:F0_mask+F_mask] = 0.0

        return data_tf_mask

    def _padding(self, batch):
        batch_data = dict()
        batch_size = len(batch)
        max_frame = max(batch, key=lambda x:x[0])[0]
        max_frame = ((max_frame + self._pad - 1)//self._pad) * self._pad

        batch_wavname = list()
        batch_keyname = list()
        
        np_state_lab = np.zeros((batch_size, 1), dtype='int32')
        np_large_margin = np.zeros((batch_size, self._speakers), dtype='float32')

        # if self._dev:
        #     np_input_fea = np.zeros((batch_size, 3*max_frame, self._fea_dim), dtype='float32')
        #     np_src_mask  = np.zeros((batch_size, 3*max_frame), dtype='float32')
        # else:
        #     np_input_fea = np.zeros((batch_size, max_frame, self._fea_dim), dtype='float32')
        #     np_src_mask  = np.zeros((batch_size, max_frame), dtype='float32')
        np_input_fea = np.zeros((batch_size, max_frame, self._fea_dim), dtype='float32')
        np_src_mask  = np.zeros((batch_size, max_frame), dtype='float32')

        #print('packind batch start ...')
        # catch warnings
        #warnings.filterwarnings('error')
        for i, (length, key, datum) in enumerate(batch):
            #logging.info("now packing key : %s"%(key.decode()))
            feature = np.fromstring(datum.data, dtype=np.float32).reshape(length, self._fea_dim)
            # check data
            if (np.isnan(np.max(feature))) or (np.isnan(np.min(feature))):
                logging.info("wrong data key : %s has nan !!"%(key.decode()))
                continue
            if (np.isinf(np.max(feature))) or (np.isinf(np.min(feature))):
                logging.info("wrong data key : %s has inf !!"%(key.decode()))
                continue
            if (np.max(feature) > 100) or (np.min(feature) < -100):
                logging.info("wrong data key : %s has bigger data !!"%(key.decode()))
                continue

            #np.savetxt("feature.txt", feature)
            #if self._dev:
                #np.savetxt("feature.txt", feature)
                #feature = np.tile(feature, (3, 1))
                #length = 3*length
                #np.savetxt("feature_mv.txt", feature)
                #os._exit(0)
            
            feature = (feature-self._mean_file)*self._variance_file
            #np.savetxt("feature_mv.txt", feature)
            #os._exit(0)

            if self._tf_mask:
                data_length = int(length/3)

                anchor_mask  = self._tf_masking(data_length)
                postive_mask = self._tf_masking(data_length)
                negative_mask = self._tf_masking(data_length)

                #import pdb
                #pdb.set_trace()

                data_tf_mask = np.vstack((anchor_mask, postive_mask, negative_mask))
                feature = feature * data_tf_mask

            target = datum.speaker_label-self._start_label
            if target >= self._speakers or target < 0:
                print('Error-5: label={} of {} is exceed target groups !'.format(datum.speaker_label, datum.wave_name.decode()))
                continue

            np_input_fea[i][:length][:] = feature
            np_state_lab[i]= target
            np_src_mask[i][:length] = 1.
            np_large_margin[i][int(np_state_lab[i])] = self._margin
            batch_wavname.append(datum.wave_name.decode())
            batch_keyname.append(key.decode())
            #print('batch datum %s' % (key.decode()))

        batch_data['src']            = np_input_fea
        batch_data['state_lab']      = np_state_lab
        batch_data['src_mask']       = np_src_mask
        batch_data['large_margin']   = np_large_margin
        batch_data['wav_names']      = batch_wavname
        batch_data['key_names']      = batch_keyname

        return batch_data

    def _create_batch(self, data_pool, out_queue):
        #data_pool =  sorted(data_pool, key=lambda x:x[0])
        if self._shuffle:
            random.shuffle(data_pool)

        batch_pool = list()
        idx_batch = 0
        batch = list()
        for data in data_pool:
            #print('create_batch data %s' % (data[1]))
            idx_batch += 1
            if idx_batch <= self._batch:
                batch.append(data)
            elif batch != []:
                batch_pool.append(batch)
                batch = []
                batch.append(data)
                idx_batch = 1

        if len(batch) > 0:
            batch_pool.append(batch)
            batch = []

        if self._shuffle:
            random.shuffle(batch_pool)
        for value in batch_pool:
            out_queue.put(self._padding(value))

        return batch

    def _package(self, in_queue, out_queue):
        # print('package-pid:', os.getpid())
        warnings.filterwarnings('error')
        data_pool = list()
        idx = 0
        while True:
            key_value = in_queue.get()
            if key_value == self._exit_code:
                self._num_exit_code += 1
                if self._num_exit_code == self._num_thread:
                    break
            else:
                #print('get_data %s' % (key_value[0])) 
                try:                              
                    idx += 1
                    datum = Datum()
                    datum.ParseFromString(key_value[1])
                    frame_len = int(datum.frame_length)
                    data_pool.append((frame_len, key_value[0], datum))
                    if idx == self._num_sort:
                        self._create_batch(data_pool, out_queue)
                        idx = 0
                        data_pool.clear()
                except:
                    print('cant parse data %s' % (key_value[0]))


        if len(data_pool) > 0:
            #print('remain pool data start ...')
            self._create_batch(data_pool, out_queue)

        out_queue.put(self._exit_code)


    def _get_chunk(self):
        chunk_id = [self._lmdb_key[ii:ii+self._chunk] for ii in range(0, len(self._lmdb_key), self._chunk)]
        if self._shuffle:
            random.shuffle(chunk_id)

        chunk_num = len(chunk_id)
        if chunk_num < self._num_thread:
            self._num_thread = 1
            return [chunk_id]
        step = math.ceil(chunk_num / self._num_thread)
        return [chunk_id[ii:ii+step] for ii in range(0, len(chunk_id), step)]

    def _read_key(self, lmdb_key, rank, world):
        while True:
            if os.path.exists(lmdb_key):
                break
            else:
                print('Error-1: {} not exist, the read thread retries in 5 seconds !'.format(lmdb_key))
                time.sleep(5)

        with open(lmdb_key, 'r') as fin:
            lmdb_key = [line.strip().encode() for line in fin]
            world_size = math.ceil(len(lmdb_key) / world)
            return lmdb_key[rank*world_size : (rank+1)*world_size]


    def __iter__(self):
        return self

    def __next__(self):
        try:
            #batch_data = self._data_pool.get(timeout=100)
            batch_data = self._data_pool.get()
            if batch_data == self._exit_code:
                #print('info: read finished !')
                raise StopIteration
            else:
                return batch_data
        except:
            #print('Error-4: Unable to read data, the read thread retries in 100 seconds !')
            raise StopIteration



if __name__ == '__main__':
    train = '/lustre2/sre/jinli/20190115Tele_HaiLiang_XJ1/2.train_cnn/test_data/data'
    key = '/lustre2/sre/jinli/20190115Tele_HaiLiang_XJ1/2.train_cnn/test_data/keys/total.key'
    rank = 0
    world= 1
    dataset = DataSet(train, key, rank, world)
    import time

    dataset.reset()

    for bid, data in enumerate(dataset):
        print(data['src'].shape)
        if bid == 100:
            print('break')
            break
    sum = 0
    print('here')
    dataset.reset()
    print('here-dataset')
    for bid, data in enumerate(dataset):
        sum += data['src'].shape[0]
        # print(sum)
    print('sum=',sum)
    # print(dataset._debug)
