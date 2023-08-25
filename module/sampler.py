import lmdb
import re
import torch
import random
import numpy as np
from collections import defaultdict
from .datum_pb2 import Datum
import threading

class LMDBParser(object):
	def __init__(self,lmdb_file,label_key_file,chunk_spk_num=100,spk_voices=5,is_train=True):
		self.lmdb_file = lmdb_file
		self.label_key_file = label_key_file
		self.label_keys = defaultdict(list)
		self.label_keys_remaining = defaultdict(list)
		self.chunk_spk = chunk_spk_num
		self.spk_voices = spk_voices
		self.is_train = is_train

	def read_label_keys(self):
		with open(self.label_key_file) as fin:
			for line in fin:
				array = re.split(r'\s+',line)
				self.label_keys[array[0]].append(array[1].encode())
		for label in self.label_keys.keys():
			if len(self.label_keys[label]) >= self.spk_voices:
				self.label_keys_remaining[label] = len(self.label_keys[label])

	def start(self):
		self.read_label_keys()
		self._lmdb_env = lmdb.open(self.lmdb_file,readonly=True,lock=False)
		self._lmdb_txn = self._lmdb_env.begin()

	def _thread_get_data(self):
		while 1:
			try:
				current_key = self.chunk_keys.pop()
				datum = Datum()
				value = self._lmdb_txn.get(current_key)
				datum.ParseFromString(value)
				feature_frames = int(datum.frame_length)
				frame_dim = int(datum.feature_dim)
				feature = np.fromstring(datum.data,dtype = np.float32).reshape(feature_frames,frame_dim)
				
				if self.is_train:
					feature_frames = int(feature_frames/3)
					feature = feature[0:feature_frames][:]

				label = datum.speaker_label
				wave_name = datum.wave_name

				self.features_length.append(feature_frames)
				self.frame_dim = frame_dim

				self.lock.acquire()
				self.features.append(feature)
				self.labels.append(label)
				self.wave_names.append(wave_name)
				self.lock.release()
			except:
				break

	def get_trunk_data(self):
		self.features = []
		self.labels = []
		self.wave_names = []
		self.features_length = []
		self.chunk_keys = []
		self.lock = threading.Lock()

		try:
			selected_chunk_labels = random.sample(self.label_keys_remaining.keys(),self.chunk_spk)
			for label in selected_chunk_labels:
				voices = self.label_keys[label]
				random.shuffle(voices)
				for i in range(self.spk_voices):
					self.chunk_keys.append(voices[i])
					self.label_keys_remaining[label] -= 1
				if self.label_keys_remaining[label] < self.spk_voices:
					self.label_keys_remaining.pop(label)
		except:
			raise StopIteration

		thread_list = []
		for i in range(64):#********threads number************
			t = threading.Thread(target=self._thread_get_data,args=[])
			thread_list.append(t)
			#print('thread')
			
		for t in thread_list:
			t.setDaemon(True)
			t.start()
			#print('thread start')

		for t in thread_list:
			t.join()
			#print('thread wait')

		max_frame = max(self.features_length)
		max_frame = (max_frame + 4 - 1) // 4 * 4#ensure max_frame is even number

		return self.features,self.labels,self.wave_names,max_frame,self.frame_dim

	def __del__(self):
		self._lmdb_env.close()
		#del self._lmdb_cursor
		del self._lmdb_txn

class  DataSampler(object):
	def __init__(self,lmdb_parser,batch_spk_num,voices_per_spk):
		self.lmdb_parser = lmdb_parser
		self.speaker_number = batch_spk_num
		self.voices_per_spk = voices_per_spk
		self.batch_size = batch_spk_num * voices_per_spk
		self.max_frames = 0
		self.feature_dim = 0

		self.pool_features = []
		self.pool_labels = []
		self.pool_wave_names = []
		self.groups = defaultdict(list)#{label:voices}

	def start(self):
		self.lmdb_parser.start()

	def get_next_pool(self):
		self.pool_features = []
		self.pool_labels = []
		self.pool_wave_names = []
		self.groups = defaultdict(list)
		self.pool_features,self.pool_labels,self.pool_wave_names,self.max_frames,self.feature_dim = self.lmdb_parser.get_trunk_data()

	def create_groups(self):
		group_samples = defaultdict(list)
		for idx,label in enumerate(self.pool_labels):
			group_samples[label].append(idx)

		label_to_remove = []
		for label in group_samples.keys():
			if len(group_samples[label]) < self.voices_per_spk:
				label_to_remove.append(label)

		for label in label_to_remove:
			group_samples.pop(label)

		return group_samples

	def __iter__(self):
		return self

	def __next__(self):
		# ensures there are enough classes to sample from
		while len(self.groups.keys()) < self.speaker_number:
#*************************************
			# import time
			# tic = time.time()
			# print('creat groups')
			self.get_next_pool()
			# toc = time.time()
			# print('cost time {}'.format(toc - tic))
			# print('number labels is {}'.format(len(self.pool_labels)))
			# print('number wave_names is {}'.format(len(self.pool_wave_names)))
			# print('number features is {}'.format(len(self.pool_features)))
			# print('number sentence_boundarys is {}'.format(len(self.pool_sentence_boundarys)))

#**************************************
			self.groups = self.create_groups()
		# shuffle samples within labels
		for label in self.groups.keys():
			random.shuffle(self.groups[label])
        # keep track of the number of samples left for each label
		group_samples_remaining = {}
		for label in self.groups.keys():
			group_samples_remaining[label] = len(self.groups[label])
		#select speakers
		group_labels = list(self.groups.keys())
		selected_labels = torch.multinomial(torch.ones(len(group_labels)),self.speaker_number).tolist()

		features = np.zeros((self.batch_size,self.max_frames,self.feature_dim),dtype='float32')
		feature_mask = np.zeros((self.batch_size,self.max_frames,self.feature_dim),dtype='float32')
		sentence_mask = np.zeros((self.batch_size,self.max_frames,),dtype='float32')
		labels = np.zeros((self.batch_size,1),dtype='float32')

		#sample data
		curr_spk_num = 0
		for i in selected_labels:
			label = group_labels[i]
			voices = self.groups[label]
			curr_spk_num += 1
			for j in range(self.voices_per_spk):
				sample_idx = len(voices) - group_samples_remaining[label]
				sample_idx = voices[sample_idx]
				group_samples_remaining[label] -= 1
				frame_length = self.pool_features[sample_idx].shape[0]
				features[(curr_spk_num-1)*self.voices_per_spk+j,0:frame_length,:] = self.pool_features[sample_idx]
				feature_mask[(curr_spk_num-1)*self.voices_per_spk+j,0:frame_length,:] = 1
				labels[(curr_spk_num-1)*self.voices_per_spk+j,0] = self.pool_labels[sample_idx]

			if group_samples_remaining[label] < self.voices_per_spk:
				self.groups.pop(label)

		tensor_features = torch.Tensor(features)
		tensor_feature_mask = torch.Tensor(feature_mask)
		tensor_labels = torch.Tensor(labels).long()

		return tensor_features,tensor_feature_mask,tensor_labels

def train_data_sampler(lmdb_file,label_key_file,chunk_spk_num,batch_spk_num,spk_voices,is_train):
	lmdb_parser = LMDBParser(lmdb_file,label_key_file,chunk_spk_num,spk_voices,is_train)
	batch_data = DataSampler(lmdb_parser,batch_spk_num,spk_voices)
	return batch_data