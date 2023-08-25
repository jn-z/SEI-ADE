import numpy as np
import math
import torch
from torch.utils import data
import h5py
import random
from scipy.io import wavfile
from collections import defaultdict
from random import randint
import pdb
from scipy.fftpack import fft,ifft


class ForwardLibriSpeechRawXXreverseDataset(data.Dataset):
    def __init__(self, raw_file, list_file):
        """ raw_file: train-clean-100.h5
            list_file: list/training.txt
            audio_window: 20480
        """
        self.raw_file = raw_file
        self.utts = []

        with open(list_file) as f:
            temp = f.readlines()
        temp = [x.strip() for x in temp]

        self.h5f = h5py.File(self.raw_file, 'r')
        for i in temp:  # sanity check
            utt_len = self.h5f[i].shape[0]
            self.utts.append(i)

    def __len__(self):
        """Denotes the total number of utterances
        """
        return len(self.utts)

    def __getitem__(self, index):
        utt_id = self.utts[index]  # get the utterance id
        original = self.h5f[utt_id][:]

        return utt_id, self.h5f[utt_id][:], original[::-1].copy()


class ForwardLibriSpeechReverseRawDataset(data.Dataset):
    def __init__(self, raw_file, list_file):
        """ raw_file: train-clean-100.h5
            list_file: list/training.txt
            audio_window: 20480
        """
        self.raw_file = raw_file
        self.utts = []

        with open(list_file) as f:
            temp = f.readlines()
        temp = [x.strip() for x in temp]

        self.h5f = h5py.File(self.raw_file, 'r')
        for i in temp:  # sanity check
            utt_len = self.h5f[i].shape[0]
            self.utts.append(i)

    def __len__(self):
        """Denotes the total number of utterances
        """
        return len(self.utts)

    def __getitem__(self, index):
        utt_id = self.utts[index]  # get the utterance id
        original = self.h5f[utt_id][:]

        return utt_id, original[::-1].copy()  # reverse


class ForwardLibriSpeechRawDataset(data.Dataset):
    def __init__(self, raw_file, list_file):
        """ raw_file: train-clean-100.h5
            list_file: list/training.txt
            audio_window: 20480
        """
        self.raw_file = raw_file
        self.utts = []

        with open(list_file) as f:
            temp = f.readlines()
        temp = [x.strip() for x in temp]

        self.h5f = h5py.File(self.raw_file, 'r')
        for i in temp:  # sanity check
            utt_len = self.h5f[i].shape[0]
            self.utts.append(i)

    def __len__(self):
        """Denotes the total number of utterances
        """
        return len(self.utts)

    def __getitem__(self, index):
        utt_id = self.utts[index]  # get the utterance id

        return utt_id, self.h5f[utt_id][:]


class ReverseRawDataset(data.Dataset):
    def __init__(self, raw_file, list_file, audio_window):
        """ RawDataset trained reverse;
            raw_file: train-clean-100.h5
            list_file: list/training.txt
            audio_window: 20480
        """
        self.raw_file = raw_file
        self.audio_window = audio_window
        self.utts = []

        with open(list_file) as f:
            temp = f.readlines()
        temp = [x.strip() for x in temp]

        self.h5f = h5py.File(self.raw_file, 'r')
        for i in temp:  # sanity check
            utt_len = self.h5f[i].shape[0]
            if utt_len > 10:
                self.utts.append(i)
        """
        with open(index_file) as f:
            content = f.readlines()
        content = [x.strip() for x in content]
        self.spk2idx = {}
        for i in content:
            spk = i.split(' ')[0]
            idx = i.split(' ')[1]
            self.spk2idx[spk] = int(idx)
        """

    def __len__(self):
        """Denotes the total number of utterances
        """
        return len(self.utts)

    def __getitem__(self, index):
        utt_id = self.utts[index]  # get the utterance id
        utt_len = self.h5f[utt_id].shape[0]  # get the number of data points in the utterance
        index = np.random.randint(
            utt_len - self.audio_window + 1)  # get the index to read part of the utterance into memory
        # speaker = utt_id.split('-')[0]
        # label   = self.spk2idx[speaker]

        original = self.h5f[utt_id][index:index + self.audio_window]
        return original[::-1].copy()  # reverse


class ForwardDatasetSITWSilence(data.Dataset):
    ''' dataset for forward passing sitw without vad '''

    def __init__(self, wav_file):
        """ wav_file: /export/c01/jlai/thesis/data/sitw_dev_enroll/wav.scp
        """
        self.wav_file = wav_file

        with open(wav_file) as f:
            temp = f.readlines()
        self.utts = [x.strip().split(' ')[0] for x in temp]
        self.wavs = [x.strip().split(' ')[1] for x in temp]

    def __len__(self):
        """Denotes the total number of utterances
        """
        return len(self.utts)

    def __getitem__(self, index):
        wav_path = self.wavs[index]  # get the wav file path
        fs, data = wavfile.read(wav_path)

        return self.utts[index], data


class ForwardDatasetSwbdSreSilence(data.Dataset):
    ''' dataset for forward passing swbd_sre or sre16 without vad '''

    def __init__(self, wav_dir, scp_file):
        """ wav_dir: /export/c01/jlai/thesis/data/swbd_sre_combined/wav/
            list_file: /export/c01/jlai/thesis/data/swbd_sre_combined/list/log/swbd_sre_utt.{1..50}.scp
        """
        self.wav_dir = wav_dir

        with open(scp_file) as f:
            temp = f.readlines()
        self.utts = [x.strip() for x in temp]

    def __len__(self):
        """Denotes the total number of utterances
        """
        return len(self.utts)

    def __getitem__(self, index):
        utt_id = self.utts[index]  # get the utterance id
        path = self.wav_dir + utt_id
        fs, data = wavfile.read(path)

        return utt_id, data


class RawDatasetSwbdSreOne(data.Dataset):
    ''' dataset for swbd_sre with vad ; for training cpc with ONE voiced segment per recording '''

    def __init__(self, raw_file, list_file):
        """ raw_file: swbd_sre_combined_20k_20480.h5
            list_file: list/training3.txt, list/val3.txt
        """
        self.raw_file = raw_file

        with open(list_file) as f:
            temp = f.readlines()
        all_utt = [x.strip() for x in temp]

        # dictionary mapping unique utt id to its number of voied segments
        self.utts = defaultdict(lambda: 0)
        for i in all_utt:
            count = i.split('-')[-1]
            utt_uniq = i[:-(len(count) + 1)]
            self.utts[utt_uniq] += 1  # count

    def __len__(self):
        """Denotes the total number of utterances
        """
        return len(self.utts)

    def __getitem__(self, index):
        utt_id = self.utts.keys()[index]  # get the utterance id
        count = self.utts[utt_id]  # number of voiced segments for the utterance id
        select = randint(1, count)
        h5f = h5py.File(self.raw_file, 'r')

        return h5f[utt_id + '-' + str(select)][:]


class RawDatasetSwbdSreSilence(data.Dataset):
    ''' dataset for swbd_sre without vad; for training cpc with ONE voiced/unvoiced segment per recording '''

    def __init__(self, raw_file, list_file, audio_window):
        """ raw_file: swbd_sre_combined_20k_20480.h5
            list_file: list/training2.txt, list/val2.txt
        """
        self.raw_file = raw_file
        self.audio_window = audio_window

        with open(list_file) as f:
            temp = f.readlines()
        self.utts = [x.strip() for x in temp]

    def __len__(self):
        """Denotes the total number of utterances
        """
        return len(self.utts)

    def __getitem__(self, index):
        utt_id = self.utts[index]  # get the utterance id
        h5f = h5py.File(self.raw_file, 'r')
        utt_len = h5f[utt_id].shape[0]  # get the number of data points in the utterance
        index = np.random.randint(
            utt_len - self.audio_window + 1)  # get the index to read part of the utterance into memory

        return h5f[utt_id][index:index + self.audio_window]


class RawDatasetSwbdSre(data.Dataset):
    ''' dataset for swbd_sre with vad ; for training cpc with ONE voiced segment per recording '''

    def __init__(self, raw_file, list_file):
        """ raw_file: swbd_sre_combined_20k_20480.h5
            list_file: list/training.txt
        """
        self.raw_file = raw_file

        with open(list_file) as f:
            temp = f.readlines()
        self.utts = [x.strip() for x in temp]

    def __len__(self):
        """Denotes the total number of utterances
        """
        return len(self.utts)

    def __getitem__(self, index):
        utt_id = self.utts[index]  # get the utterance id
        h5f = h5py.File(self.raw_file, 'r')

        return h5f[utt_id][:]


class RawDatasetSpkClass(data.Dataset):
    def __init__(self, raw_file, all_file, list_file, index_file, audio_window, frame_window):
        """ raw_file: train-clean-100.h5
            list_file: list/training.txt
            index_file: spk2idx
            audio_window: 20480
        """
        self.raw_file  = raw_file
        #self.audio_window = get_windows(raw_file,list_file)
        self.audio_window = audio_window
        self.frame_window = frame_window
        self.utts = []
        self.length = []
        self.bits = 2
        self.snr_range = (15, 30)
        self.clip_factors = [0.8, 0.8, 0.7]
        with open(list_file) as f:
            temp = f.readlines()
        with open(all_file) as f:
            all_temp = f.readlines()
        self.h5f = h5py.File(self.raw_file, 'r')
        import pdb
        #pdb.set_trace()
        for i in temp: # sanity check
            utt_len = self.h5f[i.strip()].shape[0]
            if utt_len > 64:
                self.utts.append(i.strip())
        for j in all_temp: # sanity check
            max_len = self.h5f[j.strip()].shape[0]
            self.length.append(max_len)
        #else:
        #print(i.strip())
        #self.utts = [x.strip() for x in temp]
        #pdb.set_trace()
        with open(index_file) as f:
            content = f.readlines()
        content = [x.strip() for x in content]
        self.spk2idx = {}
        for i in content:
            #pdb.set_trace()
            spk = i.split(' ')[0]
            #utt_len = self.h5f[spk].shape[0]
            idx = int(i.split(' ')[1])
            #if utt_len > 20480 and spk in temp:
            self.spk2idx[spk] = idx

    def __len__(self):
        """Denotes the total number of utterances
        """
        return len(self.utts)

    def __getitem__(self, index):
        max_len = 1280
        name_list = []#pdb.set_trace()
        yushu = max_len % self.audio_window
        if yushu == 0:
            re_length = max_len
        else:
            re_length = max_len + self.audio_window - yushu
        #dateset_rand = random.random()
        #if dateset_rand <= 0.3 :
        #   index = random.randint(0,1076)
        #elif dateset_rand >= 0.7 :
        #   index = random.randint(1076,3649)
        #else:
        #   index = random.randint(3649,47896)
        utt_id = self.utts[index].strip()
        #pdb.set_trace()
        utt_len = self.h5f[utt_id].shape[0]
        feature_data = self.h5f[utt_id]
        data_range = np.max(feature_data) - np.min(feature_data)
        #pdb.set_trace()
        norm_data = (feature_data - np.min(feature_data)) / data_range
        avg_data = np.mean(norm_data, axis=0)
        sigma = np.std(norm_data, axis=0)
        feature_data2 = (norm_data - avg_data) / sigma
        feature_data2 = np.resize(feature_data2, re_length) # add repeatedly from beginning to end
        feature_data2 = feature_data2[np.newaxis, :]
        wav = torch.from_numpy(feature_data2).float()
        if random.random() < -1.0:
             len_wav = wav.shape[1]
             noise = torch.randn(wav.shape)
             norm_constant = 2.0 ** (self.bits - 1)
             norm_wave = wav / norm_constant
             norm_noise = noise / norm_constant
             signal_power = torch.sum(norm_wave ** 2) / len_wav
             noise_power = torch.sum(norm_noise ** 2) / len_wav
             snr = np.random.randint(self.snr_range[0], self.snr_range[1])
             covariance = torch.sqrt((signal_power / noise_power) * 10 ** (- snr / 10))
             wav = wav + covariance * noise

        if random.random() < -1.0:
             wav_length = wav.shape[1]
             cf = random.choice(self.clip_factors)
             clipped_wav = torch.clamp(wav.view(-1), cf * torch.min(wav), cf * torch.max(wav))
             wav = clipped_wav.view(1, -1)

        feature_data2 = np.squeeze(wav)
		#import pdb
        #pdb.set_trace()
        speaker = utt_id.split(' ')[0]
        label = torch.tensor(self.spk2idx[speaker])
        speaker_name = speaker.strip()
        name_list.append(speaker_name)
        #if(utt_len > self.audio_window):
        #    index = np.random.randint(utt_len - self.audio_window + 1) # get the index to read part of the utterance into memory
        #    return h5f[utt_id][index:index + self.audio_window], label.repeat(self.frame_window)
        #else:

        return feature_data2, label.repeat(self.frame_window), name_list


class TorchDataSet(object):
    def __init__(self, raw_file, list_file, index_file, audio_window, frame_window, batch_size, chunk_num, dimension,
                 dev):
        self.raw_file = raw_file
        self.audio_window = audio_window
        self.frame_window = frame_window
        self.utts = []

        with open(list_file) as f:
            temp = f.readlines()
        self.h5f = h5py.File(self.raw_file, 'r')
        # pdb.set_trace()
        for i in temp:  # sanity check
            utt_len = self.h5f[i.strip()].shape[0]
            # if utt_len > 20480:
            self.utts.append(i.strip())
            # else:
            # print(i.strip())
        # self.utts = [x.strip() for x in temp]
        # pdb.set_trace()
        with open(index_file) as f:
            content = f.readlines()
        content = [x.strip() for x in content]
        self.spk2idx = {}
        for i in content:
            # pdb.set_trace()
            spk = i.split(' ')[0]
            # utt_len = self.h5f[spk].shape[0]
            idx = int(i.split(' ')[1])
            # if utt_len > 20480 and spk in temp:
            self.spk2idx[spk] = idx

        self._batch_size = batch_size
        self._chunck_num = chunk_num
        self._chunck_size = self._chunck_num * self._batch_size
        self._dimension = dimension
        self._dev = dev
        self.bits = 2
        self.snr_range = (5, 20)
        self.clip_factors = [0.65, 0.75, 0.85]
        # self._file_point = codecs.open(list_file, 'r', 'utf-8')
        # self._dataset = self._file_point.readlines()
        # self._file_point.close()
        random.shuffle(self.utts)

    def reset(self):
        random.shuffle(self.utts)

    def __iter__(self):
        data_size = len(self.utts)
        batch_data = []
        target_frames = []
        name_list = []
        max_frames = 0
        count = 0
        # feature_data = []
        fs = 30000000
        for ii in range(data_size):
            #pdb.set_trace()
            line = self.utts[ii].strip()
            # splited_line = line.split()
            # htk_feature = splited_line[0]
            target_label = int(self.spk2idx[line])
            # htk_file = HTKfile(htk_feature)
            lens = self.h5f[line].shape[0]
            feature_data = self.h5f[line][0:lens]
            feature_data = feature_data[np.newaxis, :]
            # wav_fft = fft(feature_data)  # 快速傅里叶变换
            # wav_fft[int(lens/2)+1:] = 0
            # base_wav = ifft(wav_fft)
            # t = 1j * 2 * math.pi/fs * np.ones_like(base_wav)
            # freq = random.randint(0,200) - 100
            # carriy = np.exp(t * freq)
            # base_wav = base_wav * carriy
            # feature_data = base_wav.real  # 获取实数部分
            wav = torch.from_numpy(feature_data).float()

            feature_data = np.squeeze(wav)
            zhengshu = lens // 10240
            weishu = lens % 10240
            # feature_data[len(feature_data):len(feature_data)+20480-weishu-1] = np.zeros(20480-weishu)
            # print(weishu)
            # print(zhengshu)
            # print(lens)
            # print(len(feature_data))
            # feature_data = feature_data.numpy()
            if lens < 5120:
                feature_data1 = np.append(feature_data, [0] * (10240 - weishu))
            else:
                feature_data1 = np.append(feature_data, feature_data[0:10240 - weishu])
            # print(len(feature_data1))
            feature_data1 = feature_data1[:].reshape(-1, 10240)

            # print(feature_data1.shape)
            # feature_data1 = feature_data1[:].reshape(-1,20480)
            file_name = line
            feature_data = feature_data1
            feature_frames = feature_data.shape[0]
            # print(feature_frames)
            # if feature_frames > 6000:   #\B4\F3\D3\DA6000֡\B3\A4\B5Ķ\AA\C6\FA
            # continue

            if feature_frames > max_frames:
                max_frames = feature_frames
            count += 1
            curr_feature = torch.Tensor(feature_data)
            means = curr_feature.mean(dim=0, keepdim=True)
            curr_feature_norm = curr_feature - means.expand_as(curr_feature)
            batch_data.append(curr_feature)
            # batch_data.append(curr_feature)
            target_frames.append(torch.Tensor([target_label, feature_frames]))
            name_list.append(file_name)

            # if (ii+1) % self._chunck_size == 0:
            if count % self._chunck_size == 0:
                chunk_size = len(batch_data)
                idx = 0
                data = torch.zeros(self._batch_size, max_frames, self._dimension)
                target = torch.zeros(self._batch_size, 2)
                utti = []
                for jj in range(chunk_size):
                    curr_data = batch_data[jj]
                    curr_tgt = target_frames[jj]
                    curr_utti = name_list[jj]
                    curr_frame = curr_data.size(0)

                    # data[idx,:curr_frame,:] = curr_data[:,:]
                    data_w = max_frames % curr_frame
                    data_z = max_frames // curr_frame
                    cc = 0
                    # print (data_w)
                    # print(data_z)
                    # print(max_frames)
                    # print(curr_frame)
                    for cc in range(data_z):
                        data[idx, curr_frame * cc:(cc + 1) * curr_frame, :] = curr_data[:, :]
                        # print (cc)
                    # print(cc)
                    if data_w > 0:
                        data[idx, curr_frame * (cc + 1):(cc + 1) * curr_frame + data_w, :] = curr_data[:data_w, :]

                    target[idx, :] = curr_tgt[:]
                    utti.append(curr_utti)
                    idx += 1

                    if idx % self._batch_size == 0:
                        idx = 0
                        yield data, target, utti

                max_frames = 0
                batch_data = []
                target_frames = []
                name_list = []
                count = 0

            else:
                pass

        chunk_size = len(batch_data)
        if self._dev:
            idx = 0
            data = torch.zeros(self._batch_size, max_frames, self._dimension)
            target = torch.zeros(self._batch_size, 2)
            utti = []
            for jj in range(chunk_size):
                curr_data = batch_data[jj]
                curr_tgt = target_frames[jj]
                curr_utti = name_list[jj]
                curr_frame = curr_data.size(0)

                data[idx, :curr_frame, :] = curr_data[:, :]
                target[idx, :] = curr_tgt[:]
                utti.append(curr_utti)
                idx += 1

                if idx % self._batch_size == 0:
                    idx = 0
                    yield data, target, utti
            if idx > 0:
                yield data[:idx], target[:idx], utti[:idx]


        elif chunk_size > self._batch_size:
            idx = 0
            data = torch.zeros(self._batch_size, max_frames, self._dimension)
            target = torch.zeros(self._batch_size, 2)
            utti = []
            for jj in range(chunk_size):
                curr_data = batch_data[jj]
                curr_tgt = target_frames[jj]
                curr_utti = name_list[jj]
                curr_frame = curr_data.size(0)

                data[idx, :curr_frame, :] = curr_data[:, :]
                target[idx, :] = curr_tgt[:]
                utti.append(curr_utti)
                idx += 1

                if idx % self._batch_size == 0:
                    idx = 0
                    yield data, target, utti


class TorchDataSet_train(object):
    def __init__(self, raw_file, list_file, audio_window, batch_size, chunk_num, dimension, dev):
        self.raw_file = raw_file
        self.audio_window = audio_window
        self.utts = []

        with open(list_file) as f:
            temp = f.readlines()
        self.h5f = h5py.File(self.raw_file, 'r')
        # pdb.set_trace()
        for i in temp:  # sanity check
            utt_len = self.h5f[i.strip()].shape[0]
            if utt_len > 100:
                self.utts.append(i.strip())
                # else:
                # print(i.strip())
                # self.utts = [x.strip() for x in temp]
                # pdb.set_trace()
                # with open(index_file) as f:
                # content = f.readlines()
                # content = [x.strip() for x in content]
                # self.spk2idx = {}
                # for i in content:
                # #pdb.set_trace()
                # spk = i.split(' ')[0]
                # #utt_len = self.h5f[spk].shape[0]
                # idx = int(i.split(' ')[1])
                # #if utt_len > 20480 and spk in temp:
                # self.spk2idx[spk] = idx

        self._batch_size = batch_size
        self._chunck_num = chunk_num
        self._chunck_size = self._chunck_num * self._batch_size
        self._dimension = dimension
        self._dev = dev
        # self._file_point = codecs.open(list_file, 'r', 'utf-8')
        # self._dataset = self._file_point.readlines()
        # self._file_point.close()
        random.shuffle(self.utts)

    def reset(self):
        random.shuffle(self.utts)

    def __iter__(self):
        data_size = len(self.utts)
        batch_data = []
        # target_frames = []
        name_list = []
        max_frames = 0
        count = 0
        # feature_data = []
        for ii in range(data_size):
            line = self.utts[ii].strip()
            # splited_line = line.split()
            # htk_feature = splited_line[0]
            # target_label = int(self.spk2idx[line])
            # htk_file = HTKfile(htk_feature)
            lens = self.h5f[line].shape[0]
            feature_data = self.h5f[line][0:lens]
            # print(feature_data)
            zhengshu = lens // 10240
            weishu = lens % 10240
            # feature_data[len(feature_data):len(feature_data)+20480-weishu-1] = np.zeros(20480-weishu)
            # print(weishu)
            # print(zhengshu)
            # print(lens)
            # print(len(feature_data))
            # feature_data = feature_data.numpy()
            if lens < 5120:
                feature_data1 = np.append(feature_data, [0] * (10240 - weishu))
            else:
                feature_data1 = np.append(feature_data, feature_data[0:10240 - weishu])
            # print(len(feature_data1))
            feature_data1 = feature_data1[:].reshape(-1, 10240)

            # print(feature_data1.shape)
            # feature_data1 = feature_data1[:].reshape(-1,20480)
            file_name = line
            feature_data = feature_data1
            feature_frames = feature_data.shape[0]
            # print(feature_frames)
            # if feature_frames > 6000:   #\B4\F3\D3\DA6000֡\B3\A4\B5Ķ\AA\C6\FA
            # continue

            if feature_frames > max_frames:
                max_frames = feature_frames
            count += 1
            curr_feature = torch.Tensor(feature_data)
            means = curr_feature.mean(dim=0, keepdim=True)
            curr_feature_norm = curr_feature - means.expand_as(curr_feature)
            batch_data.append(curr_feature)
            # batch_data.append(curr_feature)
            # target_frames.append(torch.Tensor([target_label, feature_frames]))
            name_list.append(file_name)

            # if (ii+1) % self._chunck_size == 0:
            if count % self._chunck_size == 0:
                chunk_size = len(batch_data)
                idx = 0
                data = torch.zeros(self._batch_size, max_frames, self._dimension)
                # target = torch.zeros(self._batch_size, 2)
                for jj in range(chunk_size):
                    curr_data = batch_data[jj]
                    # curr_tgt = target_frames[jj]
                    curr_frame = curr_data.size(0)

                    # data[idx,:curr_frame,:] = curr_data[:,:]
                    data_w = max_frames % curr_frame
                    data_z = max_frames // curr_frame
                    cc = 0
                    # print (data_w)
                    # print(data_z)
                    # print(max_frames)
                    # print(curr_frame)
                    for cc in range(data_z):
                        data[idx, curr_frame * cc:(cc + 1) * curr_frame, :] = curr_data[:, :]
                        # print (cc)
                    # print(cc)
                    if data_w > 0:
                        data[idx, curr_frame * (cc + 1):(cc + 1) * curr_frame + data_w, :] = curr_data[:data_w, :]

                    # target[idx,:] = curr_tgt[:]
                    idx += 1

                    if idx % self._batch_size == 0:
                        idx = 0
                        yield data

                max_frames = 0
                batch_data = []
                # target_frames = []
                name_list = []
                count = 0

            else:
                pass

        chunk_size = len(batch_data)
        if self._dev:
            idx = 0
            data = torch.zeros(self._batch_size, max_frames, self._dimension)
            # target = torch.zeros(self._batch_size, 2)
            for jj in range(chunk_size):
                curr_data = batch_data[jj]
                # curr_tgt = target_frames[jj]
                curr_frame = curr_data.size(0)

                data[idx, :curr_frame, :] = curr_data[:, :]
                # target[idx,:] = curr_tgt[:]
                idx += 1

                if idx % self._batch_size == 0:
                    idx = 0
                    yield data
            if idx > 0:
                yield data[:idx]


        elif chunk_size > self._batch_size:
            idx = 0
            data = torch.zeros(self._batch_size, max_frames, self._dimension)
            # target = torch.zeros(self._batch_size, 2)
            for jj in range(chunk_size):
                curr_data = batch_data[jj]
                # curr_tgt = target_frames[jj]
                curr_frame = curr_data.size(0)

                data[idx, :curr_frame, :] = curr_data[:, :]
                # target[idx,:] = curr_tgt[:]
                idx += 1

                if idx % self._batch_size == 0:
                    idx = 0
                    yield data


class RawXXreverseDataset(data.Dataset):
    ''' RawDataset but returns sequence twice: x, x_reverse '''

    def __init__(self, raw_file, list_file, audio_window):
        """ raw_file: train-clean-100.h5
            list_file: list/training.txt
            audio_window: 20480
        """
        self.raw_file = raw_file
        self.audio_window = audio_window
        self.utts = []

        with open(list_file) as f:
            temp = f.readlines()
        temp = [x.strip() for x in temp]

        self.h5f = h5py.File(self.raw_file, 'r')
        for i in temp:  # sanity check
            utt_len = self.h5f[i].shape[0]
            if utt_len > 1500:
                self.utts.append(i)

    def __len__(self):
        """Denotes the total number of utterances
        """
        return len(self.utts)

    def __getitem__(self, index):
        utt_id = self.utts[index]  # get the utterance id
        utt_len = self.h5f[utt_id].shape[0]  # get the number of data points in the utterance
        index = np.random.randint(
            utt_len - self.audio_window + 1)  # get the index to read part of the utterance into memory
        # speaker = utt_id.split('-')[0]
        # label   = self.spk2idx[speaker]

        original = self.h5f[utt_id][index:index + self.audio_window]
        return original, original[::-1].copy()  # reverse


class RawDataset(data.Dataset):
    def __init__(self, raw_file, list_file, audio_window):
        """ raw_file: train-clean-100.h5
            list_file: list/training.txt
            audio_window: 20480
        """
        # pdb.set_trace()
        self.raw_file = raw_file
        self.audio_window = audio_window
        self.utts = []

        with open(list_file) as f:
            temp = f.readlines()
        temp = [x.strip() for x in temp]

        self.h5f = h5py.File(self.raw_file, 'r')
        for i in temp:  # sanity check
            utt_len = self.h5f[i].shape[0]
            if utt_len > 1500:
                self.utts.append(i)
        """
        with open(index_file) as f:
            content = f.readlines()
        content = [x.strip() for x in content]
        self.spk2idx = {}
        for i in content:
            spk = i.split(' ')[0]
            idx = i.split(' ')[1]
            self.spk2idx[spk] = int(idx)
        """

    def __len__(self):
        """Denotes the total number of utterances
        """
        return len(self.utts)

    def __getitem__(self, index):
        # pdb.set_trace()
        utt_id = self.utts[index]  # get the utterance id
        utt_len = self.h5f[utt_id].shape[0]  # get the number of data points in the utterance
        index = np.random.randint(
            utt_len - self.audio_window + 1)  # get the index to read part of the utterance into memory
        # speaker = utt_id.split('-')[0]
        # label   = self.spk2idx[speaker]

        return self.h5f[utt_id][index:index + self.audio_window]
