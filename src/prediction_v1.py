import numpy as np
import logging
import torch
import torch.nn.functional as F
import pdb
import os
import cv2
from torch import nn
import torchvision.transforms.functional as TF
from torchvision import transforms as T
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as pl
import matplotlib
import math
import random
#from utils import plot_heapmap
## Get the same logger from main"
logger = logging.getLogger("cdc")

class MyRandCrop:
    """Rotate by one of the given angles."""

    def __init__(self, size):
        self.size = size

    def __call__(self, x):
        #pdb.set_trace()
        image_width, image_height = TF.get_image_size(x)
        crop_height, crop_width = self.size
        number_x = int(image_width / crop_width)
        number_y = int(image_height / crop_height)
        crop_all = []
        for j in range(number_y):
            for i in range(number_x):
                #pdb.set_trace()
                t_crop_xy = TF.crop(x, crop_height * j, crop_width * i, crop_height, crop_width)
                crop_all.append(t_crop_xy)
        return crop_all, int(number_y*number_x)
def explainability_loss(mask):
    if type(mask) not in [tuple, list]:
        mask = [mask]
    loss = 0
    for mask_scaled in mask:
        ones_var = torch.ones_like(mask_scaled)
        loss += nn.functional.binary_cross_entropy(mask_scaled, ones_var)
    return loss

def prediction_spk(args, decom_model, cdc_model, mask_model, spk_model, device, data_loader, batch_size, frame_window):
    logger.info("Starting Evaluation")
    decom_model.eval()
    cdc_model.eval() # not training cdc model
    spk_model.eval()
    mask_model.eval()
    total_acc  = 0
    ss_total_acc = 0
    total = 0
    scores, utti_all, indexs = [], [], []
    spk_file_all = []
    maxpredict_all = []
    spk_num_all = []
    f_result = open(args.result_list + '/result.txt', 'w')
    f_zhanshi = open(args.result_list + '/result_zhanshi.txt', 'w')
    f_truth = open(args.result_list + '/gruth.txt', 'w')
    # write the mask_image
    mask_file_path = './mask_images/'
    finger_feature_path = './finger_feature_images/'
    with open(args.result_list+'/map.list','r') as f:
        content = f.readlines()
        content = [x.strip() for x in content]
        spk2idx = {}
        for i in content:
            #pdb.set_trace()
            spk = i.split(' ')[0]
            #utt_len = self.h5f[spk].shape[0]
            idx = int(i.split(' ')[1])
            #if utt_len > 20480 and spk in temp:
            spk2idx[spk] = idx
    with torch.no_grad():
        for [data, target,utti_chushi] in data_loader:
            utti = utti_chushi[0]
            cpc_src_rec = []
            b, f_total = data.size()
            data1 = data.float().unsqueeze(1).to(device)
            target = target.to(device)
            fake_data = decom_model(data1)  # torch.Size([64, 3, 13200])
            data_cpc_channel = fake_data[:, 0, :].unsqueeze(1).to(device)
            data_ss_channel = fake_data[:, 1, :].unsqueeze(1).to(device)
            hidden_ori = cdc_model.init_hidden(len(data_cpc_channel), use_gpu=True)
            output, rev_output, hidden = cdc_model.predict(data_cpc_channel,data_ss_channel, hidden_ori)
            cpc_feature = output.contiguous().view((-1, 256))  # torch.Size([64*41, 256])
            ss_feature = rev_output.contiguous().view((-1, 256))  # torch.Size([64, 41, 256])
            cpc_src = cpc_feature.view(-1, 32, 256)
            ss_src = ss_feature.view(-1, 32, 256)
            cpc_src_moudle_list, number_moudle = MyRandCrop(size=(8, 64))(cpc_src)
            idx = torch.randperm(number_moudle)
            for i in range(number_moudle):
                cpc_src_rec.append(cpc_src_moudle_list[idx[i]])
            cpc_src = torch.cat(cpc_src_rec, dim=1)
            cpc_src = T.Resize(size=(32, 256))(cpc_src)
            cpc_src = cpc_src.view(-1, 1, 32, 256)
            cpc_mask_all = mask_model(cpc_src)
            cpc_mask = cpc_mask_all[0].float()
            cpc_mask = cpc_mask.view(-1, 32, 256)
            cpc_mask = torch.ones_like(cpc_mask) - cpc_mask
            cpc_src = cpc_src.view(-1, 32, 256)
            cpc_src_1 = torch.mul(cpc_mask, cpc_src)
            state_lab = target[:,0].view(-1,1)
            state_lab = [int(ss) for ss in state_lab]
            state_lab = torch.Tensor(state_lab).long().view(-1,1).to(device)
            anchor_sv, anchor_ss,  predict_rev, predict,  ce_loss, ss_loss, tar, ss_tar= spk_model(cpc_src, ss_src, state_lab)
            tar_total = tar.cpu()  # /(data_line+1)
            ss_tar_total = ss_tar.cpu()
            #pdb.set_trace()
            tar_total = tar_total
            ss_tar_total = ss_tar_total
            predict = tar_total.max(dim=1)[1]
            ss_predict = ss_tar_total.max(dim=1)[1]
            total_acc += predict.eq(state_lab.view_as(predict).cpu()).sum().item()
            ss_total_acc += ss_predict.eq(state_lab.view_as(ss_predict).cpu()).sum().item()
            score = tar_total.max(dim=1)[0]
            scores.append(score)
            indexs.append(ss_predict)
            for i in range(0, len(utti)):
                f_zhanshi.write(str(predict.numpy()[i]) + str(state_lab.cpu().numpy()[i]) + '\n')
                f_result.write(str(predict.numpy()[i]) + '\n')
                f_truth.write(str(state_lab.cpu().numpy()[i]) + '\n')
                utti_all.append(utti[i])
                if i < 50:
                    #pdb.set_trace()
                    plt.figure()
                    x = np.array(cpc_src[i, :, :].cpu().numpy()* 255, np.int32)
                    plt.imshow(x)
                    #plt.plot(x)
                    plt.savefig(mask_file_path + utti[i] + ".jpg")
                    #mask_save = np.uint16(cpc_mask[i, :, :].cpu().numpy() * 255)
                    #cv2.imwrite(mask_file_path + utti[i] + ".jpg", mask_save)
                    plt.figure()
                    y = np.array(np.abs(cpc_src_1[i, :, :].cpu().numpy()) * 255, np.int32)
                    plt.imshow(y)
                    plt.savefig(finger_feature_path + utti[i] + ".jpg")
                    #feature_save = np.uint16(cpc_src[i,:,:].cpu().numpy() * 255)
                    #cv2.imwrite(finger_feature_path + utti[i] + ".jpg", feature_save)
            total += b

    mm = total_acc/total
    print("验证正确率:" + str(mm))
    f_result.write(str(ss_total_acc)+'\n')
    f_result.write(str(total)+'\n')
    f_result.write(str(mm)+'\n')
    #logger.info("===> Final predictions done. Here is a snippet")
    #logger.info('===> Evaluation set: Average loss: {:.4f}\tAccuracy: {:.4f}\tnum eval: {:.4f}\n'.format(
             #   total_loss, total_acc,1.0*len(data_loader.dataset)))
