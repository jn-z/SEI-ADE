import torch
import logging
import os
from torch import nn
import torch.nn.functional as F
from torch import sigmoid
import pdb
import numpy as np
import random
from torchvision import transforms as T
## Get the same logger from main"
logger = logging.getLogger("cdc")
import torchvision.transforms.functional as TF


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




def get_features(sig, window_length=500, window_step=56, NFFT=446, max_frames=256):
    feat_mat = []
    for i in range(max_frames):
        start = window_step * i
        end = start + window_length
        slice_sig = sig[start:end]
        feature = STFT(slice_sig, NFFT)
        feat_mat.append(feature)
    feat_mat = np.array(feat_mat, dtype=float)
    return feat_mat


def STFT(frames, NFFT):
    complex_spectrum = np.fft.rfft(frames, NFFT)
    complex_spectrum = np.absolute(complex_spectrum)
    return 1.0 / NFFT * np.square(complex_spectrum)


def trainXXreverse(args, model, device, train_loader, optimizer, epoch, batch_size):
    model.train()
    for batch_idx, [data, data_r] in enumerate(train_loader):
        data = data.float().unsqueeze(1).to(device)  # add channel dimension
        data_r = data_r.float().unsqueeze(1).to(device)  # add channel dimension
        optimizer.zero_grad()
        hidden1 = model.init_hidden1(len(data))
        hidden2 = model.init_hidden2(len(data))
        acc, loss, hidden1, hidden2 = model(data, data_r, hidden1, hidden2)

        loss.backward()
        optimizer.step()
        lr = optimizer.update_learning_rate()
        if batch_idx % args.log_interval == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tlr:{:.5f}\tAccuracy: {:.4f}\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), lr, acc, loss.item()))


def explainability_loss(mask):
    if type(mask) not in [tuple, list]:
        mask = [mask]
    loss = 0
    for mask_scaled in mask:
        ones_var = torch.ones_like(mask_scaled)
        loss += nn.functional.binary_cross_entropy(mask_scaled, ones_var)
    return loss


def train_spk(args, cdc_model, decom_model, mask_model, spk_model, device, train_loader, optimizer_decom, optimizer_mask, optimizer_cpc, optimizer,
              epoch, batch_size, frame_window):
    cdc_model.train()  # not training cdc model
    spk_model.train()
    decom_model.train()
    mask_model.train()
    mask_model.init_weights()
    for batch_idx, [data, target, utti] in enumerate(train_loader):
        b, f_total = data.size()
        data1 = data.float().unsqueeze(1).to(device)
        cpc_src_rec = []
        optimizer.zero_grad()
        optimizer_decom.zero_grad()
        optimizer_cpc.zero_grad()
        fake_data = decom_model(data1)  # torch.Size([64, 3, 13200])
        data_cpc_channel = fake_data[:, 0, :].unsqueeze(1).to(device)
        data_ss_channel = fake_data[:, 1, :].unsqueeze(1).to(device)

        target = target.to(device)
        state_lab = target[:, 0].view(-1, 1)
        state_lab = [int(ss) for ss in state_lab]
        state_lab = torch.Tensor(state_lab).long().view(-1, 1).to(device)

        hidden_ori = cdc_model.init_hidden(len(data_cpc_channel), use_gpu=True)

        output, ss_output, acc, fe_loss, rev_acc, rev_fe_loss, hidden = cdc_model(data_cpc_channel, data_ss_channel, hidden_ori)  # torch.Size([64, 41, 256])
        cpc_feature = output.contiguous().view((-1, 256))  # torch.Size([64*41, 256])
        ss_feature = ss_output.contiguous().view((-1, 256))  # torch.Size([64, 41, 256])
        ss_src = ss_feature.view(-1, 32, 256)
        cpc_src = cpc_feature.view(-1, 32, 256)
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
        cpc_src = torch.mul(cpc_mask, cpc_src)
        batch_size, fea_frames, fea_dim = cpc_src.size()
        # state_lab = target

        anchor_sv, anchor_ss,  predict_rev, predict,  ce_loss, ss_loss, tar, ss_tar= spk_model(cpc_src, ss_src, state_lab)
        ce_loss = (ce_loss.sum() / (batch_size))
        ss_loss = (ss_loss.sum() / (batch_size))
        mask_loss = explainability_loss(cpc_mask_all)
        all_ce_loss = args.alpha_resnet * ce_loss + args.alpha_cpc * fe_loss
        all_ss_loss = ss_loss + args.alpha_rev_cpc * rev_fe_loss
        all_loss = args.alpha * all_ce_loss + all_ss_loss + args.mask_alpha * mask_loss
        all_loss.backward()
        optimizer_decom.step()
        lr_decom = optimizer_decom.update_learning_rate()

        optimizer_mask.step()
        lr_mask = optimizer_mask.update_learning_rate()

        optimizer_cpc.step()
        lr_cpc = optimizer_cpc.update_learning_rate()

        optimizer.step()
        lr = optimizer.update_learning_rate()
        # pdb.set_trace()
        tar_total = tar.cpu()  # /(data_line+1)
        ss_tar_total = ss_tar.cpu()

        predict = tar_total.max(dim=1)[1]
        ss_predict = ss_tar_total.max(dim=1)[1]
        acc = 1. * predict.eq(state_lab.view_as(predict).cpu()).sum().item() / batch_size
        ss_acc = 1. * ss_predict.eq(state_lab.view_as(ss_predict).cpu()).sum().item() / batch_size

        if batch_idx % args.log_interval == 0:
            logger.info(
                'Train Epoch: {} \tss_loss:{:.5f}\trev_fe_loss:{:.5f}\tce_loss:{:.5f}\tfe_loss:{:.5f}\tAccuracy: {:.4f}\tLoss: {:.6f}\tss_Acc: {:.4f}\tmask_Loss: {:.6f}'.format(
                    epoch, ss_loss, rev_fe_loss, ce_loss,fe_loss, acc, all_ce_loss.item(), ss_acc, mask_loss))


def train(args, model, decom_model, device, train_loader, train_loader2, optimizer_decom, optimizer, epoch, batch_size):
    model.train()
    decom_model.train()
    batch_idx = 0
    for data in train_loader2:
        # pdb.set_trace()
        b, f_total, l = data.size()
        loss_total = 0
        acc_total = 0
        for data_line in range(f_total):
            data1 = data[:, data_line, :]
            data1 = data1.float().unsqueeze(1).to(device)  # add channel dimension
            optimizer.zero_grad()
            optimizer_decom.zero_grad()
            fake_data = decom_model(data1)  # torch.Size([64, 3, 13200])
            data_cpc_channel = fake_data[:, 0, :].unsqueeze(1).to(device)
            data_ss_channel = fake_data[:, 1, :].unsqueeze(1).to(device)
            hidden_ori = model.init_hidden(len(data_cpc_channel), use_gpu=True)
            output, rev_output, acc, fe_loss, rev_acc, rev_fe_loss, hidden = model(data_cpc_channel, data_ss_channel,
                                                                                   hidden_ori)  # torch.Size([64, 41, 256])
            # cpc_feature = output.contiguous().view((-1, 256))  # torch.Size([64*41, 256])
            # ss_feature = rev_output.contiguous().view((-1, 256))  # torch.Size([64, 41, 256])
            loss = fe_loss + args.alpha_rev_cpc * rev_fe_loss
            # loss_total += loss
            # acc_total += acc
            # loss_total /= f_total
            # acc_total /= f_total
            loss.backward()
            optimizer_decom.step()
            lr_decom = optimizer_decom.update_learning_rate()

            optimizer.step()
            lr = optimizer.update_learning_rate()
        if batch_idx % args.log_interval == 0:
            logger.info(
                'Train Epoch: {} [{}]\tlr:{:.5f}\tlr_decom:{:.5f}\tAccuracy: {:.4f}\tRevAccuracy: {:.4f}\tLoss: {:.6f}\tFE_Loss: {:.6f}\tss_Loss: {:.6f}'.format(
                    epoch, batch_idx, lr, lr_decom, acc, rev_acc, loss.item(), fe_loss.item(), rev_fe_loss.item()))
        batch_idx += 1


def snapshot(dir_path, run_name, state):
    snapshot_file = os.path.join(dir_path,
                                 run_name + '-model_best.pth')

    torch.save(state, snapshot_file)
    logger.info("Snapshot saved to {}\n".format(snapshot_file))
