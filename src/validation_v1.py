import numpy as np
import logging
import torch
import torch.nn.functional as F
import pdb
from torch import nn
from torchvision import transforms as T
import torchvision.transforms.functional as TF
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

def validationXXreverse(args, model, device, data_loader, batch_size):
    logger.info("Starting Validation")
    model.eval()
    total_loss = 0
    total_acc  = 0 

    with torch.no_grad():
        for [data, data_r] in data_loader:
            data   = data.float().unsqueeze(1).to(device) # add channel dimension
            data_r = data_r.float().unsqueeze(1).to(device) # add channel dimension
            hidden1 = model.init_hidden1(len(data))
            hidden2 = model.init_hidden2(len(data))
            acc, loss, hidden1, hidden2 = model(data, data_r, hidden1, hidden2)
            total_loss += len(data) * loss 
            total_acc  += len(data) * acc

    total_loss /= len(data_loader.dataset) # average loss
    total_acc  /= len(data_loader.dataset) # average acc

    logger.info('===> Validation set: Average loss: {:.4f}\tAccuracy: {:.4f}\n'.format(
                total_loss, total_acc))

    return total_acc, total_loss

def validation_spk(args, cdc_model, decom_model,mask_model, spk_model, device, data_loader, batch_size, frame_window):

        logger.info("Starting Validation")
        cdc_model.eval()  # not training cdc model
        decom_model.eval()
        spk_model.eval()
        mask_model.eval()
        total_loss = 0
        total_acc = 0
        ss_total_acc = 0
        with torch.no_grad():
            for [data, target, utti] in data_loader:
                b, f_total = data.size()
                cpc_src_rec = []
                data1 = data.float().unsqueeze(1).to(device)
                target = target.to(device)
                    # state_lab = target.view(-1,1)
                state_lab = target[:, 0].view(-1, 1)
                state_lab = [int(ss) for ss in state_lab]
                state_lab = torch.Tensor(state_lab).long().view(-1, 1).to(device)
                fake_data = decom_model(data1)  # torch.Size([64, 3, 13200])
                data_cpc_channel = fake_data[:, 0, :].unsqueeze(1).to(device)
                data_ss_channel = fake_data[:, 1, :].unsqueeze(1).to(device)
                hidden_ori = cdc_model.init_hidden(len(data_cpc_channel), use_gpu=True)
                output, rev_output, acc, fe_loss, rev_acc, rev_fe_loss, hidden = cdc_model(data_cpc_channel, data_ss_channel, hidden_ori)  # torch.Size([64, 41, 256])

                cpc_feature = output.contiguous().view((-1, 256))  # torch.Size([64*41, 256])
                ss_feature = rev_output.contiguous().view((-1, 256))  # torch.Size([64, 41, 256])
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

                anchor_sv, anchor_ss,  predict_rev, predict,  ce_loss, ss_loss, tar, ss_tar= spk_model(cpc_src, ss_src, state_lab)
                ce_loss = (ce_loss.sum() / (batch_size))
                ss_loss = (ss_loss.sum() / (batch_size))
                mask_loss = explainability_loss(cpc_mask_all)
                all_ce_loss = args.alpha_resnet * ce_loss + args.alpha_cpc * fe_loss
                all_ss_loss = ss_loss + args.alpha_rev_cpc * rev_fe_loss
                loss = args.alpha * all_ce_loss + all_ss_loss + args.mask_alpha * mask_loss
                    # fea_frames = 64
                tar_total = tar.cpu()  # /(data_line+1)
                ss_tar_total = ss_tar.cpu()
                total_loss += loss

                predict = tar_total.max(dim=1)[1]
                ss_predict = ss_tar_total.max(dim=1)[1]
                total_acc += predict.eq(state_lab.view_as(predict).cpu()).sum().item()
                ss_total_acc += ss_predict.eq(state_lab.view_as(ss_predict).cpu()).sum().item()
        #pdb.set_trace()
        total_loss /= len(data_loader.dataset) * frame_window  # average loss
        total_acc /= 1. * len(data_loader.dataset) * frame_window  # average acc
        ss_total_acc /= 1. * len(data_loader.dataset) * frame_window  # average acc

        # total_loss /= (len(data)*frame_window) # average loss
        # total_acc  /= (1.*len(data)*frame_window) # average acc

        logger.info(
            '===> Validation set: Average loss: {:.4f}\tAccuracy: {:.4f}\tss_Accuracy2: {:.4f}\tdev_num: {:.4f}\n'.format(
                total_loss, total_acc, ss_total_acc, 1.0 * len(data_loader.dataset)))

        return total_acc, ss_total_acc, total_loss

def validation(args, model, decom_model, device, data_loader, batch_size):
    logger.info("Starting Validation")
    model.eval()
    decom_model.eval()
    fe_total_loss = 0
    fe_total_acc  = 0
    rev_fe_total_loss = 0
    rev_fe_total_acc = 0

    with torch.no_grad():
        for data in data_loader2:
            b,f_total,l = data.size()
            fe_tmp_loss = 0
            rev_fe_tmp_loss = 0
            fe_tmp_acc = 0
            rev_fe_tmp_acc = 0
            for data_line in range(f_total):
                data1 = data[:,data_line,:]
                data1 = data1.float().unsqueeze(1).to(device) # add channel dimension
                fake_data = decom_model(data1)  # torch.Size([64, 3, 13200])
                data_cpc_channel = fake_data[:, 0, :].unsqueeze(1).to(device)
                data_ss_channel = fake_data[:, 1, :].unsqueeze(1).to(device)
                hidden_ori = model.init_hidden(len(data_cpc_channel), use_gpu=True)
                output, rev_output, acc, fe_loss, rev_acc, rev_fe_loss, hidden = model(data_cpc_channel, data_ss_channel, hidden_ori)
                fe_tmp_loss +=  fe_loss
                rev_fe_tmp_loss +=  rev_fe_loss
                fe_tmp_acc  +=  acc
                rev_fe_tmp_acc += rev_acc
            fe_tmp_loss /= f_total
            rev_fe_tmp_loss /= f_total
            fe_tmp_acc /= f_total
            rev_fe_tmp_acc /= f_total
            fe_total_loss += len(data1) * fe_tmp_loss
            rev_fe_total_loss += len(data1) * rev_fe_tmp_loss
            fe_total_acc  += len(data1) * fe_tmp_acc
            rev_fe_total_acc += len(data1) * rev_fe_tmp_acc
        fe_total_loss /= len(data_loader.dataset) # average loss
        rev_fe_total_loss /= len(data_loader.dataset)  # average loss
        fe_total_acc  /= len(data_loader.dataset) # average acc
        rev_fe_total_acc  /= len(data_loader.dataset)

    logger.info('===> Validation set: fe_loss: {:.4f}\tfeAcc: {:.4f}\trevfeloss: {:.4f}\trevfeAcc: {:.4f}\n'.format(
                fe_total_loss, fe_total_acc, rev_fe_total_loss, rev_fe_total_acc))

    return fe_total_acc + rev_fe_total_acc, fe_total_loss + args.alpha_rev_cpc * rev_fe_total_loss
