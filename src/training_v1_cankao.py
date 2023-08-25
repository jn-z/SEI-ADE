import torch
import logging
import os
import torch.nn.functional as F
import pdb
import numpy as np
## Get the same logger from main"
logger = logging.getLogger("cdc")
def get_features(sig , window_length=500, window_step=56, NFFT=446, max_frames=256):
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
    complex_spectrum=np.fft.rfft(frames,NFFT)
    complex_spectrum=np.absolute(complex_spectrum)
    return 1.0/NFFT * np.square(complex_spectrum)
def trainXXreverse(args, model, device, train_loader, optimizer, epoch, batch_size):
    model.train()
    for batch_idx, [data, data_r] in enumerate(train_loader):
        data   = data.float().unsqueeze(1).to(device) # add channel dimension
        data_r = data_r.float().unsqueeze(1).to(device) # add channel dimension
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

def train_spk(args, cdc_model, decom_model, spk_model, device, train_loader,train_loader2, optimizer, epoch, batch_size, frame_window):
    cdc_model.train() # not training cdc model
    spk_model.train()
    decom_model.train()
    for batch_idx, [data, target, utti] in enumerate(train_loader2):
        b, f_total, l = data.size()
        # tar_total = torch.zeros([b*2,10])
        tar_total = torch.zeros([b, args.spk_num])
        # loss = 0
        # f_total = 1
        # gg = np.random.randint(f_total)
        # f_total = 1
        for data_line in range(f_total):
            data1 = data[:, data_line, :]
            data1 = data1.float().unsqueeze(1).to(device)  # add channel dimension torch.Size([64, 1, 13200])
            # for three channel signal decom...

            optimizer.zero_grad()
            fake_data = decom_model(data1)  #torch.Size([64, 3, 13200])
            pdb.set_trace()
            data_cpc_channel = fake_data[:,0,:].unsqueeze(1).to(device)
            data_ss_channel = fake_data[:, 1, :].unsqueeze(1).to(device)
            data_mask_channel = fake_data[:, 2, :].unsqueeze(1).to(device)
            target = target.to(device)
            state_lab = target[:, 0].view(-1, 1)
            state_lab = [int(ss) for ss in state_lab]
            state_lab = torch.Tensor(state_lab).long().view(-1, 1).to(device)

            hidden = cdc_model.init_hidden(len(data_cpc_channel), use_gpu=True)
            output, acc, loss, hidden = cdc_model(data_cpc_channel,data_ss_channel, state_lab, npart)
            anchor_sv, anchor_ss, tar_select_new, ss_tar_select_new, ce_loss, ss_loss, tar, ss_tar
            cpc_feature = output.contiguous().view((-1, 256)) #torch.Size([64, 41, 256])
            # target = target.view((-1,1))
            # data1 = data1.view((-1,256))
            # target2 = []
            # for i in range(0,len(target)):
            # for jj in range(0,128):
            # target2.append(target[i])
            # target2 = torch.Tensor(target2).view(-1,1)

            # pdb.set_trace()
            # output = spk_model.forward(data)
            cpc_src = cpc_feature.view(-1, 41, 256)
            batch_size, fea_frames, fea_dim = src.size()
            # state_lab = target

            anchor_sv, predict, ce_loss, tar = spk_model(cpc_src, ss_src, state_lab, 0)
            # print(predict.size())
            # loss = F.nll_loss(output, target2.long().squeeze().to(device))
            # pdb.set_trace()
            loss = (ce_loss.sum() / (batch_size))
            loss.backward()
            optimizer.step()
            lr = optimizer.update_learning_rate()
            # pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            # pdb.set_trace()
            tar_total += tar.cpu()  # /(data_line+1)
        tar_total = tar_total / f_total
        # loss = loss / f_total
        # loss.backward()
        # optimizer.step()
        predict = tar_total.max(dim=1)[1]
        # acc = 1.*predict.eq(state_lab_mix.long().view(-1,1).to(device).view_as(predict).cpu()).sum().item()/(batch_size*2)
        acc = 1. * predict.eq(state_lab.view_as(predict).cpu()).sum().item() / batch_size

        # print(predict.eq(state_lab.view_as(predict).cpu()).sum().item())
        # print (batch_size)
        if batch_idx % args.log_interval == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tlr:{:.5f}\tAccuracy: {:.4f}\tLoss: {:.6f}'.format(
                epoch, batch_idx * b / frame_window, len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), lr, acc, loss.item()))

def train(args, model, device, train_loader,train_loader2, optimizer, epoch, batch_size):
    model.train()
    batch_idx = 0
    for data in train_loader2:
        #pdb.set_trace()
        b,f_total,l = data.size()
        loss_total = 0
        acc_total = 0
        for data_line in range(f_total):
            data1 = data[:,data_line,:]
            data1 = data1.float().unsqueeze(1).to(device) # add channel dimension
            optimizer.zero_grad()
            #hidden = torch.zeros(1, int(len(data1)/4), 256).cuda()
            #print (len(data1))
            hidden = model.init_hidden(len(data1), use_gpu=True)
            acc, loss, hidden = model(data1, hidden)
            #loss_total += loss
            #acc_total += acc
        #loss_total /= f_total
        #acc_total /= f_total
            loss.backward()
            optimizer.step()
            lr = optimizer.update_learning_rate()
        if batch_idx % args.log_interval == 0:
            logger.info('Train Epoch: {} [{}]\tlr:{:.5f}\tAccuracy: {:.4f}\tLoss: {:.6f}'.format(
                epoch, batch_idx , lr, acc, loss.item()))
        batch_idx += 1
def snapshot(dir_path, run_name, state):
    snapshot_file = os.path.join(dir_path,
                    run_name + '-model_best.pth')
    
    torch.save(state, snapshot_file)
    logger.info("Snapshot saved to {}\n".format(snapshot_file))
