import numpy as np
import logging
import torch
import torch.nn.functional as F
import pdb
## Get the same logger from main"
logger = logging.getLogger("cdc")

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

def validation_spk(args, cdc_model, spk_model, device, data_loader,data_loader2, batch_size, frame_window):
    logger.info("Starting Validation")
    cdc_model.eval() # not training cdc model 
    spk_model.eval()
    total_loss = 0

    total_acc  = 0 
    total_acc2 = 0
    with torch.no_grad():
        for [data, target,utti] in data_loader2:
            b,f_total,l = data.size()
            tar_total = torch.zeros([b,10])
            loss = 0
            for data_line in range(f_total):
                data1 = data[:,data_line,:]
                data1 = data1.float().unsqueeze(1).to(device) # add channel dimension
                target = target.to(device)
                hidden = cdc_model.init_hidden(len(data1))
                output, hidden = cdc_model.predict(data1, hidden)
                data1 = output.contiguous().view((-1,256))
                #target = target.view((-1,))
                # target2 = []
                # for i in range(0,len(target)):
                    # for jj in range(0,128):
                        # target2.append(target[i])
                # #pdb.set_trace()
                # target2 = torch.Tensor(target2).view(-1,1).long().to(device)
                #output = spk_model.forward(data)
                src = data1.view(-1,64,256)
                batch_size, fea_frames, fea_dim = src.size()
                #state_lab = target.view(-1,1)
                state_lab = target[:,0].view(-1,1)
                state_lab = [int(ss) for ss in state_lab]
                state_lab = torch.Tensor(state_lab).long().view(-1,1).to(device)
                fea_frames = 64
                
                #anchor_sv, predict, ce_loss,tar = spk_model(src,state_lab, fea_frames,0)
                src = src[:,-1,:]
                state_lab = state_lab.view(-1)
                predict = spk_model.forward(src)
                tar_total += predict.cpu()#/(data_line+1)
                loss_nll = F.nll_loss(predict,state_lab)
                loss += loss_nll
                #total_loss += F.nll_loss(output, target2.squeeze(), size_average=False).item() # sum up batch loss
                #pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
                #pred = output.view(-1,128,8).sum(dim=1).max(1, keepdim=True)[1]
            tar_total = tar_total / f_total
            total_loss += (loss / (f_total * batch_size))
            predict = tar_total.max(dim=1)[1]
            total_acc += predict.eq(state_lab.view_as(predict).cpu()).sum().item()
            #for ll in range(0,len(predict)):
             #   if(predict[ll] == 0):
            #        predict[ll] = 0
            #    else:
            #        predict[ll] = 1
            # for ll in range(0,len(target)):
                # if(target[ll] == 0):
                    # target[ll] = 0
                # else:
                    # target[ll] = 1
            total_acc2 += predict.eq(state_lab.view_as(predict).cpu()).sum().item()
    total_loss /= len(data_loader.dataset)*frame_window # average loss
    total_acc  /= 1.*len(data_loader.dataset)*frame_window # average acc
    total_acc2  /= 1.*len(data_loader.dataset)*frame_window
    #total_loss /= (len(data)*frame_window) # average loss
    #total_acc  /= (1.*len(data)*frame_window) # average acc

    logger.info('===> Validation set: Average loss: {:.4f}\tAccuracy: {:.4f}\tAccuracy2: {:.4f}\tdev_num: {:.4f}\n'.format(
                total_loss, total_acc,total_acc2,1.0*len(data_loader.dataset)))

    return total_acc, total_loss

def validation(args, model, device, data_loader,data_loader2, batch_size):
    logger.info("Starting Validation")
    model.eval()
    total_loss = 0
    total_acc  = 0 

    with torch.no_grad():
        for data in data_loader2:
            b,f_total,l = data.size()
            tmp_loss = 0
            tmp_acc = 0
            for data_line in range(f_total):
                data1 = data[:,data_line,:]
                data1 = data1.float().unsqueeze(1).to(device) # add channel dimension
                hidden = model.init_hidden(len(data1), use_gpu=True)
                acc, loss, hidden = model(data1, hidden)
                tmp_loss +=  loss 
                tmp_acc  +=  acc
            tmp_loss /= f_total
            tmp_acc /= f_total
            total_loss += len(data1) * tmp_loss 
            total_acc  += len(data1) * tmp_acc
    total_loss /= len(data_loader.dataset) # average loss
    total_acc  /= len(data_loader.dataset) # average acc

    logger.info('===> Validation set: Average loss: {:.4f}\tAccuracy: {:.4f}\n'.format(
                total_loss, total_acc))

    return total_acc, total_loss
