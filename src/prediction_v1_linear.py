import numpy as np
import logging
import torch
import torch.nn.functional as F
import pdb
## Get the same logger from main"
logger = logging.getLogger("cdc")

def prediction_spk(args, cdc_model, spk_model, device, data_loader, data_loader2,batch_size, frame_window):
    logger.info("Starting Evaluation")
    cdc_model.eval() # not training cdc model 
    spk_model.eval()
    total_loss = 0
    total_acc  = 0 
    total = 0
    f_result = open('result.txt','w')
    with torch.no_grad():
        for [data, target,utti] in data_loader2:
            #pdb.set_trace()
            b,f_total,l = data.size()
            tar_total = torch.zeros([b,10])
            src_total = torch.zeros(b,256)
            for data_line in range(f_total):
                data1 = data[:,data_line,:]
                f= 1
                b,l = data1.size()
                data1 = data1.view(b,-1)
                data1 = data1.float().unsqueeze(1).to(device) # add channel dimension
                target = target.to(device)
                #pdb.set_trace()
                hidden = cdc_model.init_hidden(len(data1))
                output, hidden = cdc_model.predict(data1, hidden)
                data1 = output.contiguous().view((-1,256))
                #target = target.view((-1,))
                # target2 = []
                # for i in range(0,len(target)):
                    # #for jj in range(0,128):
                    # for jj in range(0,128*f):
                        # target2.append(target[i])
                #pdb.set_trace()
                #target2 = torch.Tensor(target2).view(-1,1).long().to(device)
                #output = spk_model.forward(data)
                #total_loss += F.nll_loss(output, target2.squeeze(), size_average=False).item() # sum up batch loss
                #pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
                #pred = output.view(-1,128,8).sum(dim=1).max(1, keepdim=True)[1]
                #src = data.view(-1,128,256)
                src = data1.view(-1,64*f,256)
                batch_size, fea_frames, fea_dim = src.size()
                state_lab = target[:,0].view(-1,1)
                state_lab = [int(ss) for ss in state_lab]
                state_lab = torch.Tensor(state_lab).long().view(-1,1).to(device)
                #fea_frames = 128
                #pdb.set_trace()
                #anchor_sv, predict, ce_loss,tar = spk_model(src,state_lab, fea_frames,0)
                state_lab = state_lab.view(-1)
                src = src[:,-1,:]
                src_total += src.cpu()
                tar = spk_model.forward(src)
                if f_total > 5:
                    tar_total += tar.cpu()/(data_line+1)
                else:
                    tar_total += tar.cpu()
            #pdb.set_trace()
            tar_total = tar_total / f_total
            src_total = src_total/ f_total
            for i in range(0,len(utti)):
                np.savetxt('./test_cpc_fea/'+utti[i]+'.txt',src_total[i].cpu(),fmt="%.4f")
            predict = tar_total.max(dim=1)[1]
            #pdb.set_trace()
            for i in range(0,len(utti)):
                f_result.write(utti[i]+' '+str(predict.numpy()[i])+'\n')
            #for ll in range(0,len(predict)):
               # if(predict[ll] == 0):
               #     predict[ll] = 0
               # else:
               #     predict[ll] = 1
            # for ll in range(0,len(target)):
                # if(target[ll] == 0):
                    # target[ll] = 0
                # else:
                    # target[ll] = 1
            total_acc += predict.eq(state_lab.view_as(predict).cpu()).sum().item()
            total += b
            # output = spk_model.forward(data)
            # total_loss += F.nll_loss(output, target, size_average=False).item() # sum up batch loss
            # pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            # total_acc += pred.eq(target.view_as(pred)).sum().item()

    #total_loss /= len(data_loader.dataset)*frame_window # average loss
    #total_acc  /= len(data_loader.dataset)*frame_window # average acc
    print(total_acc)
    print(total)
    #logger.info("===> Final predictions done. Here is a snippet")
    #logger.info('===> Evaluation set: Average loss: {:.4f}\tAccuracy: {:.4f}\tnum eval: {:.4f}\n'.format(
             #   total_loss, total_acc,1.0*len(data_loader.dataset)))
