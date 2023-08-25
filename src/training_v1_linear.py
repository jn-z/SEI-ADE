import torch
import logging
import os
import torch.nn.functional as F
import pdb
## Get the same logger from main"
logger = logging.getLogger("cdc")

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

def train_spk(args, cdc_model, spk_model, device, train_loader,train_loader2, optimizer, epoch, batch_size, frame_window):
    cdc_model.eval() # not training cdc model 
    spk_model.train()
    for batch_idx, [data, target,utti] in enumerate(train_loader2):
        b,f_total,l = data.size()
        tar_total = torch.zeros([b,10])
        for data_line in range(f_total):
            data1 = data[:,data_line,:]
            data1 = data1.float().unsqueeze(1).to(device) # add channel dimension
        
            target = target.to(device)
           
            hidden = cdc_model.init_hidden(len(data1), use_gpu=True)
            output, hidden = cdc_model.predict(data1, hidden)
            data1 = output.contiguous().view((-1,256))
            #target = target.view((-1,1))
       
            # target2 = []
            # for i in range(0,len(target)):
                # for jj in range(0,128):
                    # target2.append(target[i])
            # target2 = torch.Tensor(target2).view(-1,1)
            optimizer.zero_grad()
           # pdb.set_trace()
            #output = spk_model.forward(data)
            src = data1.view(-1,64,256)
            batch_size, fea_frames, fea_dim = src.size()
            #state_lab = target
            state_lab = target[:,0].view(-1,1)
            state_lab = [int(ss) for ss in state_lab]
            state_lab = torch.Tensor(state_lab).long().view(-1,1).to(device)
            fea_frames = 64
            state_lab = state_lab.view(-1)
           # anchor_sv, predict, ce_loss,tar = spk_model(src,state_lab, fea_frames,0)
            src = src[:,-1,:]
            predict = spk_model.forward(src)
            loss = F.nll_loss(predict,state_lab)


            
            #loss = (ce_loss.sum() / batch_size)
            loss.backward()
            optimizer.step()
            lr = optimizer.update_learning_rate()
            # pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            #pdb.set_trace()
            tar_total += predict.cpu()#/(data_line+1)
        tar_total = tar_total / f_total
        predict = tar_total.max(dim=1)[1]   
        acc = 1.*predict.eq(state_lab.view_as(predict).cpu()).sum().item()/batch_size
        
        #print(predict.eq(state_lab.view_as(predict).cpu()).sum().item())
        #print (batch_size)
        if batch_idx % args.log_interval == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tlr:{:.5f}\tAccuracy: {:.4f}\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data1) / frame_window, len(train_loader.dataset),
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
