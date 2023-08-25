import numpy as np
import logging
import torch
import torch.nn.functional as F
import pdb
## Get the same logger from main"
logger = logging.getLogger("cdc")

def prediction_spk(args, cdc_model, device, data_loader, data_loader2,batch_size, frame_window):
    logger.info("Starting Evaluation")
    cdc_model.eval() # not training cdc model
    with torch.no_grad():
        for [data, target,utti] in data_loader2:
            #pdb.set_trace()
            b,f_total,l = data.size()
            data1_total = torch.zeros([b, 2560])
            #f_total = 1
            for data_line in range(f_total):
                data1 = data[:,data_line,:]
                f= 1
                b,l = data1.size()
                data1 = data1.view(b,-1)
                data1 = data1.float().unsqueeze(1).to(device) # add channel dimension
                target = target.to(device)
                hidden = cdc_model.init_hidden(len(data1))
                output, hidden = cdc_model.predict(data1, hidden)
                data1 = output.contiguous().view((b,-1))
                if f_total > 5:
                    data1_total += data1.cpu()/(data_line+1)
                else:
                    data1_total += data1.cpu()
            data1_total = data1_total / f_total
            for i in range(0, len(utti)):
                f_cpc_feature = open(utti[i] + '.txt', 'w')
                #pdb.set_trace()
                for j in range(0, data1_total.shape[1]):
                    f_cpc_feature.write(str(data1_total.numpy()[i][j]) + '\n')