import numpy as np
import logging
import torch
import torch.nn.functional as F
import pdb

## Get the same logger from main"
logger = logging.getLogger("cdc")


def prediction_spk(args, cdc_model, spk_model, device, data_loader, data_loader2, batch_size, frame_window):
    logger.info("Starting Evaluation")
    #cdc_model.eval()  # not training cdc model
    spk_model.eval()
    total_loss = 0
    total_acc = 0
    total = 0
    scores, labels, utti_all, indexs = [], [], [], []
    f_result = open('result.txt', 'w')
    f_result_zhanshi = open('result_pred_zhanshi.txt', 'w')
    f_result_open = open('result_pred_open.txt', 'w')
    f_result_zhanshi_open = open('result_pred_open_zhanshi.txt', 'w')
    with torch.no_grad():
        for [data, target, utti] in data_loader2:
            # pdb.set_trace()
            b, f_total, l = data.size()
            tar_total = torch.zeros([b, args.spk_num])
            # f_total = 1
            for data_line in range(f_total):
                data1 = data[:, data_line, :]
                f = 1
                b, l = data1.size()
                data1 = data1.view(b, -1)
                data1 = data1.float().unsqueeze(1).to(device)  # add channel dimension
                target = target.to(device)
                # pdb.set_trace()
                src = data1.view(-1, 100, 132)
                batch_size, fea_frames, fea_dim = src.size()
                state_lab = target[:, 0].view(-1, 1)
                state_lab = [int(ss) for ss in state_lab]
                state_lab = torch.Tensor(state_lab).long().view(-1, 1).to(device)
                # fea_frames = 128
                # pdb.set_trace()
                #anchor_sv, predict, ce_loss,tar = spk_model(src,state_lab, fea_frames,0)
                tar = spk_model(src, state_lab, fea_frames, 0, is_train=False)
                # tar, anchor_sv = spk_model(src)
                if f_total > 5:
                    tar_total += tar.cpu() / (data_line + 1)
                else:
                    tar_total += tar.cpu()
            # pdb.set_trace()
            tar_total = tar_total / f_total
            score = tar_total.max(dim=1)[0]
            scores.append(score)
            labels.append(state_lab)
            predict = tar_total.max(dim=1)[1]
            indexs.append(predict)
            target_result = state_lab.view_as(predict).cpu()
            for i in range(0, len(utti)):
                f_result.write(utti[i] + ' ' + str(predict.numpy()[i]) + '\n')
                f_result_zhanshi.write(utti[i] + ' ' + str(predict.numpy()[i]) + ' ' + str(score.numpy()[i]) + '\n')
                utti_all.append(utti[i])
            total_acc += predict.eq(state_lab.view_as(predict).cpu()).sum().item()
            total += b

    print(total_acc)
    print(total)
    print(total_acc / total)
    scores = torch.cat(scores, dim=0).cpu().numpy()
    labels = torch.cat(labels, dim=0).cpu().numpy()
    indexs = torch.cat(indexs, dim=0).cpu().numpy()
    # scores = np.array(scores)[:, np.newaxis,:]
    labels = np.squeeze(np.array(labels))
    scores = np.squeeze(np.array(scores))
    indexs = np.squeeze(np.array(indexs))
    pred_softmax, pred_softmax_threshold, pred_openmax = [], [], []
    score_softmax, score_openmax = [], []
    # pdb.set_trace()
    for i in range(len(scores)):
        # for score in scores:
        # so, ss = openmax(weibull_model, categories, score,
        #                 0.5, args.weibull_alpha, "euclidean")  # openmax_prob, softmax_prob
        # pred_softmax.append(np.argmax(ss))
        pred_softmax_threshold.append(indexs[i] if scores[i] >= args.weibull_threshold else args.spk_num)
        # pred_openmax.append(np.argmax(ss) if np.max(so) >= args.weibull_threshold else args.spk_num)
        # score_softmax.append(ss)
        # score_openmax.append(so)
    for i in range(0, len(utti_all)):
        # f_result_open.write(temp[i] + ' ' + str(pred_openmax[i]) + '\n')
        # f_result_zhanshi_open.write(utti_all[i] + ' ' + str(pred_softmax_threshold[i]) + ' ' + str(pred_openmax[i]) + '\n')
        f_result_open.write(utti_all[i] + ' ' + str(pred_softmax_threshold[i]) + '\n')
        # f_result_zhanshi_open.write(utti_all[i] + ' ' + str(labels[i]) + ' ' + str(pred_softmax_threshold[i]) + '\n')
    # correct = (np.array(pred_openmax) == labels).sum().item()
    pred_list_fenbu = []
    for i in range(0, 21):
        correct_0 = (np.array(pred_softmax_threshold) == i * np.ones_like(labels)).sum().item()
        pred_list_fenbu.append(correct_0)
    print("only test in open sets:")
    print(len(labels))
    print(pred_list_fenbu)
    # logger.info("===> Final predictions done. Here is a snippet")
    # logger.info('===> Evaluation set: Average loss: {:.4f}\tAccuracy: {:.4f}\tnum eval: {:.4f}\n'.format(
    #   total_loss, total_acc,1.0*len(data_loader.dataset)))
