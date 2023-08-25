#!/bin/bash
#source /home/mkws/mhwu/bashrc-pytorch-1.2

# if [ $stage -eq 0 ]; then
    # call main.py; CPC train on LibriSpeech


# CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
    # --train-raw /yrfs5/smnh/junjiang5/2020/04_05/05/cpc_resnet/train-chin-japa.h5 \
    # --validation-raw /yrfs5/smnh/junjiang5/2020/04_05/05/cpc_resnet/train-chin-japa.h5 \
    # --eval-raw /yrfs5/smnh/junjiang5/2020/04_05/05/cpc_resnet/train-chin-japa.h5 \
    # --train-list /yrfs5/smnh/junjiang5/2020/04_05/05/cpc_resnet/list/japa_chin_train_rand.list \
    # --validation-list /yrfs5/smnh/junjiang5/2020/04_05/05/cpc_resnet/list/japa_chin_test3.list \
    # --eval-list /yrfs5/smnh/junjiang5/2020/04_05/05/cpc_resnet/list/japa_chin_test3.list \
    # --logging-dir snapshot/cpc2/ \
    # --log-interval 50 --audio-window 20480 --timestep 12 --masked-frames 10 --n-warmup-steps 1000

# fi

#if [ $stage -eq 1 ]; then
    # call spk_class.py CUDA_VISIBLE_DEVICES=2 
    python spk_class.py \
    --eval-raw /H/hjbisai/zjn/zjn/HJbisai/datasets/data_list/h5_index/test_zhanshi.h5 \
    --all-sets ./test/train.txt \
    --eval-list ./test/validation.txt \
    --index-file ./test/index_all.list \
    --index-test-file ./test/index_all.list \
    --logging-dir ./snapshot/resnet/ --log-interval 10 \
    --model-path ./snapshot/resnet/ \
    --result-list ./result
#fi
#--model-path /yrfs5/smnh/junjiang5/2020/04_05/05/cpc_resnet/snapshot/cpc_largeWav/cdc-2020-05-10_08_44_08-model_best_largeWav.pth 

#/yrfs5/smnh/junjiang5/2020/04_05/05/cpc/snapshot/cpc/cdc-2020-06-29_15_08_29-model_best.pth
