#!/bin/bash
#source /home/mkws/mhwu/bashrc-pytorch-1.2

# if [ $stage -eq 0 ]; then
    # call main.py; CPC train on LibriSpeech


# CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
    #--raw-hdf5 /H/hjbisai/home/zjn/naval_identification/data_h5/train_all.h5 \
    #--validation-raw /H/hjbisai/home/zjn/naval_identification/data_h5/train_all.h5 \
    #--eval-raw /H/hjbisai/home/zjn/naval_identification/data_h5/train_all.h5 \
    #--train-list /H/hjbisai/home/zjn/naval_identification/data_h5/xinhaoshibie/dataset2/dataset2_9001/train_jinei.txt \
    #--all-sets /H/hjbisai/home/zjn/naval_identification/data_h5/xinhaoshibie/dataset2/dataset2_9001/train_jinei.txt \
    #--validation-list /H/hjbisai/home/zjn/naval_identification/data_h5/xinhaoshibie/dataset2/dataset2_9001/train_jinei.txt \
    #--eval-list /H/hjbisai/home/zjn/naval_identification/data_h5/xinhaoshibie/dataset2/dataset2_9001/test.txt \
    #--index-file /H/hjbisai/home/zjn/naval_identification/data_h5/xinhaoshibie/dataset2/dataset2_9001/index_all.list \
    #--index-test-file /H/hjbisai/home/zjn/naval_identification/data_h5/xinhaoshibie/dataset2/dataset2_9001/index_all.list \
    #--logging-dir ./snapshot/resnet/ --log-interval 50 \
    #--model-path ./snapshot/cpc/cpc-model_best.pth \
    #--result-list ./result

# fi

#if [ $stage -eq 1 ]; then
    # call spk_class.py CUDA_VISIBLE_DEVICES=2
        python spk_class_train.py \
    --raw-hdf5 /H/hjbisai/zjn/zjn/HJbisai/datasets/data_list/h5_index/test_zhanshi.h5 \
    --all-sets ./test/train.txt \
    --validation-raw /H/hjbisai/zjn/zjn/HJbisai/datasets/data_list/h5_index/test_zhanshi.h5 \
    --eval-raw /H/hjbisai/zjn/zjn/HJbisai/datasets/data_list/h5_index/test_zhanshi.h5 \
    --train-list ./test/train.txt \
    --validation-list ./test/validation.txt \
    --eval-list ./test/validation.txt \
    --index-file ./test/index_all.list \
    --index-test-file ./test/index_all.list \
    --logging-dir ./snapshot/resnet/ --log-interval 50 \
    --model-path ./snapshot/resnet/ \
    --result-list ./result

#fi

#2lan_anzhen
# --model-path /yrfs3/mkws/junjiang5/2020/04_05/05/cpc/snapshot/cpc/cdc-2020-06-29_15_08_29-model_best.pth

