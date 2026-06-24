#!/bin/sh
python train.py --deterministic --epochs 120 --optimizer Adam --lr 0.001 --wd 0 \
  --model ai85mmsanomalynet3aux --use-bias --dataset MMS_128 --device MAX78000 \
  --batch-size 32 --qat-policy policies/qat_policy_mms.yaml \
  --compress policies/schedule-mms.yaml --validation-split 0 --workers=0 \
  --print-freq 25 --tb-image-samples 2 --tb-image-freq 10 --confusion "$@"
