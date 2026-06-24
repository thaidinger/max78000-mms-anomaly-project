#!/bin/sh
python train.py --deterministic --evaluate --optimizer Adam --lr 0.001 --wd 0 \
  --model ai85mmsanomalynet15resaux --use-bias --dataset MMS_128_V3 --device MAX78000 \
  --batch-size 256 --qat-policy policies/qat_policy_mms_deep.yaml \
  --compress policies/schedule-mms-deep.yaml --validation-split 0 --workers=0 \
  "$@"
