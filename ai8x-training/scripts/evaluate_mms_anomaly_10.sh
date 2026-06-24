#!/bin/sh
python train.py --deterministic --model ai85mmsanomalynet10aux --use-bias \
  --dataset MMS_128 --device MAX78000 --evaluate --workers=0 "$@"
