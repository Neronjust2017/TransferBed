#!/usr/bin/env bash
source /home/weiyuhua/TransferBed/env/bin/activate

# Office31
CUDA_VISIBLE_DEVICES=0 python afn.py /data/weiyuhua/TransferBed/data/office31 -d Office31 -s A -t W -a resnet50 --trade-off-entropy 0.1 --epochs 20 --seed 1 --log logs/afn/Office31_A2W
CUDA_VISIBLE_DEVICES=0 python afn.py /data/weiyuhua/TransferBed/data/office31 -d Office31 -s D -t W -a resnet50 --trade-off-entropy 0.1 --epochs 20 --seed 1 --log logs/afn/Office31_D2W
CUDA_VISIBLE_DEVICES=0 python afn.py /data/weiyuhua/TransferBed/data/office31 -d Office31 -s W -t D -a resnet50 --trade-off-entropy 0.1 --epochs 20 --seed 1 --log logs/afn/Office31_W2D
CUDA_VISIBLE_DEVICES=0 python afn.py /data/weiyuhua/TransferBed/data/office31 -d Office31 -s A -t D -a resnet50 --trade-off-entropy 0.1 --epochs 20 --seed 1 --log logs/afn/Office31_A2D
CUDA_VISIBLE_DEVICES=0 python afn.py /data/weiyuhua/TransferBed/data/office31 -d Office31 -s D -t A -a resnet50 --trade-off-entropy 0.1 --epochs 20 --seed 1 --log logs/afn/Office31_D2A
CUDA_VISIBLE_DEVICES=0 python afn.py /data/weiyuhua/TransferBed/data/office31 -d Office31 -s W -t A -a resnet50 --trade-off-entropy 0.1 --epochs 20 --seed 1 --log logs/afn/Office31_W2A


# Office-Home
#CUDA_VISIBLE_DEVICES=0 python afn.py /data/weiyuhua/TransferBed/data/office-home -d OfficeHome -s Ar -t Cl -a resnet50 --epochs 20 --seed 0 --log logs/afn/OfficeHome_Ar2Cl
#CUDA_VISIBLE_DEVICES=0 python afn.py /data/weiyuhua/TransferBed/data/office-home -d OfficeHome -s Ar -t Pr -a resnet50 --epochs 20 --seed 0 --log logs/afn/OfficeHome_Ar2Pr
#CUDA_VISIBLE_DEVICES=0 python afn.py /data/weiyuhua/TransferBed/data/office-home -d OfficeHome -s Ar -t Rw -a resnet50 --epochs 20 --seed 0 --log logs/afn/OfficeHome_Ar2Rw
#CUDA_VISIBLE_DEVICES=0 python afn.py /data/weiyuhua/TransferBed/data/office-home -d OfficeHome -s Cl -t Ar -a resnet50 --epochs 20 --seed 0 --log logs/afn/OfficeHome_Cl2Ar
#CUDA_VISIBLE_DEVICES=0 python afn.py /data/weiyuhua/TransferBed/data/office-home -d OfficeHome -s Cl -t Pr -a resnet50 --epochs 20 --seed 0 --log logs/afn/OfficeHome_Cl2Pr
#CUDA_VISIBLE_DEVICES=0 python afn.py /data/weiyuhua/TransferBed/data/office-home -d OfficeHome -s Cl -t Rw -a resnet50 --epochs 20 --seed 0 --log logs/afn/OfficeHome_Cl2Rw
#CUDA_VISIBLE_DEVICES=0 python afn.py /data/weiyuhua/TransferBed/data/office-home -d OfficeHome -s Pr -t Ar -a resnet50 --epochs 20 --seed 0 --log logs/afn/OfficeHome_Pr2Ar
#CUDA_VISIBLE_DEVICES=0 python afn.py /data/weiyuhua/TransferBed/data/office-home -d OfficeHome -s Pr -t Cl -a resnet50 --epochs 20 --seed 0 --log logs/afn/OfficeHome_Pr2Cl
#CUDA_VISIBLE_DEVICES=0 python afn.py /data/weiyuhua/TransferBed/data/office-home -d OfficeHome -s Pr -t Rw -a resnet50 --epochs 20 --seed 0 --log logs/afn/OfficeHome_Pr2Rw
#CUDA_VISIBLE_DEVICES=0 python afn.py /data/weiyuhua/TransferBed/data/office-home -d OfficeHome -s Rw -t Ar -a resnet50 --epochs 20 --seed 0 --log logs/afn/OfficeHome_Rw2Ar
#CUDA_VISIBLE_DEVICES=0 python afn.py /data/weiyuhua/TransferBed/data/office-home -d OfficeHome -s Rw -t Cl -a resnet50 --epochs 20 --seed 0 --log logs/afn/OfficeHome_Rw2Cl
#CUDA_VISIBLE_DEVICES=0 python afn.py /data/weiyuhua/TransferBed/data/office-home -d OfficeHome -s Rw -t Pr -a resnet50 --epochs 20 --seed 0 --log logs/afn/OfficeHome_Rw2Pr

# PACS
CUDA_VISIBLE_DEVICES=0 python afn.py /data/weiyuhua/TransferBed/data/pacs -d PACS -s Ar -t Ca -a resnet18 --epochs 20 --seed 0 --log logs/afn/PACS_Ar2Ca
CUDA_VISIBLE_DEVICES=0 python afn.py /data/weiyuhua/TransferBed/data/pacs -d PACS -s Ar -t Ph -a resnet18 --epochs 20 --seed 0 --log logs/afn/PACS_Ar2Ph
CUDA_VISIBLE_DEVICES=0 python afn.py /data/weiyuhua/TransferBed/data/pacs -d PACS -s Ar -t Sk -a resnet18 --epochs 20 --seed 0 --log logs/afn/PACS_Ar2Sk
CUDA_VISIBLE_DEVICES=0 python afn.py /data/weiyuhua/TransferBed/data/pacs -d PACS -s Ca -t Ar -a resnet18 --epochs 20 --seed 0 --log logs/afn/PACS_Ca2Ar
CUDA_VISIBLE_DEVICES=0 python afn.py /data/weiyuhua/TransferBed/data/pacs -d PACS -s Ca -t Ph -a resnet18 --epochs 20 --seed 0 --log logs/afn/PACS_Ca2Ph
CUDA_VISIBLE_DEVICES=0 python afn.py /data/weiyuhua/TransferBed/data/pacs -d PACS -s Ca -t Sk -a resnet18 --epochs 20 --seed 0 --log logs/afn/PACS_Ca2Sk
CUDA_VISIBLE_DEVICES=0 python afn.py /data/weiyuhua/TransferBed/data/pacs -d PACS -s Ph -t Ar -a resnet18 --epochs 20 --seed 0 --log logs/afn/PACS_Ph2Ar
CUDA_VISIBLE_DEVICES=0 python afn.py /data/weiyuhua/TransferBed/data/pacs -d PACS -s Ph -t Ca -a resnet18 --epochs 20 --seed 0 --log logs/afn/PACS_Ph2Ca
CUDA_VISIBLE_DEVICES=0 python afn.py /data/weiyuhua/TransferBed/data/pacs -d PACS -s Ph -t Sk -a resnet18 --epochs 20 --seed 0 --log logs/afn/PACS_Ph2Sk
CUDA_VISIBLE_DEVICES=0 python afn.py /data/weiyuhua/TransferBed/data/pacs -d PACS -s Sk -t Ar -a resnet18 --epochs 20 --seed 0 --log logs/afn/PACS_Sk2Ar
CUDA_VISIBLE_DEVICES=0 python afn.py /data/weiyuhua/TransferBed/data/pacs -d PACS -s Sk -t Ca -a resnet18 --epochs 20 --seed 0 --log logs/afn/PACS_Sk2Ca
CUDA_VISIBLE_DEVICES=0 python afn.py /data/weiyuhua/TransferBed/data/pacs -d PACS -s Sk -t Ph -a resnet18 --epochs 20 --seed 0 --log logs/afn/PACS_Sk2Ph


# VisDA-2017
#CUDA_VISIBLE_DEVICES=0 python afn.py /data/weiyuhua/TransferBed/data/visda-2017 -d VisDA2017 -s Synthetic -t Real -a resnet101 -r 0.3 -b 36 \
#    --epochs 10 -i 1000 --seed 0 --per-class-eval --center-crop --log logs/afn/VisDA2017
