#!/usr/bin/env bash
source /home/weiyuhua/TransferBed/env/bin/activate

# PACS
CUDA_VISIBLE_DEVICES=3 python afn.py data/pacs -d PACS -t Ar -a resnet18 --epochs 30  --seed 0 --log logs/afn/PACS_:2Ar
CUDA_VISIBLE_DEVICES=3 python afn.py data/pacs -d PACS -t Ca -a resnet18 --epochs 30  --seed 0 --log logs/afn/PACS_:2Ca
CUDA_VISIBLE_DEVICES=3 python afn.py data/pacs -d PACS -t Ph -a resnet18 --epochs 30  --seed 0 --log logs/afn/PACS_:2Ph
CUDA_VISIBLE_DEVICES=3 python afn.py data/pacs -d PACS -t Sk -a resnet18 --epochs 30  --seed 0 --log logs/afn/PACS_:2Sk

# Office-Home
CUDA_VISIBLE_DEVICES=3 python afn.py data/office-home -d OfficeHome -t Ar -a resnet50 --epochs 30  --seed 0 --log logs/afn/OfficeHome_:2Ar
CUDA_VISIBLE_DEVICES=3 python afn.py data/office-home -d OfficeHome -t Cl -a resnet50 --epochs 30  --seed 0 --log logs/afn/OfficeHome_:2Cl
CUDA_VISIBLE_DEVICES=3 python afn.py data/office-home -d OfficeHome -t Pr -a resnet50 --epochs 30  --seed 0 --log logs/afn/OfficeHome_:2Pr
CUDA_VISIBLE_DEVICES=3 python afn.py data/office-home -d OfficeHome -t Rw -a resnet50 --epochs 30  --seed 0 --log logs/afn/OfficeHome_:2Rw

# DomainNet
#CUDA_VISIBLE_DEVICES=0 python afn.py data/domainnet -d DomainNet -t c -a resnet101  --epochs 40 -i 5000 -p 500  --seed 0 --lr 0.004 --log logs/afn/DomainNet_:2c
#CUDA_VISIBLE_DEVICES=0 python afn.py data/domainnet -d DomainNet -t i -a resnet101  --epochs 40 -i 5000 -p 500  --seed 0 --lr 0.004 --log logs/afn/DomainNet_:2i
#CUDA_VISIBLE_DEVICES=0 python afn.py data/domainnet -d DomainNet -t p -a resnet101  --epochs 40 -i 5000 -p 500  --seed 0 --lr 0.004 --log logs/afn/DomainNet_:2p
#CUDA_VISIBLE_DEVICES=0 python afn.py data/domainnet -d DomainNet -t q -a resnet101  --epochs 40 -i 5000 -p 500  --seed 0 --lr 0.004 --log logs/afn/DomainNet_:2q
#CUDA_VISIBLE_DEVICES=0 python afn.py data/domainnet -d DomainNet -t r -a resnet101  --epochs 40 -i 5000 -p 500  --seed 0 --lr 0.004 --log logs/afn/DomainNet_:2r
#CUDA_VISIBLE_DEVICES=0 python afn.py data/domainnet -d DomainNet -t s -a resnet101  --epochs 40 -i 5000 -p 500  --seed 0 --lr 0.004 --log logs/afn/DomainNet_:2s

