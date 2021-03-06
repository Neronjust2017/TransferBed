#!/usr/bin/env bash
source /home/weiyuhua/Code/TransferBed/env/bin/activate

#Camelyon17
CUDA_VISIBLE_DEVICES=0 python mdd.py data/Camelyon17 -d Camelyon17 -t H0 -a resnet18 --epochs 30 --bottleneck-dim 2048 --seed 0 --log logs/mdd/Camelyon17_:2H0
CUDA_VISIBLE_DEVICES=0 python mdd.py data/Camelyon17 -d Camelyon17 -t H1 -a resnet18 --epochs 30 --bottleneck-dim 2048 --seed 0 --log logs/mdd/Camelyon17_:2H1
CUDA_VISIBLE_DEVICES=0 python mdd.py data/Camelyon17 -d Camelyon17 -t H2 -a resnet18 --epochs 30 --bottleneck-dim 2048 --seed 0 --log logs/mdd/Camelyon17_:2H2
CUDA_VISIBLE_DEVICES=0 python mdd.py data/Camelyon17 -d Camelyon17 -t H3 -a resnet18 --epochs 30 --bottleneck-dim 2048 --seed 0 --log logs/mdd/Camelyon17_:2H3
CUDA_VISIBLE_DEVICES=0 python mdd.py data/Camelyon17 -d Camelyon17 -t H4 -a resnet18 --epochs 30 --bottleneck-dim 2048 --seed 0 --log logs/mdd/Camelyon17_:2H4

## PACS
#CUDA_VISIBLE_DEVICES=0 python mdd.py data/pacs -d PACS -t Ar -a resnet18 --epochs 30 --bottleneck-dim 2048 --seed 0 --log logs/mdd/PACS_:2Ar
#CUDA_VISIBLE_DEVICES=0 python mdd.py data/pacs -d PACS -t Ca -a resnet18 --epochs 30 --bottleneck-dim 2048 --seed 0 --log logs/mdd/PACS_:2Ca
#CUDA_VISIBLE_DEVICES=0 python mdd.py data/pacs -d PACS -t Ph -a resnet18 --epochs 30 --bottleneck-dim 2048 --seed 0 --log logs/mdd/PACS_:2Ph
#CUDA_VISIBLE_DEVICES=0 python mdd.py data/pacs -d PACS -t Sk -a resnet18 --epochs 30 --bottleneck-dim 2048 --seed 0 --log logs/mdd/PACS_:2Sk
#
## Office-Home
#CUDA_VISIBLE_DEVICES=0 python mdd.py data/office-home -d OfficeHome -t Ar -a resnet50 --epochs 30 --bottleneck-dim 2048 --seed 0 --log logs/mdd/OfficeHome_:2Ar
#CUDA_VISIBLE_DEVICES=0 python mdd.py data/office-home -d OfficeHome -t Cl -a resnet50 --epochs 30 --bottleneck-dim 2048 --seed 0 --log logs/mdd/OfficeHome_:2Cl
#CUDA_VISIBLE_DEVICES=0 python mdd.py data/office-home -d OfficeHome -t Pr -a resnet50 --epochs 30 --bottleneck-dim 2048 --seed 0 --log logs/mdd/OfficeHome_:2Pr
#CUDA_VISIBLE_DEVICES=0 python mdd.py data/office-home -d OfficeHome -t Rw -a resnet50 --epochs 30 --bottleneck-dim 2048 --seed 0 --log logs/mdd/OfficeHome_:2Rw

# DomainNet
#CUDA_VISIBLE_DEVICES=0 python mdd.py data/domainnet -d DomainNet -t c -a resnet101  --epochs 40 -i 5000 -p 500 --bottleneck-dim 2048 --seed 0 --lr 0.004 --log logs/mdd/DomainNet_:2c
#CUDA_VISIBLE_DEVICES=0 python mdd.py data/domainnet -d DomainNet -t i -a resnet101  --epochs 40 -i 5000 -p 500 --bottleneck-dim 2048 --seed 0 --lr 0.004 --log logs/mdd/DomainNet_:2i
#CUDA_VISIBLE_DEVICES=0 python mdd.py data/domainnet -d DomainNet -t p -a resnet101  --epochs 40 -i 5000 -p 500 --bottleneck-dim 2048 --seed 0 --lr 0.004 --log logs/mdd/DomainNet_:2p
#CUDA_VISIBLE_DEVICES=0 python mdd.py data/domainnet -d DomainNet -t q -a resnet101  --epochs 40 -i 5000 -p 500 --bottleneck-dim 2048 --seed 0 --lr 0.004 --log logs/mdd/DomainNet_:2q
#CUDA_VISIBLE_DEVICES=0 python mdd.py data/domainnet -d DomainNet -t r -a resnet101  --epochs 40 -i 5000 -p 500 --bottleneck-dim 2048 --seed 0 --lr 0.004 --log logs/mdd/DomainNet_:2r
#CUDA_VISIBLE_DEVICES=0 python mdd.py data/domainnet -d DomainNet -t s -a resnet101  --epochs 40 -i 5000 -p 500 --bottleneck-dim 2048 --seed 0 --lr 0.004 --log logs/mdd/DomainNet_:2s

