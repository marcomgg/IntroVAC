#!/bin/bash
DATASET_ROOT='/datasets'

CUDA_VISIBLE_DEVICES=0 python -m introvac with \
gpu=0 \
betas='[10.0, 100.0, 3.0, 0.01, 5.0]' \
epochs=30 \
attributes='["Eyeglasses"]' \
log_freq=100 \
binary=True \
ratio=2 \
gamma=0.5 \
batch_size=64 \
lr=2E-4 \
pretrain=5 \
lrs='[60, 90, 120]' \
dataset_root=$DATASET_ROOT

CUDA_VISIBLE_DEVICES=0 python -m introvac with \
gpu=0 \
betas='[10.0, 100.0, 3.0, 0.01, 5.0]' \
epochs=150 \
attributes='["FacialHair"]' \
log_freq=100 \
binary=True \
ratio=1 \
gamma=0.5 \
batch_size=64 \
lr=2E-4 \
pretrain=5 \
lrs='[60, 90, 120]' \
dataset_root=$DATASET_ROOT

CUDA_VISIBLE_DEVICES=0 python -m introvac with \
gpu=0 \
betas='[10.0, 100.0, 3.0, 0.01, 5.0]' \
epochs=150 \
attributes='["Eyeglasses","FacialHair"]' \
log_freq=100 \
binary=False \
ratio=1 \
gamma=0.5 \
batch_size=64 \
lr=2E-4 \
pretrain=5 \
lrs='[60, 90, 120]' \
dataset_root=$DATASET_ROOT

# Standard model
CUDA_VISIBLE_DEVICES=0 python -m introvac with \
gpu=0 \
betas='[10.0, 100.0, 3.0, 0.01, 5.0]' \
epochs=600 \
attributes='["FacialHair"]' \
log_freq=100 \
binary=True \
ratio=1 \
gamma=0.5 \
batch_size=64 \
lr=2E-4 \
pretrain=601 \
lrs='[]' \
dataset_root=$DATASET_ROOT