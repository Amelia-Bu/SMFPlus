#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from SMF_model import SMF
from argparse import ArgumentParser
# from train_data_aug_local import TrainData
from dataloader import TrainData, ValidData
import os


def parse_args():
    """Command-line argument parser for training."""

    # New parser
    parser = ArgumentParser(description='PyTorch implementation of UHDFour from Li et al. (2023)')

    # Data parameters
    parser.add_argument('-d', '--dataset-name', help='name of dataset', choices=['UHD', 'LOLv1', 'LOLv2', 'NewDataSet'],
                        default='LOLv1')
    parser.add_argument('-t', '--train-dir', help='training set path', default='./../data/LOLv1/our485/')
    parser.add_argument('-v', '--valid-dir', help='test set path', default='./../data/LOLv1/eval15/')
    # parser.add_argument('-t', '--train-dir', help='training set path', default='./../data/LOLv2/Synthetic/Train/')
    # parser.add_argument('-v', '--valid-dir', help='test set path', default='./../data/LOLv2/Synthetic/Test/')
    # parser.add_argument('--ckpt-save-path', help='checkpoint save path', default='./../ckpts_training_LOLv2_Syn/')

    # parser.add_argument('-t', '--train-dir', help='training set path', default='./../data/NewDataSet/train/')
    # parser.add_argument('-v', '--valid-dir', help='test set path', default='./../data/NewDataSet/eval/')
    parser.add_argument('--ckpt-save-path', help='checkpoint save path', default='./../ckpts_training_ablation/')
    parser.add_argument('--ckpt-overwrite', help='overwrite model checkpoint on save', action='store_true')
    # parser.add_argument('--ckpt-load-path', help='start training with a pretrained model', default='/home/wanghuaiyao/UHDFour-pro1/UHDFour_code-main/ckpts_training/LOLv1-0930-1230/UHDFour-epoch984-0.07820.pt')
    # parser.add_argument('--ckpt-load-path', help='start training with a pretrained model', default='/home/wanghuaiyao/UHDFour-pro1/UHDFour_code-main/ckpts_training_LOLv2_real/LOLv1-1010-1633/UHDFour-epoch1-0.03626.pt')
    parser.add_argument('--ckpt-load-path', help='start training with a pretrained model',
                        default=None)
    parser.add_argument('--report-interval', help='batch report interval', default=1, type=int)
    parser.add_argument('-ts', '--train-size', nargs='+', help='size of train dataset', default=[400, 600],
                        type=int)  # [400,600]
    # Training hyperparameters
    parser.add_argument('-lr', '--learning-rate', help='learning rate', default=0.0001, type=float)
    parser.add_argument('-a', '--adam', help='adam parameters', nargs='+', default=[0.9, 0.99, 1e-8], type=list)
    parser.add_argument('-b', '--batch-size', help='minibatch size', default=6, type=int)  # 6
    parser.add_argument('-e', '--nb-epochs', help='number of epochs', default=1000, type=int)
    parser.add_argument('--cuda', default=True, help='use cuda', action='store_true')
    parser.add_argument('--gpu', type=str, default='1', help='gpu device id')
    parser.add_argument('--plot-stats', help='plot stats after every epoch', action='store_true')
    parser.add_argument('-s', '--seed', help='fix random seed', type=int)
    return parser.parse_args()


if __name__ == '__main__':
    """Trains UHDFour."""

    # Parse training parameters
    params = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = params.gpu
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    # --- Load training data and validation/test data --- # params.train_size
    train_loader = DataLoader(TrainData(params.train_size, params.train_dir), batch_size=params.batch_size, shuffle=True
                              , num_workers=2, pin_memory=True, prefetch_factor=8)
    valid_loader = DataLoader(ValidData(params.train_size, params.valid_dir), batch_size=1, shuffle=False,
                              pin_memory=True)
    # 实例化UHDFour
    UHDFour = SMF(params, trainable=True)
    UHDFour.train(train_loader, valid_loader)


