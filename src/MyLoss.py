import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.models.vgg import vgg16
import numpy as np
from torch.utils.tensorboard import SummaryWriter

class layer_Loss (nn.Module):
    def __init__(self, weight=1.0):
        super(layer_Loss, self).__init__()
        self.weight = weight
        self.criterion = nn.L1Loss(reduction='sum')
    def forward(self, Y_list, T_list):
        n = len(Y_list)
        loss = 0
        # for m in range(0, n-1):
        #     loss += self.weight*(2**(n-m-2))*self.criterion(Y_list[m], F.interpolate(T_list[m],(Y_list[m].shape[2],Y_list[m].shape[3]),mode='bilinear',align_corners=True))/Y_list[m].shape[0]
        loss += self.weight*(2**2)*self.criterion(Y_list[1], F.interpolate(T_list[0],(Y_list[1].shape[2],Y_list[1].shape[3]),mode='bilinear',align_corners=True))/Y_list[1].shape[0]
        loss += self.weight*(2**1)*self.criterion(Y_list[2], F.interpolate(T_list[1],(Y_list[2].shape[2],Y_list[2].shape[3]),mode='bilinear',align_corners=True))/Y_list[2].shape[0]

        return loss

class Rec_Loss(nn.Module):
    def __init__(self, weight=1):
        super(Rec_Loss, self).__init__()
        self.weight = weight
        self.criterion = nn.L1Loss(reduction='sum')
    def forward(self, Y_list, T_list):
        loss = self.weight * self.criterion(Y_list[-1], T_list[-1])/Y_list[-1].shape[0]
        return loss

class My_loss(nn.Module):
    def __init__(self, layer_weight=1.0, Rec_weight=1.0):
        super(My_loss, self).__init__()
        self.layer_loss = layer_Loss(layer_weight)
        self.rec_loss = Rec_Loss(Rec_weight)
    def forward(self, Y_list, T_list, P_Y = None, withoutadvloss=False):

        layerloss = self.layer_loss(Y_list, T_list)
        recloss = self.rec_loss(Y_list, T_list)
        myloss = layerloss + recloss
        return recloss, layerloss, myloss