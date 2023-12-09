"""
## ECCV 2022
"""
import matplotlib.pyplot as plt
# --- Imports --- #
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.utils
from torch.utils.tensorboard import SummaryWriter
import cv2
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import numpy as np
from torch.nn import init
from lpls_decomposition import lplas_decomposition as decomposition

writer = SummaryWriter(log_dir='./log/train_loss', flush_secs=10)

class SEAttention(nn.Module):

    def __init__(self, channel=1024, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class cnnblock(nn.Module):
    def __init__(self, in_channle, out_channle):
        super(cnnblock, self).__init__()
        self.cnn_conv1 = nn.Conv2d(in_channle, out_channle, 3, 1, 1)
        self.ac1 = nn.LeakyReLU(inplace=True)

        self.cnn_conv2 = nn.Conv2d(out_channle, out_channle, 3, 1, 1)
        self.ac2 = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.cnn_conv1(x.float())
        x = self.ac1(x)
        x = self.cnn_conv2(x)
        x = self.ac2(x)
        return x

class Upsample(nn.Module):
    """Upscaling"""
    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        self.bilinear = bilinear
        if self.bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        else:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.ac = nn.LeakyReLU(inplace=True)

    def forward(self, x, shape1, shape2):
        x = self.up(x)
        # input is CHW
        diffY = shape1 - x.shape[2]
        diffX = shape2 - x.shape[3]
        if self.bilinear:
            x = self.conv(x)
        x = self.ac(x)
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        return x

class UNet_4layers(nn.Module):
    def __init__(self, firstoutputchannl=64):
        super(UNet_4layers, self).__init__()
        self.outputchannl = 3
        self.block1 = cnnblock(3, firstoutputchannl)
        self.maxpool = nn.MaxPool2d(2)
        self.block2 = cnnblock(firstoutputchannl, 2*firstoutputchannl)
        self.block3 = cnnblock(2*firstoutputchannl, 4*firstoutputchannl)
        self.block4 = cnnblock(4*firstoutputchannl, 8*firstoutputchannl)
        self.block5 = cnnblock(8*firstoutputchannl, 16*firstoutputchannl)
        self.SEAttention = SEAttention(16*firstoutputchannl)

        self.up1 = Upsample(16*firstoutputchannl, 8*firstoutputchannl)
        self.block6 = cnnblock(16*firstoutputchannl, 8*firstoutputchannl)

        self.up2 = Upsample(8*firstoutputchannl, 4*firstoutputchannl)
        self.block7 = cnnblock(8*firstoutputchannl, 4*firstoutputchannl)

        self.up3 = Upsample(4*firstoutputchannl, 2*firstoutputchannl)
        self.block8 = cnnblock(4*firstoutputchannl, 2*firstoutputchannl)

        self.up4 = Upsample(2*firstoutputchannl, firstoutputchannl)
        self.block9 = cnnblock(2*firstoutputchannl, firstoutputchannl)
        self.finalconv = nn.Conv2d(firstoutputchannl, self.outputchannl, 1, 1, 0)

    def forward(self, x):
        out1 = self.block1(x)
        out2 = self.block2(self.maxpool(out1))
        out3 = self.block3(self.maxpool(out2))
        out4 = self.block4(self.maxpool(out3))
        out5 = self.block5(self.maxpool(out4))
        se_attention = self.SEAttention(out5)
        in6 = torch.cat([self.up1(se_attention, out4.shape[2], out4.shape[3]), out4], 1)
        out6 = self.block6(in6)
        in7 = torch.cat([self.up2(out6, out3.shape[2], out3.shape[3]), out3], 1)
        out7 = self.block7(in7)
        in8 = torch.cat([self.up3(out7, out2.shape[2], out2.shape[3]), out2], 1)
        out8 = self.block8(in8)
        in9 = torch.cat([self.up4(out8, out1.shape[2], out1.shape[3]),  out1], 1)
        out9 = self.block9(in9)
        predict = self.finalconv(out9)
        return predict


class Spa_unet_2layers(nn.Module):
    def __init__(self, firstoutputchannl=64):
        super(Spa_unet_2layers, self).__init__()
        self.outputchannl = 3
        self.maxpool = nn.MaxPool2d(2)
        self.block1 = cnnblock(3, firstoutputchannl)
        self.block2 = cnnblock(firstoutputchannl, 2 * firstoutputchannl)
        self.block3 = cnnblock(2 * firstoutputchannl, 4 * firstoutputchannl)

        self.SEAttention = SEAttention(4 * firstoutputchannl)

        self.up1 = Upsample(4 * firstoutputchannl, 2 * firstoutputchannl)
        self.block4 = cnnblock(4 * firstoutputchannl, 2 * firstoutputchannl)
        self.up2 = Upsample(2 * firstoutputchannl, firstoutputchannl)
        self.block5 = cnnblock(2 * firstoutputchannl, firstoutputchannl)
        self.finalconv = nn.Conv2d(firstoutputchannl, self.outputchannl, 1, 1, 0)

    def forward(self, x):
        out1 = self.block1(x)
        out2 = self.block2(self.maxpool(out1))
        out3 = self.block3(self.maxpool(out2))
        se_attention = self.SEAttention(out3)
        in4 = torch.cat([self.up1(se_attention, out2.shape[2], out2.shape[3]), out2], 1)
        out4 = self.block4(in4)
        in5 = torch.cat([self.up2(out4, out1.shape[2], out1.shape[3]), out1], 1)
        out5 = self.block5(in5)
        predict = self.finalconv(out5)
        # torchvision.utils.save_image(x, './results/process/' + '/Spatial_result.jpg')
        return predict


class Spa_unet_3layers(nn.Module):
    def __init__(self, firstoutputchannl=64):
        super(Spa_unet_3layers, self).__init__()
        self.outputchannl = 3
        self.maxpool = nn.MaxPool2d(2)
        self.block1 = cnnblock(3, firstoutputchannl)
        self.block2 = cnnblock(firstoutputchannl, 2 * firstoutputchannl)
        self.block3 = cnnblock(2 * firstoutputchannl, 4 * firstoutputchannl)
        self.block4 = cnnblock(4 * firstoutputchannl, 8 * firstoutputchannl)

        self.SEAttention = SEAttention(8 * firstoutputchannl)

        self.up1 = Upsample(8 * firstoutputchannl, 4 * firstoutputchannl)
        self.block5 = cnnblock(8 * firstoutputchannl, 4 * firstoutputchannl)
        self.up2 = Upsample(4 * firstoutputchannl, 2 * firstoutputchannl)
        self.block6 = cnnblock(4 * firstoutputchannl, 2 * firstoutputchannl)
        self.up3 = Upsample(2 * firstoutputchannl, firstoutputchannl)
        self.block7 = cnnblock(2 * firstoutputchannl, firstoutputchannl)
        self.finalconv = nn.Conv2d(firstoutputchannl, self.outputchannl, 1, 1, 0)

    def forward(self, x):
        out1 = self.block1(x)
        out2 = self.block2(self.maxpool(out1))
        out3 = self.block3(self.maxpool(out2))
        out4 = self.block4(self.maxpool(out3))
        se_attention = self.SEAttention(out4)
        in5 = torch.cat([self.up1(se_attention, out3.shape[2], out3.shape[3]), out3], 1)
        out5 = self.block5(in5)
        in6 = torch.cat([self.up2(out5, out2.shape[2], out2.shape[3]), out2], 1)
        out6 = self.block6(in6)
        in7 = torch.cat([self.up3(out6, out1.shape[2], out1.shape[3]), out1], 1)
        out7 = self.block7(in7)
        predict = self.finalconv(out7)
        # torchvision.utils.save_image(x, './results/process/' + '/Spatial_result.jpg')
        return predict

class Spa_unet_4layers(nn.Module):
    def __init__(self, firstoutputchannl=64):
        super(Spa_unet_4layers, self).__init__()
        self.outputchannl = 3
        self.maxpool = nn.MaxPool2d(2)
        self.block1 = cnnblock(3, firstoutputchannl)
        self.block2 = cnnblock(firstoutputchannl, 2 * firstoutputchannl)
        self.block3 = cnnblock(2 * firstoutputchannl, 4 * firstoutputchannl)
        self.block4 = cnnblock(4 * firstoutputchannl, 8 * firstoutputchannl)
        self.block5 = cnnblock(8 * firstoutputchannl, 16 * firstoutputchannl)
        self.SEAttention = SEAttention(16 * firstoutputchannl)

        self.up1 = Upsample(16 * firstoutputchannl, 8 * firstoutputchannl)
        self.block6 = cnnblock(16 * firstoutputchannl, 8 * firstoutputchannl)
        self.up2 = Upsample(8 * firstoutputchannl, 4 * firstoutputchannl)
        self.block7 = cnnblock(8 * firstoutputchannl, 4 * firstoutputchannl)
        self.up3 = Upsample(4 * firstoutputchannl, 2 * firstoutputchannl)
        self.block8 = cnnblock(4 * firstoutputchannl, 2 * firstoutputchannl)
        self.up4 = Upsample(2 * firstoutputchannl, firstoutputchannl)
        self.block9 = cnnblock(2 * firstoutputchannl, firstoutputchannl)
        self.finalconv = nn.Conv2d(firstoutputchannl, self.outputchannl, 1, 1, 0)

    def forward(self, x):
        out1 = self.block1(x)
        out2 = self.block2(self.maxpool(out1))
        out3 = self.block3(self.maxpool(out2))
        out4 = self.block4(self.maxpool(out3))
        out5 = self.block5(self.maxpool(out4))
        se_attention = self.SEAttention(out5)
        in6 = torch.cat([self.up1(se_attention, out4.shape[2], out4.shape[3]), out4], 1)
        out6 = self.block6(in6)
        in7 = torch.cat([self.up2(out6, out3.shape[2], out3.shape[3]), out3], 1)
        out7 = self.block7(in7)
        in8 = torch.cat([self.up3(out7, out2.shape[2], out2.shape[3]), out2], 1)
        out8 = self.block8(in8)
        in9 = torch.cat([self.up4(out8, out1.shape[2], out1.shape[3]), out1], 1)
        out9 = self.block9(in9)
        predict = self.finalconv(out9)
        # torchvision.utils.save_image(x, './results/process/' + '/Spatial_result.jpg')
        # save_img3('/home/wanghuaiyao/UHDFour-pro1/UHDFour_code-main/src/results/predict', predict.detach())
        return predict

class FreBlock(nn.Module):
    def __init__(self, nc):
        super(FreBlock, self).__init__()
        self.processmag = nn.Sequential(
            nn.Conv2d(3, nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, 3, 1, 1, 0))
        self.processpha = nn.Sequential(
            nn.Conv2d(3, nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, 3, 1, 1, 0))
        self.finalConv = nn.Conv2d(3, 3, 1, 1, 0)

    def forward(self, x):  #输入x的形式  x = mag * exp(i * pha)
        mag = torch.abs(x) #提取振幅
        # save_img2(mag.cpu(), '/home/wanghuaiyao/UHDFour-pro1/UHDFour_code-main/src/results/')
        pha = torch.angle(x) #以弧度表示的相位
        # save_img2(mag.cpu(), '/home/wanghuaiyao/UHDFour-pro1/UHDFour_code-main/src/results/pha')
        mag = self.processmag(mag)
        # save_img2(mag.cpu(), '/home/wanghuaiyao/UHDFour-pro1/UHDFour_code-main/src/results/process')
        pha = self.processpha(pha)
        # save_img2(mag.cpu(), '/home/wanghuaiyao/UHDFour-pro1/UHDFour_code-main/src/results/pha_prosess')
        real = mag * torch.cos(pha)
        # save_img2(mag.cpu(), '/home/wanghuaiyao/UHDFour-pro1/UHDFour_code-main/src/results/real')
        imag = mag * torch.sin(pha)
        # save_img2(mag.cpu(), '/home/wanghuaiyao/UHDFour-pro1/UHDFour_code-main/src/results/imag')
        x_out = torch.complex(real, imag)
        # save_img2(mag.cpu(), '/home/wanghuaiyao/UHDFour-pro1/UHDFour_code-main/src/results/out')
        # torchvision.utils.save_image(x_out, './results/process/' + '/Fre_result.jpg')

        return x_out


class ProcessBlock(nn.Module):  #结合空间域和频率域
    def __init__(self, in_nc):
        super(ProcessBlock, self).__init__()
        self.spatial_process = Spa_unet_2layers(in_nc)
        self.frequency_process = FreBlock(in_nc)
        self.cat = nn.Conv2d(3, 3, 1, 1, 0)
        # self.conv = nn.Conv2d(3, 3, 1, 1, 0)


    def forward(self, x):
        # 同时使用空间加频域处理
        xori = x
        _, _, H, W = x.shape
        x_freq = torch.fft.rfft2(x, s=(H, W), norm='backward')
        x = self.spatial_process(x)
        x_freq = self.frequency_process(x_freq)
        x_freq_spatial = torch.fft.irfft2(x_freq, s=(H, W), norm='backward')
        # x_freq_spatial1 = self.conv(x_freq_spatial)
        xcat = x + x_freq_spatial + xori
        x_out = self.cat(xcat)
        # print('[En_arch 180]H{}, W{} x_out.shape:{}'.format(H, W, x_out.shape))
        # torchvision.utils.save_image(x_out+xori, './results/' + '/block_result.jpg')

        '''# 去掉频域处理
        xori = x
        _, _, H, W = x.shape
        # x_freq = torch.fft.rfft2(x, s=(H, W), norm='backward')
        # print('[En_arch 120] x.shape:{}', x.shape)
        x = self.spatial_process(x)
        xcat = x + xori
        # print('[En_arch 235] xcat.shape:{}', xcat.shape)
        x_out = self.cat(xcat)
        '''
        '''# 去掉空间域
        xori = x
        _, _, H, W = x.shape
        x_freq = torch.fft.rfft2(x, s=(H, W), norm='backward')
        x_freq = self.frequency_process(x_freq)
        x_freq_spatial = torch.fft.irfft2(x_freq, s=(H, W), norm='backward')
        xcat = x_freq_spatial + xori
        # print('[En_arch 235] xcat.shape:{}', xcat.shape)
        x_out = self.cat(xcat)
        '''
        return x_out+xori

class ProcessBlock1(nn.Module):  #结合空间域和频率域
    def __init__(self, in_nc):
        super(ProcessBlock1, self).__init__()
        self.spatial_process = Spa_unet_3layers(in_nc)
        self.frequency_process = FreBlock(in_nc)
        self.cat = nn.Conv2d(3, 3, 1, 1, 0)  #本来是2*inc
        # self.conv = nn.Conv2d(3, 3, 1, 1, 0)

    def forward(self, x):
         # 同时使用空间加频域处理
        xori = x
        _, _, H, W = x.shape
        x_freq = torch.fft.rfft2(x, s=(H, W), norm='backward')
        # print('[En_arch 120] x.shape:{}', x.shape)
        x = self.spatial_process(x)
        x_freq = self.frequency_process(x_freq)
        x_freq_spatial = torch.fft.irfft2(x_freq, s=(H, W), norm='backward')
        # x_freq_spatial1 = self.conv(x_freq_spatial)
        xcat = x + x_freq_spatial + xori
        x_out = self.cat(xcat)
        # print('[En_arch 180]H{}, W{} x_out.shape:{}'.format(H, W, x_out.shape))
        # torchvision.utils.save_image(x_out+xori, './results/' + '/block_result.jpg')

        ''''# 去掉频域
        xori = x
        _, _, H, W = x.shape
        # x_freq = torch.fft.rfft2(x, s=(H, W), norm='backward')
        # print('[En_arch 120] x.shape:{}', x.shape)
        x = self.spatial_process(x)
        xcat = x + xori
        # print('[En_arch 235] xcat.shape:{}', xcat.shape)
        x_out = self.cat(xcat)
        '''
        '''# 去掉空间域
        xori = x
        _, _, H, W = x.shape
        x_freq = torch.fft.rfft2(x, s=(H, W), norm='backward')
        x_freq = self.frequency_process(x_freq)
        x_freq_spatial = torch.fft.irfft2(x_freq, s=(H, W), norm='backward')
        xcat = x_freq_spatial + xori
        # print('[En_arch 235] xcat.shape:{}', xcat.shape)
        x_out = self.cat(xcat)
        '''
        return x_out + xori


class ProcessBlock2(nn.Module):  #结合空间域和频率域
    def __init__(self, in_nc):
        super(ProcessBlock2, self).__init__()
        self.spatial_process = Spa_unet_4layers(in_nc)
        self.frequency_process = FreBlock(in_nc)
        self.cat = nn.Conv2d(3, 3, 1, 1, 0)  #本来是2*inc
        # self.conv = nn.Conv2d(3, 3, 1, 1, 0)

    def forward(self, x):
        # 同时使用空间加频域处理
        xori = x
        _, _, H, W = x.shape
        x_freq = torch.fft.rfft2(x, s=(H, W), norm='backward')
        # print('[En_arch 120] x.shape:{}', x.shape)
        x = self.spatial_process(x)
        x_freq = self.frequency_process(x_freq)
        x_freq_spatial = torch.fft.irfft2(x_freq, s=(H, W), norm='backward')
        # x_freq_spatial1 = self.conv(x_freq_spatial)
        xcat = x + x_freq_spatial + xori
        x_out = self.cat(xcat)
        # print('[En_arch 180]H{}, W{} x_out.shape:{}'.format(H, W, x_out.shape))
        # torchvision.utils.save_image(x_out+xori, './results/' + '/block_result.jpg')

        ''''# 去掉频域处理
        xori = x
        _, _, H, W = x.shape
        # x_freq = torch.fft.rfft2(x, s=(H, W), norm='backward')
        # print('[En_arch 120] x.shape:{}', x.shape)
        x = self.spatial_process(x)
        xcat = x + xori
        # print('[En_arch 235] xcat.shape:{}', xcat.shape)
        x_out = self.cat(xcat)
        '''
        ''' # 去掉空间域
        xori = x
        _, _, H, W = x.shape
        x_freq = torch.fft.rfft2(x, s=(H, W), norm='backward')
        x_freq = self.frequency_process(x_freq)
        x_freq_spatial = torch.fft.irfft2(x_freq, s=(H, W), norm='backward')
        xcat = x_freq_spatial + xori
        # print('[En_arch 235] xcat.shape:{}', xcat.shape)
        x_out = self.cat(xcat)
        '''
        return x_out + xori


class InteractNet(nn.Module):
    def __init__(self):
        super(InteractNet, self).__init__()
        # self.extract = nn.Conv2d(3, nc//2, 1, 1, 0)
        self.unet = UNet_4layers(24)
        self.process = ProcessBlock(24)  #这里输入通道不确定啊
        self.process1 = ProcessBlock1(24)
        self.process2 = ProcessBlock2(16)
        self.up = Upsample(3, 3)
        self.up1 = Upsample(3, 3)
        self.up2 = Upsample(3, 3)
        self.up3 = Upsample(3, 3)

    def forward(self, L_list, data):
        # print('[En_arch 143] len(L_list):{}, L_list[0].shape:{},L_list[1].shape:{}, L_list[2].shape:{}'.format(len(L_list), L_list[0].shape, L_list[1].shape, L_list[2].shape))
        # torchvision.utils.save_image(L_list[2], './results/process/' + '/L_list_output3.jpg')  #输入到目前为止是对的
        # torchvision.utils.save_image(L_list[0], './results/process/' + '/L_list_output1.jpg')
        # torchvision.utils.save_image(L_list[1], './results/process/' + '/L_list_output2.jpg')
        #完整版代码
        y = self.unet(data)
        y1_0 = self.process(L_list[0])
        # torchvision.utils.save_image(y1_0, './results/process/' + '/y10.jpg')
        y1_1 = self.up1(y1_0, L_list[1].shape[2], L_list[1].shape[3])
        # torchvision.utils.save_image(y1_1, './results/process/' + '/y11.jpg')
        # print('[En_arch 214] y1_1.shape:{}'.format(y1_1.shape))
        y2_0 = self.process1(y1_1 + L_list[1]) + y1_1
        # torchvision.utils.save_image(y2_0, './results/process/' + '/y20.jpg')
        # print('[En_arch 217] y2_0.shape:{}'.format(y2_0.shape))
        y2_1 = self.up2(y2_0, L_list[2].shape[2], L_list[2].shape[3])
        # torchvision.utils.save_image(y2_1, './results/process/' + '/y21.jpg')
        # print('[En_arch 220] y2_1.shape:{}'.format(y2_1.shape))
        y3_0 = self.process2(y2_1 + L_list[2]) + y2_1
        # torchvision.utils.save_image(y3_0, './results/process/' + '/y30.jpg')
        # print('[En_arch 223] y3_0.shape:{}'.format(y3_0.shape))
        y3_1 = self.up3(y3_0, L_list[3].shape[2], L_list[3].shape[3])
        # print('[Eh_arch 150] y1_1.shape:{},y2_1.shape:{},y3_0.shape:{}'.format(y1_1.shape, y2_1.shape, y3_0.shape))
        print('[456 SMF_arch] shape:{},{}'.format(y3_1.size(), y.size()))
        y3_2 = y3_1 + y
        Y_list = [y1_0, y2_0, y3_2, y]
        torchvision.utils.save_image(y, './results/process/' + 'Unet_output.jpg')
        torchvision.utils.save_image(Y_list[2], './results/process/' + 'Y_list_output3.jpg')
        torchvision.utils.save_image(Y_list[0], './results/process/' + 'Y_list_output1.jpg')
        torchvision.utils.save_image(Y_list[1], './results/process/' + 'Y_list_output2.jpg')

        '''
        # 没有Unet
        y1_0 = self.process(L_list[0])
        y1_1 = self.up1(y1_0, L_list[1].shape[2], L_list[1].shape[3])
        y2_0 = self.process1(y1_1 + L_list[1]) + y1_1
        y2_1 = self.up2(y2_0, L_list[2].shape[2], L_list[2].shape[3])
        y3_0 = self.process2(y2_1 + L_list[2]) + y2_1
        y3_2 = y3_0
        Y_list = [y1_0, y2_0, y3_2, data]
        torchvision.utils.save_image(Y_list[2], './results/process/' + '/Y_list_output3.jpg')
        torchvision.utils.save_image(Y_list[0], './results/process/' + '/Y_list_output1.jpg')
        torchvision.utils.save_image(Y_list[1], './results/process/' + '/Y_list_output2.jpg')
        '''
        '''# 没有下半频域
        y = self.unet(data)
        Y_list = [y, y, y, y]
        torchvision.utils.save_image(y, './results/process/' + '/Unet_output.jpg')
        torchvision.utils.save_image(Y_list[2], './results/process/' + '/Y_list_output3.jpg')
        torchvision.utils.save_image(Y_list[0], './results/process/' + '/Y_list_output1.jpg')
        torchvision.utils.save_image(Y_list[1], './results/process/' + '/Y_list_output2.jpg')
        '''
        return Y_list

