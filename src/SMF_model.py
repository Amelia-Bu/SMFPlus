#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam
import torchvision

from utils import *
import os
import json
import networks
# from logger import logger
from torch.utils.tensorboard import SummaryWriter
from MyLoss import My_loss
import torch.nn.functional as F

writer = SummaryWriter(log_dir='./log/train_loss', flush_secs=10)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

class SMF(object):
    """Implementation of UHDFour from Li et al. (2023)."""

    def __init__(self, params, trainable):
        """Initializes model."""

        self.p = params
        self.trainable = trainable
        self._compile()

    def _compile(self):
        """Compiles model (architecture, loss function, optimizers, etc.)."""

        print(' UHDFour from Li et al. (2023)')

        # Model (3x3=9 channels for Monte Carlo since it uses 3 HDR buffers)
        if self.p.dataset_name == 'UHD':
            from SMF_arch import InteractNet as SMF_Net
        else:
            # from Enhance_arch_MF_4layers import InteractNet as UHD_Net
            from SMF_arch import InteractNet as SMF_Net
        self.model = SMF_Net()
        total_params = count_parameters(self.model)
        print(f"模型的总参数量为 {total_params} 个。")
        # Set optimizer and loss, if in training mode
        if self.trainable:
            # self.optim = Adam(self.model.parameters(),
            #                   lr=self.p.learning_rate,
            #                   betas=self.p.adam[:2],
            #                   eps=self.p.adam[2])
            self.optim = Adam(self.model.parameters(),
                              lr=self.p.learning_rate,
                              weight_decay=0.0001)

            # 热重启学习率策略
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optim, T_0=2,
                                                                                  T_mult=2)  # CosineAnnealingLR

        # CUDA support
        self.L1 = nn.L1Loss()
        self.L2 = nn.MSELoss()
        self.use_cuda = torch.cuda.is_available() and self.p.cuda
        if self.use_cuda:
            self.model = self.model.cuda()
            if self.trainable:
                self.L1 = self.L1.cuda()
                self.L2 = self.L2.cuda()
        # self.model = torch.nn.DataParallel(self.model)

    def _print_params(self):
        """Formats parameters to print when training."""

        # print('Training parameters: ')
        logger.info('---Training parameters---')
        self.p.cuda = self.use_cuda
        param_dict = vars(self.p)
        pretty = lambda x: x.replace('_', ' ').capitalize()
        # print('\n'.join('  {} = {}'.format(pretty(k), str(v)) for k, v in param_dict.items()))
        # print()
        logger.info('\n'.join('  {} = {}'.format(pretty(k), str(v)) for k, v in param_dict.items()))
        logger.info('')

    def save_model(self, epoch, stats, first=False):
        """Saves model to files; can be overwritten at every epoch to save disk space."""

        # Create directory for model checkpoints, if nonexistent
        if first:
            ckpt_dir_name = f'{datetime.now():{self.p.dataset_name}-%m%d-%H%M}'
            if self.p.ckpt_overwrite:
                ckpt_dir_name = self.p.dataset_name

            self.ckpt_dir = os.path.join(self.p.ckpt_save_path, ckpt_dir_name)
            if not os.path.isdir(self.p.ckpt_save_path):
                os.mkdir(self.p.ckpt_save_path)
            if not os.path.isdir(self.ckpt_dir):
                os.mkdir(self.ckpt_dir)

        # Save checkpoint dictionary
        if self.p.ckpt_overwrite:
            fname_unet = '{}/SMF-{}.pt'.format(self.ckpt_dir, self.p.dataset_name)
        else:
            print('[SMF 100] valid_loss:{}'.format(stats['valid_loss'][epoch]))
            valid_loss = stats['valid_loss'][epoch]
            fname_unet = '{}/SMF-epoch{}-{:>1.5f}.pt'.format(self.ckpt_dir, epoch + 1, valid_loss)
        # print('Saving checkpoint to: {}\n'.format(fname_unet))
        logger.info('Saving checkpoint to: {}\n'.format(fname_unet))

        torch.save(self.model.state_dict(), fname_unet)

        # Save stats to JSOaN
        fname_dict = '{}/SMF-stats.json'.format(self.ckpt_dir)
        with open(fname_dict, 'w') as fp:
            json.dump(stats, fp, indent=2)

    def load_model(self, ckpt_fname):
        """Loads model from checkpoint file."""

        # print('Loading checkpoint from: {}'.format(ckpt_fname))
        logger.info('Loading checkpoint from: {}'.format(ckpt_fname))
        if self.use_cuda:
            self.model.load_state_dict(torch.load(ckpt_fname))
        else:
            self.model.load_state_dict(torch.load(ckpt_fname, map_location='cpu'))

    def _on_epoch_end(self, stats, train_loss, epoch, epoch_start, valid_loader):
        """Tracks and saves starts after each epoch."""
        # import pdb;pdb.set_trace()
        # Evaluate model on validation set
        # print('\rTesting model on validation set... ', end='')
        logger.info('Testing model on validation set... ')
        epoch_time = time_elapsed_since(epoch_start)[0]
        valid_loss, valid_time, valid_psnr = self.eval(valid_loader)

        show_on_epoch_end(epoch_time, valid_time, valid_loss, valid_psnr)

        # Decrease learning rate if plateau
        self.scheduler.step(valid_loss)
        print('[SMF 135] valid_loss:{}'.format(valid_loss))
        # Save checkpoint
        stats['train_loss'].append(train_loss)
        stats['valid_loss'].append(valid_loss)
        stats['valid_psnr'].append(valid_psnr)
        self.save_model(epoch, stats, epoch == 0)

        writer.add_scalar(tag="Train Loss",  # 可以暂时理解为图像的名字
                          scalar_value=train_loss,  # 纵坐标的值
                          global_step=epoch + 1  # 当前是第几次迭代，可以理解为横坐标的值
                          )
        writer.add_scalar(tag="Valid Loss",  # 可以暂时理解为图像的名字
                          scalar_value=valid_loss,  # 纵坐标的值
                          global_step=epoch + 1  # 当前是第几次迭代，可以理解为横坐标的值
                          )
        writer.add_scalar(tag="Valid PSNR",
                          scalar_value=valid_psnr,
                          global_step=epoch + 1
                          )
        # Plot stats
        if self.p.plot_stats:
            plot_per_epoch(self.ckpt_dir, 'Valid loss', stats['valid_loss'], 'L1_Loss')
            plot_per_epoch(self.ckpt_dir, 'Valid PSNR', stats['valid_psnr'], 'PSNR (dB)')

    @torch.no_grad()
    def eval(self, valid_loader):
        """Evaluates denoiser on validation set."""

        # self.DCE_Net.load_state_dict(torch.load('/mnt/lustre/cyli/pyuser/CVPR2020_Zero-DCE/CVPR2020_lowlight/snapshots_pretrained/Epoch99.pth'))
        # self.DCE_Net.eval()
        self.model.train(False)

        valid_start = datetime.now()
        loss_meter = AvgMeter()
        psnr_meter = AvgMeter()

        for batch_idx, (source_list, target_list, data_lowlight, target_gt) in enumerate(valid_loader):

            if self.use_cuda:
                source_list = [data.clone().detach().cuda() for data in source_list]
                # print('source.shape(UHDFour_model):', source.shape)
                # print('target1.shape(UHDFour_model):', target1.shape)
                # source = source.reshape(850, 1200)
                # source_down = source_down.cuda()

                target_list = [data.clone().detach().cuda() for data in target_list]
                data_lowlight = data_lowlight.cuda()
                target_gt = target_gt.cuda()

            final_result = self.model(source_list, data_lowlight)
            # print('[SMF model 184] final_result.size[0]{} [1]{} [2]{}'.format(final_result[0].shape, final_result[1].shape, final_result[2].shape))
            # print('[SMF model 185 target.size [0]{} [1]{} [2]{}'.format(target_list[0].shape, target_list[1].shape, target_list[2].shape))

            # Update loss
            loss = self.L1(final_result[3], target_gt)
            loss_meter.update(loss.item())

            # Compute PSRN
            # for i in range(1):
                # import pdb;pdb.set_trace()
                # final_result = final_result.cpu()
                # target1 = target_list.cpu()
            psnr_meter.update(psnr(target_list[3].cpu(), final_result[3].cpu()).item())

        valid_loss = loss_meter.avg
        valid_time = time_elapsed_since(valid_start)[0]
        psnr_avg = psnr_meter.avg

        return valid_loss, valid_time, psnr_avg

    def train(self, train_loader, valid_loader):
        """Trains UHDNet on training set."""

        self.model.train(True)
        if self.p.ckpt_load_path is not None:
            self.model.load_state_dict(torch.load(self.p.ckpt_load_path), strict=False)
            # print('The pretrain model is loaded.')
            logger.info('The pretrain model is loaded...')
        self._print_params()

        num_batches = len(train_loader)
        assert num_batches % self.p.report_interval == 0, 'Report interval must divide total number of batches'

        # Dictionaries of tracked stats
        stats = {'train_loss': [],
                 'valid_loss': [],
                 'valid_psnr': []}

        # load VGG19 function
        # print("VGG19 pretrained model path(UHDFourmoduel):", os.path.abspath('.'))
        logger.info("VGG19 pretrained model path(UHDFourmoduel):{}".format(os.path.abspath('..')))
        # VGG = networks.VGG19(init_weights='./pre_trained_VGG19_model/vgg19.pth', feature_mode=True)
        VGG = networks.VGG19(init_weights='../pre_trained_VGG19_model/vgg19.pth', feature_mode=True)
        VGG.cuda()
        VGG.eval()
        # Main training loop
        train_start = datetime.now()
        counter = 0

        for epoch in range(self.p.nb_epochs):
            # print('EPOCH {:d} / {:d}'.format(epoch + 1, self.p.nb_epochs))
            logger.info('EPOCH {:d} / {:d}'.format(epoch + 1, self.p.nb_epochs))
            # print('[SMF_model 228] epoch')
            # print('[SMF_model 229] model ', self.model)
            # Some stats trackers
            epoch_start = datetime.now()
            train_loss_meter = AvgMeter()
            loss_meter = AvgMeter()
            time_meter = AvgMeter()
            loss_R_L = My_loss()
            # Minibatch SGD
            for batch_idx, (source_list, target_list, data_lowlight, target_gt) in enumerate(train_loader):
                # print('[UHDFour 244] source_list[0].shape:{}, source_list[1].shape:{}, source_list[2].shape:{}'.format(source_list[0].shape, source_list[1].shape, source_list[2].shape))
                # save_img3('/home/wanghuaiyao/UHDFour-pro1/UHDFour_code-main/src/results/process/t0', target_list[0])
                # save_img3('/home/wanghuaiyao/UHDFour-pro1/UHDFour_code-main/src/results/process/t1', target_list[1])
                # save_img3('/home/wanghuaiyao/UHDFour-pro1/UHDFour_code-main/src/results/process/t2', target_list[2])

                batch_start = datetime.now()
                progress_bar(batch_idx, num_batches, self.p.report_interval, loss_meter.val)
                # factor=torch.mean(target)/torch.mean(source)
                # import pdb;pdb.set_trace()
                if self.use_cuda:
                    source_list = [data.clone().detach().cuda() for data in source_list]
                    # print('[SMF_model 243] source_lisat.shape:{}'.format(source_list[0].shape))
                    target_list = [data.clone().detach().cuda() for data in target_list]
                    data_lowlight = data_lowlight.cuda()
                    target_gt = target_gt.cuda()
                # import pdb;pdb.set_trace()
                self.optim.zero_grad()
                # save_img1(source_list, '/home/wanghuaiyao/UHDFour-pro1/UHDFour_code-main/src/results/process/source')
                # torchvision.utils.save_image(source_list[0], './results/process/' + '/source_output1.jpg')
                # torchvision.utils.save_image(source_list[1], './results/process/' + '/source_output2.jpg')
                # torchvision.utils.save_image(source_list[2], './results/process/' + '/source_output3.jpg')\
                # print('[UHDF 294] data_lowlight.device()', data_lowlight.cuda().device)
                # print('[UHDF 295] source_list.device()', source_list[0].device)
                final_result = self.model(source_list, data_lowlight)
                # print('[UHDFour 284] final_result.shape[0]:{} [1]:{} [2]:{}'.format(final_result[0].shape, final_result[1].shape, final_result[2].shape))
                # save_img1(final_result, '/home/wanghuaiyao/UHDFour-pro1/UHDFour_code-main/src/results/process/')
                # save_img1(target_list, '/home/wanghuaiyao/UHDFour-pro1/UHDFour_code-main/src/results/process/gt')
                # print('[UHDF 291] current path:{}'.format(os.getcwd()))


                # torchvision.utils.save_image(final_result[0], './results/process/' + '/train_output1.jpg')
                # torchvision.utils.save_image(final_result[1],'./results/process/' + '/train_output2.jpg')
                # torchvision.utils.save_image(final_result[2],'./results/process/' + '/train_output3.jpg')
                # torchvision.utils.save_image(target_list[2],'./results/process/' + '/GT_example.jpg')
                # torchvision.utils.save_image(source_list[0],'./results/process/' + '/input1.jpg')
                # print('[UHDF_model 249] final_result[0].shape{},final_result[1].shape{},final_result[2].shape{}'.format(final_result[0].shape,final_result[1].shape,final_result[2].shape))
                # Loss function
                # if self.p.dataset_name == 'UHD':
                #     scale = 0.125
                # else:
                #     scale = 0.5
                print()
                # loss_l1 = 5 * F.smooth_l1_loss(final_result[2], target_list[2])  #1
                # loss_l1down = 0.5 * F.smooth_l1_loss(final_result_down,
                #                                      F.interpolate(target, scale_factor=scale, mode='bilinear'))
                # print('[UHDF 310] final_result[1].shape:', final_result[0].shape, final_result[1].shape, final_result[2].shape)
                # print('[UHDF 311] target_result[1].shape:', target_list[0].shape, target_list[1].shape,
                #       target_list[2].shape)
                # loss_l1_1 = 5 * F.smooth_l1_loss(final_result[1], target_list[1])  #2
                #这个color_loss之后要改一下
                # print('[UHDF 323] final_result[2].type:{}, target_gt.type:{}'.format(final_result[2].dtype, target_gt.dtype))
                # color_loss = torch.nn.MSELoss()(final_result[3], target_gt)   #3

                result_feature = VGG(final_result[3])
                target_feature = VGG(F.interpolate(target_list[3].float(), scale_factor=1.0, mode='bilinear'))
                loss_per = 0.001 * self.L2(result_feature, target_feature)
                # print('[UHDF_model 263] loss_per:{}'.format(loss_per))

                loss_rec = F.mse_loss(final_result[3], target_gt)
                # print('[UHDF_model 263] len(final_result):{}, len(target_list):{}'.format(len(final_result), len(target_list)))
                # loss_ssim = 0.002 * (1 - pytorch_ssim.ssim(final_result[2].float(), target_list[2].float()))  #4
                # loss_ssim = pytorch_ssim.ssim(final_result[0].float(), target_list[0].float())  #4
                # print('[UHDF_model 266] loss_l1:{}, loss_ssim:{}'.format(loss_l1, loss_ssim))
                # print('final_result:{}, target_list:{}'.format(len(final_result),len(target_list)))

                rec_loss, layer_loss, myloss = loss_R_L(final_result, target_list)

                # print('[UHDF_model 337] loss_l1:{}, loss_ssim:{}, loss_per:{}, loss_l1_1:{}, color_loss:{}'.format(loss_l1,loss_ssim,loss_per,loss_l1_1,color_loss))
                # print('[UHDF_model 338] myloss  rec_loss:{}, layer_loss:{}, myloss:{}'.format(rec_loss, layer_loss,myloss))
                # loss_final = loss_l1 + loss_ssim + loss_per + loss_l1_1 + color_loss + myloss  #5
                # loss_final = rec_loss + loss_per + loss_rec
                loss_final = myloss + loss_per + loss_rec
                # print('[UHDF_model 267] loss_final:{}'.format(loss_final))
                loss_meter.update(loss_final.item())

                loss = loss_final
                # Zero gradients, perform a backward pass, and update the weights
                # self.optim.zero_grad()
                loss.backward()
                # print('11111111')
                # torch.nn.utils.clip_grad_norm(self.model.parameters(), 0.1) ###########new added
                self.optim.step()

                # Report/update statistics
                time_meter.update(time_elapsed_since(batch_start)[1])
                if (batch_idx + 1) % self.p.report_interval == 0 and batch_idx:
                    show_on_report(batch_idx, num_batches, loss_meter.avg, time_meter.avg)
                    train_loss_meter.update(loss_meter.avg)
                    loss_meter.reset()
                    time_meter.reset()
                    # if batch_idx==10:
                #    break

                # print("total", ":", loss_final.item(),  "loss_l1", ":", loss_l1.item(), "loss_ssim", ":", loss_ssim.item())
                # logger.info("total:{}   ,loss_l1:{}   ,loss_ssim:{}".format(loss_final, loss_l1, loss_ssim))  #6
                logger.info("total:{}, loss_per:{}".format(loss_final, loss_per))
            if (epoch + 1) % 100 == 0:
                torchvision.utils.save_image(final_result[0], './results/process/' + '/train_output1.jpg')
                torchvision.utils.save_image(final_result[1], './results/process/' + '/train_output2.jpg')
                torchvision.utils.save_image(final_result[2], './results/process/' + '/train_output3.jpg')
                # torchvision.utils.save_image(target_list[2], './results/process/' + '/GT_example.jpg')
                torchvision.utils.save_image(target_list[1], './results/process/' + '/GT_example.jpg') #2层的时候输出
                torchvision.utils.save_image(source_list[0], './results/process/' + '/input1.jpg')
            self._on_epoch_end(stats, train_loss_meter.avg, epoch, epoch_start, valid_loader)
            train_loss_meter.reset()
            # import pdb
            # pdb.set_trace()
        train_elapsed = time_elapsed_since(train_start)[0]
        # print('Training done! Total elapsed time: {}\n'.format(train_elapsed))
        logger.info('Training done! Total elapsed time: {}\n'.format(train_elapsed))
