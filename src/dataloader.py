# --- Imports --- #
import torch.utils.data as data
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

import torchvision.transforms as tfs
from torchvision.transforms import functional as FF
import imghdr
import random
import torch
import numpy as np
from basicsr.utils import DiffJPEG, USMSharp
from skimage import io, color
# import PIL
import torchvision
import os
import glob
from lpls_decomposition import lplas_decomposition as decomposition
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
import cv2
import albumentations as A

trans = A.Compose([
		A.OneOf([
			A.HorizontalFlip(p=0.5),
			A.VerticalFlip(p=0.5)
		], p=0.5)
        # A.GaussNoise(p=0.2),    # 将高斯噪声应用于输入图像。
        # A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=10, p=0.3),
        # 随机应用仿射变换：平移，缩放和旋转输入
        # A.RandomBrightnessContrast(p=0.5),   # 随机明亮对比度
    ])

def populate_train_list(lowlight_images_path, nomal_images_path):  # 获取训练列表
    image_list_lowlight = glob.glob(lowlight_images_path + '/*')
    image_list_nomal = glob.glob(nomal_images_path + '/*')
    # image_list_nomal = 5 * image_list_nomal
    image_list_lowlight.sort()
    image_list_nomal.sort()
    print('[dataloader 29] len(image_list_lowlight):{}, len(image_list_nomal):{}'.format(len(image_list_lowlight), len(image_list_nomal)))
    if len(image_list_lowlight) != len(image_list_nomal):
        print('Data length Error')
        # logger.info('Data length Error')
        exit()
    # print('image_list_lowlight/n', image_list_lowlight)
    # print('image_list_nomal/n', image_list_nomal)

    return image_list_lowlight, image_list_nomal

def populate_train_list_NR(lowlight_images_path):  # 获取训练列表
    image_list_lowlight = glob.glob(lowlight_images_path + '/*')
    # image_list_nomal = 5 * image_list_nomal
    image_list_lowlight.sort()
    print('[dataloader 29] len(image_list_lowlight):{}'.format(len(image_list_lowlight)))
    # print('image_list_lowlight/n', image_list_lowlight)
    # print('image_list_nomal/n', image_list_nomal)

    return image_list_lowlight

def get_img_name(train_list_lowlight, train_list_gt):
    low_img_name = [low_list.split("/")[-1] for low_list in train_list_lowlight]
    gt_img_name = [normal_list.split("/")[-1] for normal_list in train_list_gt]
    return low_img_name, gt_img_name
def get_img_name_NR(train_list_lowlight):
    low_img_name = [low_list.split("/")[-1] for low_list in train_list_lowlight]
    return low_img_name

# --- Training dataset --- #
class TrainData(data.Dataset):
    def __init__(self, crop_size, train_data_dir, level_num=4):
        super().__init__()
        self.usm_sharpener = USMSharp().cuda()  # do usm sharpening
        torch.multiprocessing.set_start_method('spawn', force=True)
        # print("now absolute path(train_data_aug_local", os.path.abspath('.'))
        # train_list = train_data_dir + 'data_list.txt'  # 'datalist.txt''train_list_recap.txt' 'fitered_trainingdata.txt'
        self.train_list_lowlight, self.train_list_gt = populate_train_list(train_data_dir+'input', train_data_dir+'gt')
        # print('[dataloader 51] traindata populatelist has done')
        self.lowlight_names, self.gt_names = get_img_name(self.train_list_lowlight, self.train_list_gt)
        self.crop_size = crop_size
        self.size_w = crop_size[0]
        self.size_h = crop_size[1]
        self.level_num = level_num
        self.train_data_dir = train_data_dir
        self.trans = trans
        # print('[dataloader]: dataloader  init  have done!')

    def get_images(self, index):
        lowlight_name = self.lowlight_names[index]
        gt_name = self.gt_names[index]
        # print('[dataloader 91] lowlightname:{}', lowlight_name)
        # print('[dataloader 92] gtname:{}', gt_name)
        # print('[dataloader 61] lowlight_names', self.lowlight_names[index])
        lowlight_o = Image.open(self.train_data_dir + 'input/' + lowlight_name)
        # lowlight_o.save("/home/wanghuaiyao/UHDFour-pro1/UHDFour_code-main/src/results/process/dataloader_lowlight_o.png") #没问题

        lowlight = lowlight_o
        # lowlight = lowlight_o.convert('RGB')  # 'input_unprocess_aligned/' #1
        # print('[dataloader113] lowlight.size: {}, type(lowlight):{}'.format(lowlight.size, type(lowlight)))
        # print('[dataloader65] lowlight:', lowlight)
        # lowlight = np.array(lowlight, dtype=np.uint8) #2
        # print('[dataloader67] lowlight.shape: {}, type(lowlight):{}'.format(lowlight.shape, type(lowlight)))
        # print('[dataloader68] lowlight:', lowlight)
        # lowlight = Image.fromarray(lowlight, mode='RGB') #3
        # lowlight.save("/home/wanghuaiyao/UHDFour-pro1/UHDFour_code-main/src/results/process/dataloader_lowlight.png") #4
        # print('[dataloader69] lowlight.size: {}, type(lowlight):{}'.format(lowlight.size, type(lowlight)))
        gt_o = Image.open(self.train_data_dir + 'gt/' + gt_name)

        gt = gt_o
        # print('[dataloader 124] gt.size: {}, type(lowlight):{}'.format(gt.size, type(lowlight)))
        # gt = gt_o.convert('RGB')  # 'gt_unprocess_aligned/''high/' #5
        # gt = np.array(gt, dtype=np.uint8) # 6
        # gt = Image.fromarray(gt, mode='RGB') #7
        # gt.save("/home/wanghuaiyao/UHDFour-pro1/UHDFour_code-main/src/results/process/dataloader_gt.png") #8
        # print('[dataloader 128] gt.shape', gt.size )
        if not isinstance(self.crop_size, str):
            i, j, h, w = tfs.RandomCrop.get_params(lowlight, output_size=(self.size_w, self.size_h))
            # i,j,h,w=tfs.RandomCrop.get_params(lowlight,output_size=(2160,3840))
            lowlight = FF.crop(lowlight, i, j, h, w)
            gt = FF.crop(gt, i, j, h, w)
        data, target = lowlight, gt
        # data, target = self.augData(lowlight.convert("RGB"), gt.convert("RGB"))
        # print('[dataloader 83] data', data)
        # print('[dataloader 84] type(data) {},  type(target)  {}'.format(type(data), type(target)))
        data, target = np.array(data) / 255, np.array(target) / 255
        # print('[dataloader 86] type(data) {},  type(target)  {}'.format(type(data), type(target)))
        # print('[dataloader 87] shape(data) {},  shape(target)  {}'.format(data.shape, target.shape))
        # print('[dataloader 88] data', data)
        # save_img3('/home/wanghuaiyao/UHDFour-pro1/UHDFour_code-main/src/results/process/Dataloader_data', data)
        # augment = self.trans(image=data, mask=target)
        # data, target = augment['image'], augment['mask']
        # print('[dataloader 145] shape(data) {},  shape(target)  {}'.format(data.shape, target.shape))
        lowlight_G_list, lowlight_L_list = decomposition(data, self.level_num)
        # save_img3('/home/wanghuaiyao/UHDFour-pro1/UHDFour_code-main/src/results/process/Dataloader_lowlight_L_list[0]',
        #           lowlight_L_list[0])
        # save_img3('/home/wanghuaiyao/UHDFour-pro1/UHDFour_code-main/src/results/process/Dataloader_lowlight_L_list[1]',
        #           lowlight_L_list[1])
        # save_img3('/home/wanghuaiyao/UHDFour-pro1/UHDFour_code-main/src/results/process/Dataloader_lowlight_L_list[2]',
        #           lowlight_L_list[2])
        # print('[dataloader 114] lowlight_L_list.dtype:{}'.format(lowlight_L_list[0].astype(np.float32).dtype))

        normal_G_list, _ = decomposition(target, self.level_num)
        # print('[dataloader 89] get image has down! ')
        # lowlight_L_list = [data.astype(np.float32) * 255 for data in lowlight_L_list]
        # print('[dataloader 158] lowlight_L_list.shape{}'.format(lowlight_L_list[0].shape))
        lowlight_L_list = [torch.from_numpy(np.transpose((data.astype(np.float32)), (2,0,1))) for data in lowlight_L_list]
        # print('[dataloader 141] lowlight_L_list.shape{}'.format(lowlight_L_list[0].shape))
        # print('[dataloader 146] current path:{} '.format(os.getcwd()))
        # torchvision.utils.save_image(lowlight_L_list[0], './results/' + '/lowlight_output1.jpg')
        # torchvision.utils.save_image(lowlight_L_list[1], './results/' + '/lowlight_output2.jpg')
        # torchvision.utils.save_image(lowlight_L_list[-1], './results/' + '/lowlight_output3.jpg')

        # cv2.imwrite('/home/wanghuaiyao/UHDFour-pro1/UHDFour_code-main/src/results/process/Dataloader_lowlight_L_list[0].png', lowlight_L_list[0])
        # cv2.imwrite(
        #     '/home/wanghuaiyao/UHDFour-pro1/UHDFour_code-main/src/results/process/Dataloader_lowlight_L_list[1].png',
        #     lowlight_L_list[1])
        # cv2.imwrite(
        #     '/home/wanghuaiyao/UHDFour-pro1/UHDFour_code-main/src/results/process/Dataloader_lowlight_L_list[2].png',
        #     lowlight_L_list[2])
        # save_img31('/home/wanghuaiyao/UHDFour-pro1/UHDFour_code-main/src/results/process/Dataloader_lowlight_L_list[0]',
        #           lowlight_L_list[0])
        # save_img31('/home/wanghuaiyao/UHDFour-pro1/UHDFour_code-main/src/results/process/Dataloader_lowlight_L_list[1]',
        #           lowlight_L_list[1])
        # save_img31('/home/wanghuaiyao/UHDFour-pro1/UHDFour_code-main/src/results/process/Dataloader_lowlight_L_list[2]',
        #           lowlight_L_list[2])
        # normal_G_list = [data.astype(np.float32) * 255 for data in normal_G_list]
        normal_G_list = [torch.from_numpy(np.transpose((data.astype(np.float32)), (2,0,1))) for data in normal_G_list]
        # torchvision.utils.save_image(normal_G_list[0], './results/' + '/normal_output1.jpg')
        # torchvision.utils.save_image(normal_G_list[1], './results/' + '/normal_output2.jpg')
        # torchvision.utils.save_image(normal_G_list[-1], './results/' + '/normal_output3.jpg')
        # save_img31('/home/wanghuaiyao/UHDFour-pro1/UHDFour_code-main/src/results/process/Dataloader_gt_G_list[0]',
        #            normal_G_list[0])
        # save_img31('/home/wanghuaiyao/UHDFour-pro1/UHDFour_code-main/src/results/process/Dataloader_gt_G_list[1]',
        #            normal_G_list[1])
        # save_img31('/home/wanghuaiyao/UHDFour-pro1/UHDFour_code-main/src/results/process/Dataloader_gt_G_list[2]',
        #            normal_G_list[2]
        # print('[dataloader 190] lowlight_L_list.shape:{},{},{}:'.format(lowlight_L_list[0].shape,lowlight_L_list[1].shape,lowlight_L_list[2].shape))
        # print('[dataloader 191] normal_G_list.shape:{},{},{}:'.format(normal_G_list[0].shape, normal_G_list[1].shape,
        #                                                                 normal_G_list[2].shape))
        # print('[dataloader 193] data.shape:{}, target.shape:{}'.format(data.shape, target.shape))
        data_lowlight = torch.from_numpy(data.astype(np.float32).transpose(2, 0, 1))
        target_gt = torch.from_numpy(target.astype(np.float32).transpose(2, 0, 1))
        # print('[dataloader 196] data.shape:{}, target.shape:{}'.format(data_lowlight.shape, target_gt.shape))
        return lowlight_L_list, normal_G_list, data_lowlight, target_gt # , lowlight.resize((width/8, height/8)),gt.resize((width/8, height/8))#,factor

    def augData(self, data, target):
        # if self.train:
        if 1:
            rand_hor = random.randint(0, 1)  # [0,1]之间的随机整数
            rand_rot = random.randint(0, 3)  # [0，3]之间的随机整数
            data = tfs.RandomHorizontalFlip(rand_hor)(data)  # 水平翻转（第一个括号里是随机翻转的概率）（第二个是翻转对象）
            target = tfs.RandomHorizontalFlip(rand_hor)(target)
            if rand_rot:
                data = FF.rotate(data, 90 * rand_rot)
                target = FF.rotate(target, 90 * rand_rot)

        data = tfs.ToTensor()(data)
        target = tfs.ToTensor()(target)

        return data, target

    ''' 函数应该返回一个包含样本的数据和标签的元组或字典。
        这些数据可以是张量或其他可用于训练和评估模型的数据类型。
    '''

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.lowlight_names)

    def cutblur(self, im1, im2, prob=1.0, alpha=1.0):  # 超分图像处理中的一种数据增强手段，将高/低分辨率的图像中的一分部进行替换
        if im1.size() != im2.size():
            raise ValueError("im1 and im2 have to be the same resolution.")

        if alpha <= 0 or np.random.rand(1) >= prob:
            return im1, im2

        cut_ratio = np.random.randn() * 0.1 + alpha

        h, w = im2.size(0), im2.size(1)
        ch, cw = np.int(h * cut_ratio), np.int(w * cut_ratio)
        cy = np.random.randint(0, h - ch + 1)
        cx = np.random.randint(0, w - cw + 1)

        # apply CutBlur to inside or outside
        if np.random.random() > 0.3:  # 0.5
            im2[cy:cy + ch, cx:cx + cw, :] = im1[cy:cy + ch, cx:cx + cw, :]

        return im1, im2

    def tensor_to_image(self, tensor):
        tensor = tensor * 255
        tensor = np.array(tensor, dtype=np.uint8)
        if np.ndim(tensor) > 3:
            assert tensor.shape[0] == 1
            tensor = tensor[0]
        return Image.fromarray(tensor)

# --- Valid dataset --- #
class ValidData(data.Dataset):

    def __init__(self, crop_size, val_data_dir, level_num=4):
        super().__init__()
        self.val_list_lowlight, self.val_list_gt = populate_train_list(val_data_dir + 'input', val_data_dir + 'gt')
        # print('[dataloader 162] Valid data populatelist has done')
        self.lowlight_names, self.gt_names = get_img_name(self.val_list_lowlight, self.val_list_gt)
        self.val_data_dir = val_data_dir
        self.crop_size = crop_size
        self.size_w = crop_size[0]
        self.size_h = crop_size[1]
        self.level_num = level_num
        # print('[dataloader]: dataloader valid init have done!')

    def get_images(self, index):
        lowlight_name = self.lowlight_names[index]
        gt_name = self.gt_names[index]
        # print('[dataloader 61] lowlight_names', self.lowlight_names[index])
        lowlight_o = Image.open(self.val_data_dir + 'input/' + lowlight_name)
        lowlight = lowlight_o
        # lowlight = lowlight_o.convert('RGB')  # 'input_unprocess_aligned/'
        # print('[dataloader64] lowlight.size: {}, type(lowlight):{}'.format(lowlight.size, type(lowlight)))
        # print('[dataloader65] lowlight:', lowlight)
        # lowlight = np.array(lowlight, dtype=np.uint8) / 255.0
        # print('[dataloader67] lowlight.shape: {}, type(lowlight):{}'.format(lowlight.shape, type(lowlight)))
        # print('[dataloader68] lowlight:', lowlight)
        # lowlight = Image.fromarray(lowlight, mode='RGB')
        # print('[dataloader69] lowlight.size: {}, type(lowlight):{}'.format(lowlight.size, type(lowlight)))
        gt_o = Image.open(self.val_data_dir + 'gt/' + gt_name)
        # gt = gt_o.convert('RGB')  # 'gt_unprocess_aligned/''high/'
        # gt = np.array(gt, dtype=np.uint8) / 255.0
        # gt = Image.fromarray(gt, mode='RGB')
        # print('[dataloader 75] data', lowlight )
        gt = gt_o
        if not isinstance(self.crop_size, str):
            i, j, h, w = tfs.RandomCrop.get_params(lowlight, output_size=(self.size_w, self.size_h))
            # i,j,h,w=tfs.RandomCrop.get_params(lowlight,output_size=(2160,3840))
            lowlight = FF.crop(lowlight, i, j, h, w)
            gt = FF.crop(gt, i, j, h, w)
        data, target = lowlight, gt
        # data, target = self.augData(lowlight.convert("RGB"), gt.convert("RGB"))
        # print('[dataloader 83] data', data)
        # print('[dataloader 84] type(data) {},  type(target)  {}'.format(type(data), type(target)))
        data, target = np.array(data) / 255, np.array(target) / 255
        # print('[dataloader 86] type(data) {},  type(target)  {}'.format(type(data), type(target)))
        # print('[dataloader 87] shape(data) {},  shape(target)  {}'.format(data.shape, target.shape))
        # print('[dataloader 88] data', data)
        lowlight_G_list, lowlight_L_list = decomposition(data, self.level_num)
        normal_G_list, _ = decomposition(target, self.level_num)
        # print('[dataloader 89] get image has down! ')
        lowlight_L_list = [torch.from_numpy(np.transpose((data.astype(np.float32)),(2,0,1))) for data in lowlight_L_list]
        # torchvision.utils.save_image(lowlight_L_list[0], './results/' + '/val_lowlight_output1.jpg')
        # torchvision.utils.save_image(lowlight_L_list[1], './results/' + '/val_lowlight_output2.jpg')
        # torchvision.utils.save_image(lowlight_L_list[-1], './results/' + '/val_lowlight_output3.jpg')
        # lowlight_G_list = [np.transpose(data, (2, 1, 0)) for data in lowlight_G_list]
        normal_G_list = [torch.from_numpy(np.transpose((data.astype(np.float32)),(2,0,1))) for data in normal_G_list]
        # print('[dataloader 95] lowlight_G_list.shape:{}'.format(lowlight_G_list[0].shape))
        # torchvision.utils.save_image(normal_G_list[0], './results/' + '/val_normal_output1.jpg')
        # torchvision.utils.save_image(normal_G_list[1], './results/' + '/val_normal_output2.jpg')
        # torchvision.utils.save_image(normal_G_list[-1], './results/' + '/val_normal_output3.jpg')
        data_lowlight = torch.from_numpy(data.astype(np.float32).transpose(2, 0, 1))
        target_gt = torch.from_numpy(target.astype(np.float32).transpose(2, 0, 1))
        return lowlight_L_list, normal_G_list, data_lowlight, target_gt
    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.lowlight_names)

class TestData(data.Dataset):
    def __init__(self, crop_size, val_data_dir, level_num=4):
        super().__init__()
        self.val_list_lowlight, self.val_list_gt = populate_train_list(val_data_dir + 'input', val_data_dir + 'gt')
        # print('[dataloader 162] Valid data populatelist has done')
        self.lowlight_names, self.gt_names = get_img_name(self.val_list_lowlight, self.val_list_gt)
        self.val_data_dir = val_data_dir
        self.crop_size = crop_size
        self.size_w = crop_size[0]
        self.size_h = crop_size[1]
        self.level_num = level_num
        # print('[dataloader]: dataloader valid init have done!')

    def get_images(self, index):
        lowlight_name = self.lowlight_names[index]
        gt_name = self.gt_names[index]
        # print('[dataloader 61] lowlight_names', self.lowlight_names[index])
        lowlight_o = Image.open(self.val_data_dir + 'input/' + lowlight_name)
        lowlight = lowlight_o
        # lowlight = lowlight_o.convert('RGB')  # 'input_unprocess_aligned/'
        # print('[dataloader64] lowlight.size: {}, type(lowlight):{}'.format(lowlight.size, type(lowlight)))
        # print('[dataloader65] lowlight:', lowlight)
        # lowlight = np.array(lowlight, dtype=np.uint8) / 255.0
        # print('[dataloader67] lowlight.shape: {}, type(lowlight):{}'.format(lowlight.shape, type(lowlight)))
        # print('[dataloader68] lowlight:', lowlight)
        # lowlight = Image.fromarray(lowlight, mode='RGB')
        # print('[dataloader69] lowlight.size: {}, type(lowlight):{}'.format(lowlight.size, type(lowlight)))
        gt_o = Image.open(self.val_data_dir + 'gt/' + gt_name)
        # gt = gt_o.convert('RGB')  # 'gt_unprocess_aligned/''high/'
        # gt = np.array(gt, dtype=np.uint8) / 255.0
        # gt = Image.fromarray(gt, mode='RGB')
        # print('[dataloader 75] data', lowlight )
        gt = gt_o
        if not isinstance(self.crop_size, str):
            i, j, h, w = tfs.RandomCrop.get_params(lowlight, output_size=(self.size_w, self.size_h))
            # i,j,h,w=tfs.RandomCrop.get_params(lowlight,output_size=(2160,3840))
            lowlight = FF.crop(lowlight, i, j, h, w)
            gt = FF.crop(gt, i, j, h, w)
        data, target = lowlight, gt
        # data, target = self.augData(lowlight.convert("RGB"), gt.convert("RGB"))
        # print('[dataloader 83] data', data)
        # print('[dataloader 84] type(data) {},  type(target)  {}'.format(type(data), type(target)))
        data, target = np.array(data) / 255, np.array(target) / 255
        # print('[dataloader 86] type(data) {},  type(target)  {}'.format(type(data), type(target)))
        # print('[dataloader 87] shape(data) {},  shape(target)  {}'.format(data.shape, target.shape))
        # print('[dataloader 88] data', data)
        lowlight_G_list, lowlight_L_list = decomposition(data, self.level_num)
        normal_G_list, _ = decomposition(target, self.level_num)
        # print('[dataloader 89] get image has down! ')
        lowlight_L_list = [torch.from_numpy(np.transpose((data.astype(np.float32)),(2,0,1))) for data in lowlight_L_list]
        # torchvision.utils.save_image(lowlight_L_list[0], './results/' + '/val_lowlight_output1.jpg')
        # torchvision.utils.save_image(lowlight_L_list[1], './results/' + '/val_lowlight_output2.jpg')
        # torchvision.utils.save_image(lowlight_L_list[-1], './results/' + '/val_lowlight_output3.jpg')
        # lowlight_G_list = [np.transpose(data, (2, 1, 0)) for data in lowlight_G_list]
        normal_G_list = [torch.from_numpy(np.transpose((data.astype(np.float32)),(2,0,1))) for data in normal_G_list]
        # print('[dataloader 95] lowlight_G_list.shape:{}'.format(lowlight_G_list[0].shape))
        # torchvision.utils.save_image(normal_G_list[0], './results/' + '/val_normal_output1.jpg')
        # torchvision.utils.save_image(normal_G_list[1], './results/' + '/val_normal_output2.jpg')
        # torchvision.utils.save_image(normal_G_list[-1], './results/' + '/val_normal_output3.jpg')
        data_lowlight = torch.from_numpy(data.astype(np.float32).transpose(2, 0, 1))
        target_gt = torch.from_numpy(target.astype(np.float32).transpose(2, 0, 1))
        return lowlight_L_list, normal_G_list, data_lowlight, target_gt, self.lowlight_names
    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.lowlight_names)

class TestData_NR(data.Dataset):
    def __init__(self, crop_size, val_data_dir, level_num=4):
        super().__init__()
        self.val_list_lowlight = populate_train_list_NR(val_data_dir + 'input')
        # print('[dataloader 162] Valid data populatelist has done')
        self.lowlight_names = get_img_name_NR(self.val_list_lowlight)
        self.val_data_dir = val_data_dir
        self.crop_size = crop_size
        self.size_w = crop_size[0]
        self.size_h = crop_size[1]
        self.level_num = level_num
        # print('[dataloader]: dataloader valid init have done!')

    def get_images(self, index):
        lowlight_name = self.lowlight_names[index]
        # print('[dataloader 61] lowlight_names', self.lowlight_names[index])
        lowlight_o = Image.open(self.val_data_dir + 'input/' + lowlight_name)
        lowlight = lowlight_o
        # lowlight = lowlight_o.convert('RGB')  # 'input_unprocess_aligned/'
        # print('[dataloader64] lowlight.size: {}, type(lowlight):{}'.format(lowlight.size, type(lowlight)))
        # print('[dataloader65] lowlight:', lowlight)
        # lowlight = np.array(lowlight, dtype=np.uint8) / 255.0
        # print('[dataloader67] lowlight.shape: {}, type(lowlight):{}'.format(lowlight.shape, type(lowlight)))
        # print('[dataloader68] lowlight:', lowlight)
        # lowlight = Image.fromarray(lowlight, mode='RGB')
        # print('[dataloader69] lowlight.size: {}, type(lowlight):{}'.format(lowlight.size, type(lowlight)))
        # gt = gt_o.convert('RGB')  # 'gt_unprocess_aligned/''high/'
        # gt = np.array(gt, dtype=np.uint8) / 255.0
        # gt = Image.fromarray(gt, mode='RGB')
        # print('[dataloader 75] data', lowlight )
        if not isinstance(self.crop_size, str):
            i, j, h, w = tfs.RandomCrop.get_params(lowlight, output_size=(self.size_w, self.size_h))
            # i,j,h,w=tfs.RandomCrop.get_params(lowlight,output_size=(2160,3840))
            lowlight = FF.crop(lowlight, i, j, h, w)
        data = lowlight
        # data, target = self.augData(lowlight.convert("RGB"), gt.convert("RGB"))
        # print('[dataloader 83] data', data)
        # print('[dataloader 84] type(data) {},  type(target)  {}'.format(type(data), type(target)))
        data = np.array(data) / 255
        # print('[dataloader 86] type(data) {},  type(target)  {}'.format(type(data), type(target)))
        # print('[dataloader 87] shape(data) {},  shape(target)  {}'.format(data.shape, target.shape))
        # print('[dataloader 88] data', data)
        lowlight_G_list, lowlight_L_list = decomposition(data, self.level_num)
        # print('[dataloader 89] get image has down! ')
        lowlight_L_list = [torch.from_numpy(np.transpose((data.astype(np.float32)),(2,0,1))) for data in lowlight_L_list]
        # torchvision.utils.save_image(lowlight_L_list[0], './results/' + '/val_lowlight_output1.jpg')
        # torchvision.utils.save_image(lowlight_L_list[1], './results/' + '/val_lowlight_output2.jpg')
        # torchvision.utils.save_image(lowlight_L_list[-1], './results/' + '/val_lowlight_output3.jpg')
        # lowlight_G_list = [np.transpose(data, (2, 1, 0)) for data in lowlight_G_list]

        data_lowlight = torch.from_numpy(data.astype(np.float32).transpose(2, 0, 1))
        return lowlight_L_list, data_lowlight, self.lowlight_names
    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.lowlight_names)