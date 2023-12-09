# --- Imports --- #
import time
import torch
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader
# from val_data import ValData
from utils import validation_PSNR, generate_filelist, validation_NIQE
from dataloader import TestData, TestData_NR
# from thop import profile
import os

# from torchsummaryX import summary

# --- Parse hyper-parameters  --- #/home/wanghuaiyao/UHDFour-pro1/UHDFour_code-main/data/LOLv1/eval15/input
parser = argparse.ArgumentParser(description='PyTorch implementation of dehamer from Li et al. (2022)')
parser.add_argument('-d', '--dataset-name', help='name of dataset',
                    choices=['UHD', 'LOLv1', 'LOLv2', 'our_test', 'NewDataSet'], default='LOLv1')
parser.add_argument('-t', '--test-image-dir', help='test images path',
                    default='/home/wanghuaiyao/SMFPlus-pro/data/LOLv1/eval15/')
parser.add_argument('-c', '--ckpts-dir', help='ckpts path',
                    default='/home/wanghuaiyao/SMFPlus-pro/ckpts_training_ablation/LOLv1-1125-1849/SMF-epoch984-0.08119.pt')  # 4layer
parser.add_argument('-val_batch_size', help='Set the validation/test batch size', default=100, type=int)
args = parser.parse_args()

val_batch_size = args.val_batch_size
dataset_name = args.dataset_name
# import pdb;pdb.set_trace()
# --- Set dataset-specific hyper-parameters  --- #
print('curruct path(test_PSNR)', os.getcwd())

if dataset_name == 'UHD':
    val_data_dir = './data/UHD-LL/testing_set/'
#     ckpts_dir = './ckpts/UHD_checkpoint.pt'
elif dataset_name == 'DICM':
    # val_data_dir = '../data/LOLv1/eval15/'
    val_data_dir = '../data/DICM/'
    # ckpts_dir = '../ckpts_training/LOLv1-0527-2101/UHDFour-epoch399-0.07460.pt'
elif dataset_name == 'LOLv2':
    val_data_dir = './data/LOL-v2/Test/'
#     ckpts_dir = './ckpts/LOLv2_checkpoint.pt'
elif dataset_name == 'NewDataSet':
    val_data_dir = '../data/NewDataSet/eval/'
else:
    val_data_dir = args.test_image_dir
# ckpts_dir =  args.ckpts_dir
# ckpts_dir = '../ckpts_training/LOLv1-0730-0014/UHDFour-epoch300-0.07741.pt'
# ckpts_dir = '../ckpts_training/LOLv1-0913-2232/UHDFour-epoch99-0.31768.pt'
ckpts_dir = '/home/wanghuaiyao/UHDFour-pro1/UHDFour_code-main/ckpts_training_ablation/LOLv1-1015-2142/UHDFour-epoch407-0.07889.pt'

# prepare .txt file
if not os.path.exists(os.path.join(val_data_dir, 'data_list.txt')):
    print(os.path.join(val_data_dir, 'data_list.txt'))
    generate_filelist(val_data_dir, valid=True)
# --- Gpu device --- #
device_ids = [Id for Id in range(torch.cuda.device_count())]
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
# --- Validation data loader --- #
val_data_loader = DataLoader(TestData([400, 600], val_data_dir), batch_size=val_batch_size, shuffle=False)
# val_data_loader = DataLoader(TestData([384, 384], val_data_dir), batch_size=val_batch_size, shuffle=False)
# val_data_loader = DataLoader(TestData_NR([400, 400], val_data_dir), batch_size=val_batch_size, shuffle=False) #DICM
# val_data_loader = DataLoader(TestData_NR([300, 300], val_data_dir), batch_size=val_batch_size, shuffle=False) #MEF
# val_data_loader = DataLoader(TestData_NR([200, 200], val_data_dir), batch_size=val_batch_size, shuffle=False) #NPE

# --- Define the network --- #
if dataset_name == 'DICM' or dataset_name == 'LOLv2' or dataset_name == 'LOLv1':
    from SMF_arch import InteractNet as UHD_Net
else:
    from SMF_arch import InteractNet as UHD_Net
net = UHD_Net()

# --- Multi-GPU --- #
net = net.to(device)
# net = nn.DataParallel(net, device_ids=device_ids)
net.load_state_dict(torch.load(ckpts_dir), strict=False)
# net.load_state_dict(torch.load(ckpts_dir))


# --- Use the evaluation model in testing --- #
net.eval()
print('--- Testing starts! ---')
start_time = time.time()
val_psnr, val_ssim, val_dists = validation_PSNR(net, val_data_loader, device, dataset_name, save_tag=True)
end_time = time.time() - start_time
print('val_psnr: {0:.2f}, val_ssim: {1:.4f}, val_dists: {2:.4f}'.format(val_psnr, val_ssim, val_dists))
print('validation time is {0:.4f}'.format(end_time))
'''计算NIQE'''
# val_niqe, min_niqe = validation_NIQE(net, val_data_loader, device, dataset_name, save_tag=True)
# end_time = time.time() - start_time
# print('val_niqe: {0:.2f} , min_niqe:{1:.2f}'.format(val_niqe, min_niqe))
# print('validation time is {0:.4f}'.format(end_time))

