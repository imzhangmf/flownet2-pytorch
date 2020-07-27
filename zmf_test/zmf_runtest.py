# input 1: a folder containing subfolders containing img pairs end with -1, -2
# input 2: which model to use
# input 3: checkpoint addr
# output a .flo, a .ppm of flow next to input pairs with -epe ending rename

import torch
import numpy as np
import argparse
import subprocess
import sys
import os
import glob
from skimage import io

from models import * #the path is depended on where you create this module
from utils.frame_utils import read_gen #the path is depended on where you create this module

# SONY
# ad1 = '../../data/dataset/newOptFlow/sony/flownet2-fail/23/(7).ARW'
# ad2 = '../../data/dataset/newOptFlow/sony/flownet2-fail/23/(8).ARW'
# outad = './new2/7-8.flo'
# viewad1 = './new2/a.jpg'
# viewad2 = './new2/b.jpg'
# crop_h = 1920
# crop_w = 2944
# ratio = 50.0
# sample_every_n_pixels = 2

# CANON
# ad1 = '../../data/dataset/newOptFlow/canon/27/(1).JPG' # cr2 2010*3011*3
# ad2 = '../../data/dataset/newOptFlow/canon/21/(1).JPG' # jpg 4000*6000*3
# outad = './new2/27-21.flo'
# viewad1 = './new2/a.jpg'
# viewad2 = './new2/b.jpg'
# crop_h = 1920*2 # *2 for jpg
# crop_w = 2944*2
# ratio = 100.
# sample_every_n_pixels = 4 # 4 - jpg, 2 - cr2

# ad1 = '../../data/dataset/newOptFlow/canon/21/(10).CR2' # cr2 2010*3011*3
# ad2 = '../../data/dataset/newOptFlow/canon/22/(10).CR2' # jpg 4000*6000*3
# outad = './new2/21-22.flo'
# viewad1 = './new2/a.jpg'
# viewad2 = './new2/b.jpg'
# crop_h = 1920 # *2 for jpg
# crop_w = 2944
# ratio = 200.
# sample_every_n_pixels = 2 # 4 - jpg, 2 - cr2

# FUJIFILM
# ima = 1
# imb = 2
# ad1 = '../../data/dataset/newOptFlow/fuji/' + str(ima) + '/(1).JPG'
# ad2 = '../../data/dataset/newOptFlow/fuji/' + str(imb) + '/(1).JPG'
# outad = './new2/' + str(ima) + '-' + str(imb) + '.flo'
# viewad1 = './new2/' + str(ima) + '.jpg'
# viewad2 = './new2/' + str(imb) + '.jpg'
# crop_h = 1920*2 # *2 for jpg
# crop_w = 2944*2
# ratio = 6000.
# sample_every_n_pixels = 4 # 4 - jpg, 2 - cr2

# input_folder = sys.argv[1]
# model = sys.argv[2]
model = 'FlowNet2'
# checkpoint_addr = sys.argv[3]
checkpoint_addr = '../../data/pretrained_model/FlowNet2-pytorch/FlowNet2_checkpoint.pth.tar'

def crop(im,h,w):
    h0 = im.shape[0]
    w0 = im.shape[1]
    if (h0 < h or w0 < w):
        print("bad crop")
        return im
    newim = im[0:h:sample_every_n_pixels,0:w:sample_every_n_pixels,:]
    return newim

def calc_and_save_flow(imAddr1, imAddr2, outAddr, ratio = 1.0):
    args = argparse.Namespace(fp16=False, rgb_max=255.0)
    #initial a Net
    if model == 'FlowNet2':
        net = FlowNet2(args).cuda()
    elif model == 'FlowNet2C':
        net = FlowNet2C(args).cuda()
    elif model == 'FlowNet2CS':
        net = FlowNet2CS(args).cuda()
    elif model == 'FlowNet2CSS':
        net = FlowNet2CSS(args).cuda()
    elif model == 'FlowNet2S':
        net = FlowNet2S(args).cuda()
    elif model == 'FlowNet2SD':
        net = FlowNet2SD(args).cuda()
    else:
        print("no such model")
    #load the state_dict
    dict = torch.load(checkpoint_addr)
    net.load_state_dict(dict["state_dict"])
    #load the image pair, you can find this operation in dataset.py
    pim1 = read_gen(imAddr1,ratio)
    print('read: '+ str(pim1.shape))
    # pim1 = crop(pim1,crop_h,crop_w)
    pim2 = read_gen(imAddr2,ratio)
    # pim2 = crop(pim2,crop_h,crop_w)
    pim1=pim1[:,::2,:]
    pim2=pim2[:,::2,:]
    images = [pim1, pim2]
    print('after crop: ' + str(np.array(images).shape))
    images = np.array(images).transpose(3, 0, 1, 2)
    im = torch.from_numpy(images.astype(np.float32)).unsqueeze(0).cuda()
    #process the image pair to obtian the flow 
    result = net(im)[0].squeeze()  
    #save flow, I reference the code in scripts/run-flownet.py in flownet2-caffe project 
    def writeFlow(name, flow):
        f = open(name, 'wb')
        f.write('PIEH'.encode('utf-8'))
        np.array([flow.shape[1], flow.shape[0]], dtype=np.int32).tofile(f)
        flow = flow.astype(np.float32)
        flow.tofile(f)
        f.flush()
        f.close()
    data = result.data.cpu().numpy().transpose(1, 2, 0)
    writeFlow(outAddr,data)
    # io.imsave(viewad1,pim1)
    # io.imsave(viewad2,pim2)

def EPE(input_flow, target_flow):
    return torch.norm(target_flow-input_flow,p=2,dim=1).mean()

def readFlow(file):
    f = open(file,'rb')
    flo_number = np.fromfile(f, np.float32, count=1)
    assert flo_number[0] == 202021.25, 'Flow number %r incorrect. Invalid .flo file' % flo_number
    w = np.fromfile(f, np.int32, count=1)
    h = np.fromfile(f, np.int32, count=1)

    data = np.fromfile(f, np.float32, count=2*w[0]*h[0])
    # Reshape data into 3D array (columns, rows, bands)
    flow = np.resize(data, (int(h), int(w), 2))
    # print(np.shape(flow))
    f.close()

    return flow

# all_files = []
# all_files += glob.glob(input_folder+"/*.ppm")
# all_files += glob.glob(input_folder+"/*/*.ppm")
# all_files += glob.glob(input_folder+"/*/*/*.ppm")
# all_files = sorted(all_files, key=lambda name: name)
# print("Explored ppm files depth = 3")
# # print(all_files)

# imAddr1 = ''
# imAddr2 = ''
# tempAddr = ''
# for addr in all_files:
#     file_name = os.path.basename(addr)
#     father_addr_len = len(addr) - len(file_name)
#     father_addr = addr[0:father_addr_len]

#     if addr[-5] == '1':
#         imAddr1 = addr
    
#     if addr[-5] == '2':
#         imAddr2 = addr
#         print("Processing:")
#         print(imAddr1)
#         print(imAddr2)

#         outAddr = imAddr1[0:-9] + ".flo"
#         calc_and_save_flow(imAddr1, imAddr2, outAddr)
#         # print(outAddr)

#         postProcess = 'python -m flowiz ' + outAddr
#         subprocess.call(postProcess, shell=True)

#         out_flo = outAddr
#         gt_flo = input_folder + '/gt/' + os.path.basename(outAddr)[0:-4] + '_flow.flo' # 00001_flow.flo
#         flow1 = readFlow(out_flo)
#         flow2 = readFlow(gt_flo)
#         tensorFlow1 = torch.from_numpy(flow1)
#         tensorFlow2 = torch.from_numpy(flow2)
#         err = EPE(tensorFlow1,tensorFlow2).numpy()
#         subprocess.call('echo ' + str(err) + ' >> ' + father_addr + '/epe.txt', shell=True)

# import itertools

# listA = range(91,93+1)
# perm = itertools.permutations(listA,2)

# for ab in list(perm):
#     ima = ab[0]
#     imb = ab[1]

#     # ad1 = '../../data/dataset/newOptFlow/fuji/' + str(ima) + '/(2).JPG'
#     # ad2 = '../../data/dataset/newOptFlow/fuji/' + str(imb) + '/(2).JPG'
#     # outad = './new2/' + str(ima) + '-' + str(imb) + '.flo'
#     # viewad1 = './new2/' + str(ima) + '.jpg'
#     # viewad2 = './new2/' + str(imb) + '.jpg'
#     # crop_h = 1920*2 # *2 for jpg
#     # crop_w = 2944*2
#     # ratio = 6000.
#     # sample_every_n_pixels = 4 # 4 - jpg, 2 - cr2

#     ad1 = '../../data/dataset/newOptFlow/fuji2/' + str(ima) + '/(1).JPG'
#     ad2 = '../../data/dataset/newOptFlow/fuji2/' + str(imb) + '/(1).JPG'
#     outad = './new2/' + str(ima) + '-' + str(imb) + '.flo'
#     viewad1 = './new2/' + str(ima) + '.jpg'
#     viewad2 = './new2/' + str(imb) + '.jpg'
#     crop_h = 1920*2
#     crop_w = 2944*2
#     ratio = 50.0
#     sample_every_n_pixels = 4

#     # ad1 = '../../data/dataset/newOptFlow/canon/' + str(ima) + '/(1).JPG' # cr2 2010*3011*3
#     # ad2 = '../../data/dataset/newOptFlow/canon/' + str(imb) + '/(1).JPG' # jpg 4000*6000*3
#     # outad = './new2/' + str(ima) + '-' + str(imb) + '.flo'
#     # viewad1 = './new2/' + str(ima) + '.jpg'
#     # viewad2 = './new2/' + str(imb) + '.jpg'
#     # crop_h = 1920*2 # *2 for jpg
#     # crop_w = 2944*2
#     # ratio = 100.
#     # sample_every_n_pixels = 4 # 4 - jpg, 2 - cr2

#     calc_and_save_flow(ad1,ad2,outad, ratio)


allimg1 = glob.glob('../../../Desktop/myoptflow/YOMO-half/iphone_test/*_img1.jpg')
allimg2 = glob.glob('../../../Desktop/myoptflow/YOMO-half/iphone_test/*_img2.jpg')
allimg1.sort()
allimg2.sort()

for i in range(len(allimg1)):
    image_path1 = allimg1[i]
    image_path2 = allimg2[i]
    bname = os.path.basename(image_path1)
    bnum = bname[0:5]
    n = '../../../Desktop/myoptflow/model_test/flownet2/iphone/%s_flow.flo'%bnum
    calc_and_save_flow(image_path1,image_path2,n)