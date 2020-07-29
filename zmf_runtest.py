import torch
import numpy as np
import argparse
import subprocess
import sys
import os
import glob
from skimage import io
import zmf

from models import * #the path is depended on where you create this module
from utils.frame_utils import read_gen #the path is depended on where you create this module

model = 'FlowNet2'
checkpoint_addr = '../../data/pretrained_model/FlowNet2-pytorch/FlowNet2_checkpoint.pth.tar'

def calc_and_save_flow(imAddr1, imAddr2, outAddr):
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
    pim1 = read_gen(imAddr1)
    pim2 = read_gen(imAddr2)
    images = [pim1, pim2]
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

import itertools

list1 = range(11,14+1)
perm1 = itertools.permutations(list1,2)
list2 = range(21,24+1)
perm2 = itertools.permutations(list2,2)
list3 = range(31,34+1)
perm3 = itertools.permutations(list3,2)
list4 = range(41,44+1)
perm4 = itertools.permutations(list4,2)
list5 = range(51,54+1)
perm5 = itertools.permutations(list5,2)

perm = list(perm1)+list(perm2)+list(perm3)+list(perm4)+list(perm5)

# print(perm)


for ab in perm:
    ima = ab[0]
    imb = ab[1]

    ad1 = '../../../Desktop/Nikon/scaled_jpg/%d_1.jpg'%ima
    ad2 = '../../../Desktop/Nikon/scaled_jpg/%d_1.jpg'%imb
    outad = '../../../Desktop/Nikon/GT/%d-%d.flo'%(ima,imb)
    print(outad)
    calc_and_save_flow(ad1,ad2,outad)















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