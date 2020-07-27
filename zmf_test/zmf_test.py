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

from models import * #the path is depended on where you create this module
from utils.frame_utils import read_gen#the path is depended on where you create this module


input_folder = sys.argv[1]
model = sys.argv[2]
checkpoint_addr = sys.argv[3]

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

all_files = []
all_files += glob.glob(input_folder+"/*.ppm")
all_files += glob.glob(input_folder+"/*/*.ppm")
all_files += glob.glob(input_folder+"/*/*/*.ppm")
all_files = sorted(all_files, key=lambda name: name)
print("Explored ppm files depth = 3")
# print(all_files)

imAddr1 = ''
imAddr2 = ''
tempAddr = ''
for addr in all_files:
    file_name = os.path.basename(addr)
    father_addr_len = len(addr) - len(file_name)
    father_addr = addr[0:father_addr_len]

    if addr[-5] == '1':
        imAddr1 = addr
    
    if addr[-5] == '2':
        imAddr2 = addr
        print("Processing:")
        print(imAddr1)
        print(imAddr2)

        outAddr = imAddr1[0:-9] + ".flo"
        calc_and_save_flow(imAddr1, imAddr2, outAddr)
        # print(outAddr)

        postProcess = 'python -m flowiz ' + outAddr
        subprocess.call(postProcess, shell=True)

        out_flo = outAddr
        gt_flo = input_folder + '/gt/' + os.path.basename(outAddr)[0:-4] + '_flow.flo' # 00001_flow.flo
        flow1 = readFlow(out_flo)
        flow2 = readFlow(gt_flo)
        tensorFlow1 = torch.from_numpy(flow1)
        tensorFlow2 = torch.from_numpy(flow2)
        err = EPE(tensorFlow1,tensorFlow2).numpy()
        subprocess.call('echo ' + str(err) + ' >> ' + father_addr + '/epe.txt', shell=True)
