import torch
import numpy as np
import argparse
import subprocess
import sys
import os

from models import * #the path is depended on where you create this module
from utils.frame_utils import read_gen#the path is depended on where you create this module 

noiseLevel = '0' #sys.argv[2]
picnum = ["00001","01964","01969","01997","01998"]
idt = int(sys.argv[1])

imAddr1 = "../data/eva-dark/" + noiseLevel + "/" + picnum[idt] + "_img1.ppm"
imAddr2 = "../data/eva-dark/" + noiseLevel + "/" + picnum[idt] + "_img2.ppm"

print(imAddr1)
print(imAddr2)

outAddr = "../works/" + picnum[idt] + ".flo"
postProcess = 'python -m flowiz ../works/*.flo'


if __name__ == '__main__':
    #obtain the necessary args for construct the flownet framework
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
    # parser.add_argument("--rgb_max", type=float, default=255.)
    # args = parser.parse_args()

    args = argparse.Namespace(fp16=False, rgb_max=255.0)

    #initial a Net
    net = FlowNet2C(args).cuda()
    #load the state_dict
    dict = torch.load("../works/FlowNet2C_model_best.pth.tar")
    # dict = torch.load("../data/FlowNet2_checkpoint.pth.tar")
    # print(dict["state_dict"])
    net.load_state_dict(dict["state_dict"])
    
    #load the image pair, you can find this operation in dataset.py
    pim1 = read_gen(imAddr1)
    pim2 = read_gen(imAddr2)

    print(pim1.shape)

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

    subprocess.call(postProcess, shell=True)
