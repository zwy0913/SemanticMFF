import os
import sys
import cv2
import glob
import time
import numpy as np
from tqdm import tqdm
from thop import profile, clever_format
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
from torchvision.utils import save_image
import torchvision.transforms.v2.functional as tv2f
from torchvision import transforms

from nets.network import SegFormer
from nets.s2net import SpatialNet
from nets.s2net import SemanticNet
from nets.generator import Generator

from misc.dataloader_gen import get_testloader

from misc.general import DeviceChecker
from misc.general import label2image


DEVICE = DeviceChecker.DEVICE


def img2tensor(img_path):
    img = Image.open(img_path).convert('RGB')
    tt = transforms.ToTensor()
    return tt(img).to(DEVICE).unsqueeze(0)


if __name__ == '__main__':
    # Configs
    model_spa_path = 'trained_weights/2025-04-12 00.32.58/debug_model_spa.ckpt'
    model_sem_path = 'trained_weights/2025-04-12 00.32.58/debug_model_sem.ckpt'
    model_gen_path = 'trained_weights/2025-04-21 08.17.14/debug_model_gen.ckpt'
    dataset_name = '50'
    dataset_path = os.path.join('L:/Datasets/Eval/', dataset_name)
    save_path = os.path.join('./results/', dataset_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # nets
    spa_net = SpatialNet(pretrained=False).to(DEVICE)
    spa_net.load_state_dict(torch.load(model_spa_path))
    spa_net.eval()
    sem_net = SemanticNet(pretrained=False).to(DEVICE)
    sem_net.load_state_dict(torch.load(model_sem_path))
    sem_net.eval()
    gen_net = Generator().to(DEVICE)
    gen_net.load_state_dict(torch.load(model_gen_path))
    gen_net.eval()
    test_list = sorted(glob.glob(os.path.join('./50', '*.*')))
    test_loader_tqdm = tqdm(test_list, colour='#F59427', leave=True, file=sys.stdout)

    cnt = 0
    start_time = time.time()

    with torch.no_grad():
        for img_path in test_loader_tqdm:
            if cnt == 0:
                imgA = img2tensor(img_path)
                cnt += 1
                continue
            imgB = img2tensor(img_path)
            inputs = torch.cat((imgA, imgB), dim=1).to(DEVICE)
            spatial, spa_feats = spa_net(inputs)
            semantic, sem_feats = sem_net(inputs)
            decision, _ = gen_net(imgA, imgB, spa_feats, sem_feats)


            decision = torch.where(decision > 0.5, 1., 0.)
            imgA = imgA * decision + imgB * (1 - decision)


            decision = torch.einsum('c w h -> w h c', decision[0]).clone().detach().cpu().numpy()
            fused = torch.einsum('c w h -> w h c', imgA[0]).clone().detach().cpu().numpy()
            cv2.imwrite(save_path + '/' + dataset_name + '-' + str(cnt).zfill(2) + '-dm.png', np.clip(decision * 255, 0, 255).astype(np.uint8))
            cv2.imwrite(save_path + '/' + dataset_name + '-' + str(cnt).zfill(2) + '-SemanticMFF.png', cv2.cvtColor(np.clip(fused * 255, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

            cnt += 1

            # originalA = cv2.imread(test_list[cnt])
            # originalB = cv2.imread(test_list[cnt])
            # decision = torch.where(decision > 0.5, 1., 0.)
            # decision = torch.einsum('c w h -> w h c', decision[0]).clone().detach().cpu().numpy()
            # fused = originalA * decision + originalB * (1 - decision)
            # cv2.imwrite(save_path + '/' + dataset_name + '-' + str(cnt).zfill(2) + '-dm.png',
            #             np.clip(decision * 255, 0, 255).astype(np.uint8))
            # cv2.imwrite(save_path + '/' + dataset_name + '-' + str(cnt).zfill(2) + '-SemanticMFF.png',
            #             np.clip(fused, 0, 255).astype(np.uint8))

    running_time_total = time.time() - start_time
    num_params = 0
    for p in spa_net.parameters():
        num_params += p.numel()
    tqdm.write("The number of spa_net's parameters: {} M".format(round(num_params / 10e5, 6)))
    num_params = 0
    for p in sem_net.parameters():
        num_params += p.numel()
    tqdm.write("The number of sem_net's parameters: {} M".format(round(num_params / 10e5, 6)))
    num_params = 0
    for p in gen_net.parameters():
        num_params += p.numel()
    tqdm.write("The number of gen_net's parameters: {} M".format(round(num_params / 10e5, 6)))
    print(f"Max allocated GPU memory: {torch.cuda.max_memory_allocated() / 1024 ** 3:.6f} GB")
    print(f"avg_process_time: {running_time_total:.6f} s")
    print('Testing finished!')
    print("results are saved in: " + save_path)
