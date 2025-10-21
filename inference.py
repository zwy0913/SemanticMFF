import os
import sys
import cv2
import glob
import time
import numpy as np
from tqdm import tqdm
from thop import profile, clever_format


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
from torchvision.utils import save_image
import torchvision.transforms.v2.functional as tv2f


from nets.network import SegFormer
from nets.s2net import SpatialNet
from nets.s2net import SemanticNet
from nets.generator import Generator

from misc.dataloader_gen import get_testloader

from misc.general import DeviceChecker
from misc.general import label2image


DEVICE = DeviceChecker.DEVICE

if __name__ == '__main__':
    # Configs
    model_spa_path = 'trained_weights/2025-04-12 00.32.58/debug_model_spa.ckpt'
    model_sem_path = 'trained_weights/2025-04-12 00.32.58/debug_model_sem.ckpt'
    model_gen_path = 'trained_weights/2025-04-21 08.17.14/debug_model_gen.ckpt'
    dataset_name = 'lytro'
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
    test_loader, test_list_A, test_list_B = get_testloader(dataset_path, 1, 99999)
    test_loader_tqdm = tqdm(test_loader, colour='#F59427', leave=True, file=sys.stdout)

    # for name, module in spa_net.named_modules():
    #     if isinstance(module, nn.Conv2d):
    #         prune.ln_structured(
    #             module,
    #             name="weight",
    #             amount=0.1,
    #             n=2,  # L2范数
    #             dim=0  # 沿输出通道维度剪枝
    #         )
    #
    # for name, module in sem_net.named_modules():
    #     if isinstance(module, nn.Conv2d):
    #         prune.ln_structured(
    #             module,
    #             name="weight",
    #             amount=0.1,
    #             n=2,  # L2范数
    #             dim=0  # 沿输出通道维度剪枝
    #         )

    

    # inference
    cnt = 1
    running_time = []
    flops_total = 0
    with torch.no_grad():
        for imgA, imgB in test_loader_tqdm:

            start_time = time.time()
            inputs = torch.cat((imgA, imgB), dim=1).to(DEVICE)
            spatial, spa_feats = spa_net(inputs)
            semantic, sem_feats = sem_net(inputs)
            # temp_decision = torch.argmax(F.log_softmax(semantic, dim=1), dim=1, keepdim=True)
            # temp_decision = torch.where(temp_decision > 0.5, 1., 0.)
            # x = torch.cat([imgA, imgB, temp_decision], dim=1)
            # x = torch.cat([imgA, imgB], dim=1)
            # imgA = torch.cat([imgA, temp_decision], dim=1)
            # imgB = torch.cat([imgB, 1 - temp_decision], dim=1)
            # spa_feats = [feat.detach().clone() * 3 for feat in spa_feats]
            # sem_feats = [feat.detach().clone() * 3 for feat in sem_feats]
            decision, _ = gen_net(imgA, imgB, spa_feats, sem_feats)
            running_time.append(time.time() - start_time)

            flops, _ = profile(spa_net, inputs=[inputs], verbose=False)
            flops_total += flops
            flops, _ = profile(sem_net, inputs=[inputs], verbose=False)
            flops_total += flops
            flops, _ = profile(gen_net, inputs=[imgA,
                                                imgB,
                                                spa_feats,
                                                sem_feats], verbose=False)
            flops_total += flops

            spa = torch.einsum('c w h -> w h c', spatial[0]).clone().detach().cpu().numpy()
            cv2.imwrite(save_path + '/' + dataset_name + '-' + str(cnt).zfill(2) + '-spa.png', cv2.cvtColor((spa * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

            # ss = pred[0].clone().detach().cpu().numpy()
            sem = label2image(torch.argmax(F.log_softmax(semantic, dim=1), dim=1))
            cv2.imwrite(save_path + '/' + dataset_name + '-' + str(cnt).zfill(2) + '-sem.png', cv2.cvtColor(sem, cv2.COLOR_RGB2BGR))

            originalA = cv2.imread(test_list_A[cnt - 1])
            originalB = cv2.imread(test_list_B[cnt - 1])
            # decision = torch.where(decision > 0.7, 1., decision)
            # decision = torch.where(decision < 0.3, 0., decision)
            decision = torch.where(decision > 0.5, 1., 0.)
            # decision = tv2f.resize(decision, [originalA.shape[0], originalA.shape[1]])
            decision = torch.einsum('c w h -> w h c', decision[0]).clone().detach().cpu().numpy()
            fused = originalA * decision + originalB * (1 - decision)
            cv2.imwrite(save_path + '/' + dataset_name + '-' + str(cnt).zfill(2) + '-dm.png', np.clip(decision * 255, 0, 255).astype(np.uint8))
            cv2.imwrite(save_path + '/' + dataset_name + '-' + str(cnt).zfill(2) + '-SemanticMFF.png', np.clip(fused, 0, 255).astype(np.uint8))


            cnt += 1

    running_time_total = 0
    for i in range(len(running_time)):
        # print("process_time: {} s".format(running_time[i]))
        if i != 0:
            running_time_total += running_time[i]
        # running_time_total += running_time[i]
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
    print(f'avg flops:{flops_total / len(running_time) / 1000 ** 3:.6f} G')
    print(f"Max allocated GPU memory: {torch.cuda.max_memory_allocated() / 1024 ** 3:.6f} GB")
    print(f"avg_process_time: {running_time_total / (len(running_time)):.6f} s")
    print('Testing finished!')
    print("results are saved in: " + save_path)
