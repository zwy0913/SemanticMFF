import os
import glob
import sys
import datetime
import random
import yaml
import cv2
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.utils import save_image
import torchvision.transforms.v2.functional as tv2f

from nets.network import SegFormer
from nets.s2net import SpatialNet
from nets.s2net import SemanticNet
from nets.generator import Generator

from misc.dataloader import get_trainloader

from misc.general import DeviceChecker
from misc.general import get_net_para
from misc.general import get_one_hot
from misc.general import label2image
from misc.general import sigmoid_rampup
from misc.general import perturb_seg
from misc.general import LoggingSaveWeights

from misc.losses import l1_loss, bce_loss
from misc.losses import l2_loss
from misc.losses import ce_loss
from misc.losses import bce_loss
from misc.losses import ssim_loss
from misc.losses import grad_loss
from misc.losses import dice_loss
from misc.losses import focal_loss

DEVICE = DeviceChecker.DEVICE

if __name__ == '__main__':
    # Configs
    data_path = ['L:/Datasets/Train_Valid/MFFdatasets-semantic', ]
    init_size = 400
    model_input_size = 384
    model_save_path = 'trained_weights'
    set_size = 99999
    batch_size = 8
    epochs = 66
    learning_rate = 0.0002
    gamma = 0.9
    scheduler_step = 1
    save_every_weights = False
    load = ''

    # nets
    spa_net = SpatialNet().to(DEVICE).train()
    sem_net = SemanticNet().to(DEVICE).train()
    gen_net = Generator().to(DEVICE).train()
    # print(Network)
    if load != '':
        spa_net.load_state_dict(torch.load('debug_model_spa.ckpt'))
        sem_net.load_state_dict(torch.load('debug_model_sem.ckpt'))
        # gen_net.load_state_dict(torch.load('debug_model_gen.ckpt'))

    # data
    train_loader, train_data_size = get_trainloader(data_path,
                                                    batch_size,
                                                    set_size,
                                                    init_size,
                                                    model_input_size)

    # log config
    log_config = ('data_path: ' + str(data_path)
                + '\nloader_length: ' + str(train_data_size) + ' // ' + str(batch_size) + ' = ' + str(len(train_loader))
                + '\ninit_size: ' + str(init_size)
                + '\nmodel_input_size: ' + str(model_input_size)
                + '\nset_size: ' + str(set_size)
                + '\nbatch_size: ' + str(batch_size)
                + '\nepochs: ' + str(epochs)
                + '\nlearning_rate: ' + str(learning_rate)
                + '\ngamma: ' + str(gamma)
                + '\nscheduler_step: ' + str(scheduler_step)
                + '\nsave_every_weights: ' + str(save_every_weights)
                + '\nspa_net_params: ' + str(round(get_net_para(spa_net) / 10e5, 6)) + 'M'
                + '\nsem_net_params: ' + str(round(get_net_para(sem_net) / 10e5, 6)) + 'M'
                + '\ngen_net_params: ' + str(round(get_net_para(gen_net) / 10e5, 6)) + 'M'
                )
    LS = LoggingSaveWeights(save_path=model_save_path,
                            config=log_config,
                            save_every_weights=save_every_weights)
    tqdm.write(log_config)

    # opt & sche
    spa_optimizer = torch.optim.AdamW(spa_net.parameters(), learning_rate, weight_decay=0.0001)
    sem_optimizer = torch.optim.AdamW(sem_net.parameters(), learning_rate, weight_decay=0.0001)
    gen_optimizer = torch.optim.AdamW(gen_net.parameters(), learning_rate, weight_decay=0.0001)
    spa_scheduler = torch.optim.lr_scheduler.StepLR(spa_optimizer, scheduler_step, gamma)
    sem_scheduler = torch.optim.lr_scheduler.StepLR(sem_optimizer, scheduler_step, gamma)
    gen_scheduler = torch.optim.lr_scheduler.StepLR(gen_optimizer, scheduler_step, gamma)

    # init
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    spa_scaler = torch.cuda.amp.GradScaler()
    sem_scaler = torch.cuda.amp.GradScaler()
    gen_scaler = torch.cuda.amp.GradScaler()

    # iter start
    iter_num = 0
    tqdm.write('\nStart iteration...\n')
    for epoch in range(epochs):

        loss_spa = 0
        epoch_loss_spa = 0
        loss_sem = 0
        epoch_loss_sem = 0
        loss_gen = 0
        epoch_loss_gen = 0

        # progress bar
        train_loader_tqdm = tqdm(train_loader, colour='#7ce9e6', leave=False, file=sys.stdout)
        for A, B, DM, GT, CL in train_loader_tqdm:

            # 只做前面
            # ########################
            # # tune spatial network #
            # ########################
            # spa_optimizer.zero_grad()
            # with ((torch.autocast(device_type=DEVICE, dtype=torch.float16))):
            #     spa_inputs = torch.cat((A, B), dim=1)
            #     spatial, logits = spa_net(spa_inputs)
            #
            #     # cal loss
            #     loss_spa = l2_loss(spatial, DM) + bce_loss(logits, DM) + 2 * grad_loss(spatial, DM)
            #     epoch_loss_spa += loss_spa / len(train_loader_tqdm)
            #
            # # backward
            # spa_scaler.scale(loss_spa).backward()
            # spa_scaler.step(spa_optimizer)
            # spa_scaler.update()
            #
            # #########################
            # # tune semantic network #
            # #########################
            # sem_optimizer.zero_grad()
            # with ((torch.autocast(device_type=DEVICE, dtype=torch.float16))):
            #     sem_inputs = torch.cat((A, B), dim=1)
            #     semantic, sem_feats = sem_net(sem_inputs)
            #
            #     # cal loss
            #     temp_decision = torch.argmax(F.softmax(semantic, dim=1), dim=1, keepdim=True)
            #     temp_decision = torch.where(temp_decision > 0, 1., 0.)
            #     temp_fused = temp_decision * A + (1 - temp_decision) * B
            #     loss_sem = 0.5 * ce_loss(semantic, CL) + 0.5 * dice_loss(semantic, CL) + 1 * l1_loss(temp_fused, GT)
            #     epoch_loss_sem += loss_sem / len(train_loader_tqdm)
            #
            # # backward
            # sem_scaler.scale(loss_sem).backward()
            # sem_scaler.step(sem_optimizer)
            # sem_scaler.update()



            # # 交替
            # if iter_num % 2 == 0:
            #     ########################
            #     # tune spatial network #
            #     ########################
            #     spa_optimizer.zero_grad()
            #     with ((torch.autocast(device_type=DEVICE, dtype=torch.float16))):
            #         spa_inputs = torch.cat((A, B), dim=1)
            #         spatial, spa_feats = spa_net(spa_inputs)
            #         # cal loss
            #         loss_spa = 0.3 * l1_loss(spatial, GT) + 0.7 * ssim_loss(spatial, GT)
            #         epoch_loss_spa += loss_spa / (len(train_loader_tqdm) / 2)
            #     # backward
            #     spa_scaler.scale(loss_spa).backward()
            #     spa_scaler.step(spa_optimizer)
            #     spa_scaler.update()
            #
            #     #########################
            #     # tune semantic network #
            #     #########################
            #     sem_optimizer.zero_grad()
            #     with ((torch.autocast(device_type=DEVICE, dtype=torch.float16))):
            #         sem_inputs = torch.cat((A, B), dim=1)
            #         semantic, sem_feats = sem_net(sem_inputs)
            #         # cal loss
            #         temp_decision = torch.argmax(F.softmax(semantic, dim=1), dim=1, keepdim=True)
            #         temp_decision = torch.where(temp_decision > 0, 1., 0.)
            #         # temp_fused = temp_decision * A + (1 - temp_decision) * B
            #         loss_sem = 0.7 * ce_loss(semantic, CL) + 0.3 * dice_loss(semantic, CL) + 1 * l1_loss(temp_decision, DM)
            #         epoch_loss_sem += loss_sem / (len(train_loader_tqdm) / 2)
            #     # backward
            #     sem_scaler.scale(loss_sem).backward()
            #     sem_scaler.step(sem_optimizer)
            #     sem_scaler.update()
            #
            # else:
            #     #########################
            #     #### tune generator  ####
            #     #########################
            #     gen_optimizer.zero_grad()
            #     with torch.no_grad():
            #         spatial, spa_feats = spa_net(torch.cat((A, B), dim=1))
            #         semantic, sem_feats = sem_net(torch.cat((A, B), dim=1))
            #     temp_decision = torch.argmax(F.softmax(semantic, dim=1), dim=1, keepdim=True)
            #     temp_decision = torch.where(temp_decision > 0, 1., 0.)
            #     with ((torch.autocast(device_type=DEVICE, dtype=torch.float16))):
            #         x = torch.cat([spatial.detach().clone(), perturb_seg(temp_decision.detach().clone(), intensity=50)],
            #                       dim=1)
            #         # x = torch.cat([A, temp_decision.detach().clone()], dim=1)
            #         # x = torch.cat([A, B], dim=1)
            #         spa_feats = [feat.detach().clone() for feat in spa_feats]
            #         sem_feats = [feat.detach().clone() for feat in sem_feats]
            #         decision, logits = gen_net(x, spa_feats, sem_feats)
            #         # cal loss
            #         loss_gen = l2_loss(decision, DM) + bce_loss(logits, DM) + 2 * grad_loss(decision, DM)
            #         epoch_loss_gen += loss_gen / (len(train_loader_tqdm) / 2)
            #     # backward
            #     gen_scaler.scale(loss_gen).backward()
            #     gen_scaler.step(gen_optimizer)
            #     gen_scaler.update()



            # 同步
            ########################
            # tune spatial network #
            ########################
            spa_optimizer.zero_grad()
            with ((torch.autocast(device_type=DEVICE, dtype=torch.float16))):
                spa_inputs = torch.cat((A, B), dim=1)
                spatial, spa_feats = spa_net(spa_inputs)
                # cal loss
                loss_spa = 0.3 * l1_loss(spatial, GT) + 0.7 * ssim_loss(spatial, GT)
                epoch_loss_spa += loss_spa / len(train_loader_tqdm)
            # backward
            spa_scaler.scale(loss_spa).backward()
            spa_scaler.step(spa_optimizer)
            spa_scaler.update()

            #########################
            # tune semantic network #
            #########################
            sem_optimizer.zero_grad()
            with ((torch.autocast(device_type=DEVICE, dtype=torch.float16))):
                sem_inputs = torch.cat((A, B), dim=1)
                semantic, sem_feats = sem_net(sem_inputs)
                # cal loss
                temp_decision = torch.argmax(F.softmax(semantic, dim=1), dim=1, keepdim=True)
                temp_decision = torch.where(temp_decision > 0.5, 1., 0.)
                # temp_fused = temp_decision * A + (1 - temp_decision) * B
                loss_sem = 0.7 * ce_loss(semantic, CL) + 0.3 * dice_loss(semantic, CL) + 5 * l1_loss(temp_decision, DM)
                epoch_loss_sem += loss_sem / len(train_loader_tqdm)
            # backward
            sem_scaler.scale(loss_sem).backward()
            sem_scaler.step(sem_optimizer)
            sem_scaler.update()

            # #########################
            # #### tune generator  ####
            # #########################
            # gen_optimizer.zero_grad()
            # # with torch.no_grad():
            # #     spatial, spa_feats = spa_net(torch.cat((A, B), dim=1))
            # #     semantic, sem_feats = sem_net(torch.cat((A, B), dim=1))
            # # temp_decision = torch.argmax(F.softmax(semantic, dim=1), dim=1, keepdim=True)
            # # temp_decision = torch.where(temp_decision > 0, 1., 0.)
            # with ((torch.autocast(device_type=DEVICE, dtype=torch.float16))):
            #     x = torch.cat([spatial.detach().clone(), perturb_seg(temp_decision.detach().clone(), intensity=66)], dim=1)
            #     # x = torch.cat([A, temp_decision.detach().clone()], dim=1)
            #     # x = torch.cat([A, B], dim=1)
            #     spa_feats=[feat.detach().clone() for feat in spa_feats]
            #     sem_feats=[feat.detach().clone() for feat in sem_feats]
            #     decision, logits = gen_net(x, spa_feats, sem_feats)
            #     # cal loss
            #     loss_gen = l2_loss(decision, DM) + bce_loss(logits, DM) + 2 * grad_loss(decision, DM)
            #     epoch_loss_gen += loss_gen / len(train_loader_tqdm)
            # # backward
            # gen_scaler.scale(loss_gen).backward()
            # gen_scaler.step(gen_optimizer)
            # gen_scaler.update()


            # debug
            # s_1 = torch.einsum('c w h -> w h c', DM[0]).clone().detach().cpu().numpy()
            # s_2 = torch.einsum('c w h -> w h c', temp_decision[0]).clone().detach().cpu().numpy()
            # s_3 = torch.einsum('c w h -> w h c', spatial[0]).clone().detach().cpu().numpy()
            # cv2.imwrite('1.png', s_1 * 255)
            # cv2.imwrite('2.png', s_2 * 255)
            # cv2.imwrite('3.png', cv2.cvtColor((s_3 * 255).astype('uint8'), cv2.COLOR_RGB2BGR))
            # s_4 = torch.einsum('c w h -> w h c', decision[0]).clone().detach().cpu().numpy()
            # cv2.imwrite('4.png', s_4 * 255)


            # update progress
            train_loader_tqdm.set_description("Epoch %s" % (str(epoch + 1)))
            train_loader_tqdm.set_postfix({'sp': float(loss_spa), 'se': float(loss_sem), 'gen': float(loss_gen) })

            # iter cnt
            iter_num += 1

            # save for debug
            if iter_num % (len(train_loader_tqdm) // 4) == 0:
                torch.save(spa_net.state_dict(), 'debug_model_spa.ckpt')
                torch.save(sem_net.state_dict(), 'debug_model_sem.ckpt')
                torch.save(gen_net.state_dict(), 'debug_model_gen.ckpt')

        # update LR
        spa_scheduler.step()
        # sem_scheduler.step()
        # gen_scheduler.step()

        # save model
        LS.save_model(spa_net, loss_spa, epoch + 1, 'model_spa')
        LS.save_model(sem_net, loss_sem, epoch + 1, 'model_sem')
        LS.save_model(gen_net, loss_gen, epoch + 1, 'model_gen')

        # log to file
        log_contents = (f"[{str(datetime.datetime.now().strftime('%Y-%m-%d %H.%M.%S'))}] Epoch {epoch + 1} "
                        f"- sp : {epoch_loss_spa:.4f} "
                        f"- se : {epoch_loss_sem:.4f} "
                        f"- ge : {epoch_loss_gen:.4f}\n")
        LS.log_content(log_contents)

        # print logs
        tqdm.write(log_contents)

    # iter end
    tqdm.write('\nIteration ended.\n')
