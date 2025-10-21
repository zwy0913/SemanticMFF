import datetime
import sys

import torch
import torch.nn.functional as F
from tqdm import tqdm

from misc.dataloader_gen import get_trainloader
from misc.general import DeviceChecker
from misc.general import LoggingSaveWeights
from misc.general import get_net_para
from misc.general import perturb_seg
from misc.losses import bce_loss
from misc.losses import grad_loss
from misc.losses import l1_loss
from misc.losses import l2_loss
from nets.generator import Generator
from nets.s2net import SemanticNet
from nets.s2net import SpatialNet

DEVICE = DeviceChecker.DEVICE

if __name__ == '__main__':
    # Configs
    data_path = [
        'L:/Datasets/Train_Valid/MFFdatasets-duts-tr',
        'L:/Datasets/Train_Valid/MFFdatasets-duts-te',
        'L:/Datasets/Train_Valid/MFFdatasets-ECSSD',
        # 'L:/Datasets/Train_Valid/MFFdatasets-msra10k',
        # 'L:/Datasets/Train_Valid/MFFdatasets-dut-omron',
        # 'L:/Datasets/Train_Valid/MFFdatasets-voc12',
        # 'L:/Datasets/Train_Valid/MFFdatasets',
    ]
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
    load = 't'

    # nets
    spa_net = SpatialNet().to(DEVICE).eval()
    sem_net = SemanticNet().to(DEVICE).eval()
    gen_net = Generator().to(DEVICE).train()
    # print(Network)
    if load != '':
        spa_net.load_state_dict(torch.load('trained_weights/2025-04-12 00.32.58/debug_model_spa.ckpt'))
        sem_net.load_state_dict(torch.load('trained_weights/2025-04-12 00.32.58/debug_model_sem.ckpt'))

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
    spa_optimizer = torch.optim.AdamW(spa_net.parameters(), learning_rate, weight_decay=0.000)
    sem_optimizer = torch.optim.AdamW(sem_net.parameters(), learning_rate, weight_decay=0.000)
    gen_optimizer = torch.optim.AdamW(gen_net.parameters(), learning_rate, weight_decay=0.000)
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
        for A, B, DM, GT in train_loader_tqdm:

            #########################
            #### tune generator  ####
            #########################
            gen_optimizer.zero_grad()
            with torch.no_grad():
                spatial, spa_feats = spa_net(torch.cat((A, B), dim=1))
                semantic, sem_feats = sem_net(torch.cat((A, B), dim=1))
            # temp_decision = torch.argmax(F.softmax(semantic, dim=1), dim=1, keepdim=True)
            # temp_decision = torch.where(temp_decision > 0.5, 1., 0.)
            # decision_input = perturb_seg(temp_decision.detach().clone(), intensity=64)
            # # decision_input = temp_decision.detach().clone()
            with ((torch.autocast(device_type=DEVICE, dtype=torch.float16))):
                # x = torch.cat([A, B, perturb_seg(temp_decision.detach().clone(), intensity=32)], dim=1)
                # x = torch.cat([A, temp_decision.detach().clone()], dim=1)
                # x = torch.cat([A, B], dim=1)
                # A = torch.cat([A, decision_input], dim=1)
                # B = torch.cat([B, 1 - decision_input], dim=1)
                # spa_feats = [feat.detach().clone() * 3 for feat in spa_feats]
                # sem_feats = [feat.detach().clone() * 3 for feat in sem_feats]
                decision, logits = gen_net(A, B, spa_feats, sem_feats)
                # cal loss
                loss_gen = l2_loss(decision, DM) + bce_loss(logits, DM) + 2 * grad_loss(decision, DM)
                epoch_loss_gen += loss_gen / len(train_loader_tqdm)
            # backward
            gen_scaler.scale(loss_gen).backward()
            gen_scaler.step(gen_optimizer)
            gen_scaler.update()


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
        # spa_scheduler.step()
        # sem_scheduler.step()
        gen_scheduler.step()

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
