import torch.optim as optim
import os
import numpy as np
from modeling.utils.UNet import UNet
from sseg_utils.loss import SegmentationLosses
from sseg_utils.saver import Saver
from sseg_utils.summaries import TensorboardSummary
from sseg_utils.metrics import Evaluator
import matplotlib.pyplot as plt
from dataloader_input_partial_map import get_all_scene_dataset, my_collate
import torch.utils.data as data
import torch
import torch.nn as nn
from core import cfg
from itertools import islice
import torch.nn.functional as F

# ======================================================================================
cfg.merge_from_file(
    'configs/exp_train_input_partial_map_occ_and_sem_for_pointgoal.yaml')
cfg.freeze()

output_folder = cfg.PRED.PARTIAL_MAP.SAVED_FOLDER
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

saver = Saver(output_folder)

cfg.dump(stream=open(f'{saver.experiment_dir}/experiment_config.yaml', 'w'))

# ==========================================================================================


def L1Loss(logit, target):
    mask_zero = (target > 0)
    logit = logit * mask_zero
    num_nonzero = torch.sum(mask_zero) + 1.
    # print(f'num_nonzero = {num_nonzero}')

    # result = loss(logit, target)
    result = (torch.abs(logit - target)).sum() / num_nonzero

    return result


def UNet_Loss(logit, mask, target):
    B, C, H, W = logit.shape
    # =========== split input into three channels
    logit_PS = logit[:, 0].unsqueeze(1)
    # print(f'logit_PS.shape = {logit_PS.shape}')
    logit_RS_RE = logit[:, 1:]
    # print(f'logit_RS_RE.shape = {logit_RS_RE.shape}')
    # ================ mask out pixels
    mask_PS = mask[:, 0].unsqueeze(1)
    mask_RS_RE = mask[:, 1:]
    # print(f'mask_PS.shape = {mask_PS.shape}')
    # print(f'mask_RS_RE.shape = {mask_RS_RE.shape}')
    logit_PS = logit_PS * mask_PS
    logit_RS_RE = logit_RS_RE * mask_RS_RE
    # print(f'logit_PS.shape = {logit_PS.shape}')
    # print(f'logit_RS_RE.shape = {logit_RS_RE.shape}')

    # =============== compute loss separately
    num_nonzero_PS = torch.sum(mask_PS) + 1.
    num_nonzero_RS_RE = torch.sum(mask_RS_RE) + 1.

    target_PS = target[:, 0].unsqueeze(1)
    target_RS_RE = target[:, 1:]
    # print(f'target_PS.shape = {target_PS.shape}')
    # print(f'target_RS_RE.shape = {target_RS_RE.shape}')

    loss_PS = F.binary_cross_entropy(
        logit_PS, target_PS, reduction='sum') / num_nonzero_PS
    loss_RS_RE = F.l1_loss(logit_RS_RE, target_RS_RE,
                           reduction='sum') / num_nonzero_RS_RE

    # loss = loss_PS + loss_RS_RE
    return loss_PS, loss_RS_RE


# ============================================ Define Tensorboard Summary =================================
summary = TensorboardSummary(saver.experiment_dir)
writer = summary.create_summary()

# =========================================================== Define Dataloader ==================================================
data_folder = cfg.PRED.PARTIAL_MAP.GEN_SAMPLES_SAVED_FOLDER
dataset_train = get_all_scene_dataset(
    'train', cfg.MAIN.TRAIN_SCENE_LIST, data_folder)
dataloader_train = data.DataLoader(dataset_train,
                                   batch_size=cfg.PRED.PARTIAL_MAP.BATCH_SIZE,
                                   num_workers=cfg.PRED.PARTIAL_MAP.NUM_WORKERS,
                                   shuffle=True,
                                   collate_fn=my_collate,
                                   pin_memory=True
                                   )

dataset_val = get_all_scene_dataset(
    'val', cfg.MAIN.VAL_SCENE_LIST, data_folder)
dataloader_val = data.DataLoader(dataset_val,
                                 batch_size=cfg.PRED.PARTIAL_MAP.BATCH_SIZE,
                                 num_workers=cfg.PRED.PARTIAL_MAP.NUM_WORKERS,
                                 shuffle=False,
                                 collate_fn=my_collate,
                                 pin_memory=True
                                 )

# ================================================================================================================================
# Define network
model = UNet(n_channel_in=cfg.PRED.PARTIAL_MAP.INPUT_CHANNEL,
             n_class_out=cfg.PRED.PARTIAL_MAP.OUTPUT_CHANNEL)
model = nn.DataParallel(model)
model = model.cuda()

# =========================================================== Define Optimizer ================================================
train_params = [{'params': model.parameters(), 'lr': cfg.PRED.PARTIAL_MAP.LR}]
optimizer = optim.Adam(
    train_params, lr=cfg.PRED.PARTIAL_MAP.LR, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

# Define Criterion
# whether to use class balanced weights
weight = None
criterion = UNet_Loss
best_test_loss = 1e10
lambda_RS_RE = cfg.PRED.PARTIAL_MAP.LAMBDA_RS_RE

# ===================================================== Resuming checkpoint ====================================================
best_pred = 0.0
if cfg.PRED.PARTIAL_MAP.RESUME != '':
    if not os.path.isfile(cfg.PRED.PARTIAL_MAP.RESUME):
        raise RuntimeError("=> no checkpoint found at '{}'" .format(
            cfg.PRED.PARTIAL_MAP.RESUME))
    checkpoint = torch.load(cfg.PRED.PARTIAL_MAP.RESUME)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    best_pred = checkpoint['best_pred']
    print("=> loaded checkpoint '{}' (epoch {})".format(
        cfg.PRED.PARTIAL_MAP.RESUME, checkpoint['epoch']))

# =================================================================trainin
for epoch in range(cfg.PRED.PARTIAL_MAP.EPOCHS):
    train_loss = 0.0
    model.train()
    iter_num = 0

    for batch in dataloader_train:
        print(f'epoch = {epoch}, iter_num = {iter_num}'.format(
            epoch, iter_num))
        images, masks, targets = batch['input'], batch['mask'], batch['output']
        # print('images = {}'.format(images.shape))   # (B, 47, 480, 480)
        # print('masks = {}'.format(masks.shape))     # (B, 3,  480, 480)
        # print('targets = {}'.format(targets.shape)) # (B, 3,  480, 480)

        images, masks, targets = images.cuda(), masks.cuda(), targets.cuda()

        # ================================================ compute loss =============================================
        output = model(images)  # B x 3 x H x W

        # print(f'output.shape = {output.shape}')
        loss_PS, loss_RS_RE = criterion(output, masks, targets)
        loss = loss_PS + lambda_RS_RE * loss_RS_RE

        # ================================================= compute gradient =================================================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        print(
            f'loss = {loss.item():.2f}, loss_PS = {loss_PS.item():.2f}, loss_RS_RE = {loss_RS_RE.item():.2f}')
        writer.add_scalars('train/total_loss_iter', {'PS_loss': loss_PS.item(),
                                                     'RS_RE_loss': lambda_RS_RE * loss_RS_RE.item(),
                                                     'total_loss': loss.item()}, iter_num + len(dataloader_train) * epoch)

        iter_num += 1

    writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
    print(
        f'[Epoch: {epoch}, numImages: {iter_num * cfg.PRED.PARTIAL_MAP.BATCH_SIZE}]')
    print(f'Loss: {train_loss:.2f}')

# ======================================================== evaluation stage =====================================================

    if epoch % cfg.PRED.PARTIAL_MAP.EVAL_INTERVAL == 0:
        model.eval()
        test_loss = 0.0
        iter_num = 0

        for batch in dataloader_val:
            print(f'epoch = {epoch}, iter_num = {iter_num}'.format(
                epoch, iter_num))
            images, masks, targets = batch['input'], batch['mask'], batch['output']
            # print('images = {}'.format(images))
            # print('targets = {}'.format(targets))
            images, masks, targets = images.cuda(), masks.cuda(), targets.cuda()

            # ========================== compute loss =====================
            with torch.no_grad():
                output = model(images)
            loss_PS, loss_RS_RE = criterion(output, masks, targets)
            loss = loss_PS + lambda_RS_RE * loss_RS_RE

            test_loss += loss.item()
            print(
                f'loss = {loss.item():.2f}, loss_PS = {loss_PS.item():.2f}, loss_RS_RE = {loss_RS_RE.item():.2f}')
            writer.add_scalars('val/total_loss_iter', {'PS_loss': loss_PS.item(),
                                                       'RS_RE_loss': lambda_RS_RE * loss_RS_RE.item(),
                                                       'total_loss': loss.item()}, iter_num + len(dataloader_val) * epoch)

            iter_num += 1

        # Fast test during the training
        writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
        print('Validation:')
        print(
            f'[Epoch: {epoch}, numImages: {iter_num * cfg.PRED.PARTIAL_MAP.BATCH_SIZE}]')
        print(f'Loss: {test_loss:.2f}')

        saver.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': test_loss,
        }, filename='checkpoint.pth.tar')

        # new_pred = mIoU
        if test_loss < best_test_loss:
            best_test_loss = test_loss

            saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': test_loss,
            }, filename='best_checkpoint.pth.tar')

    scheduler.step()
