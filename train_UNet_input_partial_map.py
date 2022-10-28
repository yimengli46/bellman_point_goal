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

#======================================================================================
cfg.merge_from_file('configs/exp_train_input_partial_map_occ_and_sem_for_pointgoal.yaml')
cfg.freeze()

output_folder = cfg.PRED.PARTIAL_MAP.SAVED_FOLDER
if not os.path.exists(output_folder):
	os.mkdir(output_folder)

saver = Saver(output_folder)

cfg.dump(stream=open(f'{saver.experiment_dir}/experiment_config.yaml', 'w'))

#==========================================================================================

def MSELoss(logit, target):
	mask_zero = (target > 0)
	logit = logit * mask_zero
	num_nonzero = torch.sum(mask_zero) + 1.
	#print(f'num_nonzero = {num_nonzero}')

	#result = loss(logit, target)
	result = ((logit - target)**2).sum() / num_nonzero

	return result

def L1Loss(logit, target):
	mask_zero = (target > 0)
	logit = logit * mask_zero
	num_nonzero = torch.sum(mask_zero) + 1.
	#print(f'num_nonzero = {num_nonzero}')

	#result = loss(logit, target)
	result = (torch.abs(logit - target)).sum() / num_nonzero

	return result

#============================================ Define Tensorboard Summary =================================
summary = TensorboardSummary(saver.experiment_dir)
writer = summary.create_summary()

#=========================================================== Define Dataloader ==================================================
data_folder = cfg.PRED.PARTIAL_MAP.GEN_SAMPLES_SAVED_FOLDER
dataset_train = get_all_scene_dataset('train', cfg.MAIN.TRAIN_SCENE_LIST, data_folder)
dataloader_train = data.DataLoader(dataset_train, 
	batch_size=cfg.PRED.PARTIAL_MAP.BATCH_SIZE, 
	num_workers=cfg.PRED.PARTIAL_MAP.NUM_WORKERS,
	shuffle=True,
	collate_fn=my_collate,
	)

dataset_val = get_all_scene_dataset('val', cfg.MAIN.VAL_SCENE_LIST, data_folder)
dataloader_val = data.DataLoader(dataset_val, 
	batch_size=cfg.PRED.PARTIAL_MAP.BATCH_SIZE, 
	num_workers=cfg.PRED.PARTIAL_MAP.NUM_WORKERS,
	shuffle=False,
	collate_fn=my_collate,
	)

#================================================================================================================================
# Define network
model = UNet(n_channel_in=cfg.PRED.PARTIAL_MAP.INPUT_CHANNEL, n_class_out=cfg.PRED.PARTIAL_MAP.OUTPUT_CHANNEL)
model = nn.DataParallel(model)
model = model.cuda()

#=========================================================== Define Optimizer ================================================
import torch.optim as optim
train_params = [{'params': model.parameters(), 'lr': cfg.PRED.PARTIAL_MAP.LR}]
optimizer = optim.Adam(train_params, lr=cfg.PRED.PARTIAL_MAP.LR, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

# Define Criterion
# whether to use class balanced weights
weight = None
criterion = L1Loss
best_test_loss = 1e10

#===================================================== Resuming checkpoint ====================================================
best_pred = 0.0
if cfg.PRED.PARTIAL_MAP.RESUME != '':
	if not os.path.isfile(cfg.PRED.PARTIAL_MAP.RESUME):
		raise RuntimeError("=> no checkpoint found at '{}'" .format(cfg.PRED.PARTIAL_MAP.RESUME))
	checkpoint = torch.load(cfg.PRED.PARTIAL_MAP.RESUME)
	model.load_state_dict(checkpoint['state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer'])
	best_pred = checkpoint['best_pred']
	print("=> loaded checkpoint '{}' (epoch {})".format(cfg.PRED.PARTIAL_MAP.RESUME, checkpoint['epoch']))

#=================================================================trainin
for epoch in range(cfg.PRED.PARTIAL_MAP.EPOCHS):
	train_loss = 0.0
	model.train()
	iter_num = 0
	
	for batch in dataloader_train:
		print('epoch = {}, iter_num = {}'.format(epoch, iter_num))
		images, targets = batch['input'], batch['output']
		#print('images = {}'.format(images.shape))
		#print('targets = {}'.format(targets.shape))
		#assert 1==2
		images, targets = images.cuda(), targets.cuda()
		
		#================================================ compute loss =============================================
		output = model(images) # batchsize x 1 x H x W
		#print(f'output.shape = {output.shape}')
		loss = criterion(output, targets)

		#================================================= compute gradient =================================================
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		train_loss += loss.item()
		print('Train loss: %.3f' % (train_loss / (iter_num + 1)))
		writer.add_scalar('train/total_loss_iter', loss.item(), iter_num + len(dataloader_train) * epoch)

		iter_num += 1

	writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
	print('[Epoch: %d, numImages: %5d]' % (epoch, iter_num * cfg.PRED.PARTIAL_MAP.BATCH_SIZE))
	print('Loss: %.3f' % train_loss)

#======================================================== evaluation stage =====================================================

	if epoch % cfg.PRED.PARTIAL_MAP.EVAL_INTERVAL == 0:
		model.eval()
		test_loss = 0.0
		iter_num = 0

		for batch in dataloader_val:
			print('epoch = {}, iter_num = {}'.format(epoch, iter_num))
			images, targets = batch['input'], batch['output']
			#print('images = {}'.format(images))
			#print('targets = {}'.format(targets))
			images, targets = images.cuda(), targets.cuda()

			#========================== compute loss =====================
			with torch.no_grad():
				output = model(images)
			loss = criterion(output, targets)

			test_loss += loss.item()
			print('Test loss: %.3f' % (test_loss / (iter_num + 1)))
			writer.add_scalar('val/total_loss_iter', loss.item(), iter_num + len(dataloader_val) * epoch)

			iter_num += 1

		# Fast test during the training
		writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
		print('Validation:')
		print('[Epoch: %d, numImages: %5d]' % (epoch, iter_num * cfg.PRED.PARTIAL_MAP.BATCH_SIZE))
		print('Loss: %.3f' % test_loss)

		saver.save_checkpoint({
				'epoch': epoch + 1,
				'state_dict': model.state_dict(),
				'optimizer': optimizer.state_dict(),
				'loss': test_loss,
			}, filename='checkpoint.pth.tar')

		#new_pred = mIoU
		if test_loss < best_test_loss:
			best_test_loss = test_loss

			saver.save_checkpoint({
				'epoch': epoch + 1,
				'state_dict': model.state_dict(),
				'optimizer': optimizer.state_dict(),
				'loss': test_loss,
			}, filename='best_checkpoint.pth.tar')

	scheduler.step()




