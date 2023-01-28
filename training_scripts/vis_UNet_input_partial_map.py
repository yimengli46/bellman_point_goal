import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from modeling.utils.UNet import UNet
import torch.nn.functional as F
from modeling.utils.baseline_utils import apply_color_to_map
from core import cfg
import torch.utils.data as data
from itertools import islice
import cv2
from dataloader_input_partial_map import get_all_scene_dataset, my_collate

cfg.merge_from_file('configs/exp_train_input_partial_map_occ_and_sem.yaml')
cfg.freeze()

Total_Samples = 100
BATCH_SIZE = 4

data_folder = cfg.PRED.PARTIAL_MAP.GEN_SAMPLES_SAVED_FOLDER
dataset_val = get_all_scene_dataset('val', ['2azQ1b91cZZ_0'], data_folder)
dataloader_val = data.DataLoader(
    dataset_val,
    batch_size=BATCH_SIZE,
    num_workers=2,
    shuffle=False,
    collate_fn=my_collate,
)

device = torch.device('cuda')
'''
model = UNet(n_channel_in=cfg.PRED.PARTIAL_MAP.INPUT_CHANNEL, n_class_out=cfg.PRED.PARTIAL_MAP.OUTPUT_CHANNEL).cuda()
checkpoint = torch.load(f'{cfg.PRED.PARTIAL_MAP.SAVED_FOLDER}/{cfg.PRED.PARTIAL_MAP.INPUT}/experiment_6/checkpoint.pth.tar')
model.load_state_dict(checkpoint['state_dict'])
model.eval()
'''

#assert 1==2

with torch.no_grad():
    count = 0
    for batch in dataloader_val:
        images, targets, HWs, frontiers = batch['input'], batch[
            'output'], batch['shape'], batch['frontiers']
        original_targets = batch['original_target']
        #print('images = {}'.format(images))
        #print('targets = {}'.format(targets))
        #images, targets = images.cuda(), targets.cuda()

        #outputs = model(images)

        #images = images.cpu().numpy()
        #targets = targets.cpu().numpy()
        #outputs = outputs.cpu().numpy() # batch_size, 4, H, W

        for idx in range(BATCH_SIZE):
            print(f'count = {count}')
            occ_map_Mp = np.argmax(images[idx, 0:3], axis=0)  # H, W
            sem_map_Mp = np.argmax(images[idx, 3:], axis=0)  # H, W

            target = original_targets[idx]  # H, W, 4
            output = outputs[idx].transpose((1, 2, 0))  # H, W, 4

            H, W = HWs[idx]
            frons = frontiers[idx]

            occ_map_Mp = cv2.resize(occ_map_Mp, (W, H),
                                    interpolation=cv2.INTER_NEAREST)
            sem_map_Mp = cv2.resize(sem_map_Mp, (W, H),
                                    interpolation=cv2.INTER_NEAREST)
            output = cv2.resize(output, (W, H),
                                interpolation=cv2.INTER_NEAREST)

            color_sem_map_Mp = apply_color_to_map(sem_map_Mp)
            #assert 1==2

            #======================= show the whole prediction ======================
            #'''
            fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(20, 30))
            ax[0][0].imshow(occ_map_Mp, cmap='gray')
            ax[0][0].get_xaxis().set_visible(False)
            ax[0][0].get_yaxis().set_visible(False)
            ax[0][0].set_title('input: occupancy_map_Mp')
            ax[0][1].imshow(color_sem_map_Mp)
            ax[0][1].get_xaxis().set_visible(False)
            ax[0][1].get_yaxis().set_visible(False)
            ax[0][1].set_title('input: semantic_map_Mp')
            ax[1][0].imshow(target[:, :, 0])
            ax[1][0].get_xaxis().set_visible(False)
            ax[1][0].get_yaxis().set_visible(False)
            ax[1][0].set_title('label: U_a')
            ax[1][1].imshow(output[:, :, 0])
            ax[1][1].get_xaxis().set_visible(False)
            ax[1][1].get_yaxis().set_visible(False)
            ax[1][1].set_title('predict: U_a')
            ax[2][0].imshow(target[:, :, 1])
            ax[2][0].get_xaxis().set_visible(False)
            ax[2][0].get_yaxis().set_visible(False)
            ax[2][0].set_title('label: U_dall')
            ax[2][1].imshow(output[:, :, 1])
            ax[2][1].get_xaxis().set_visible(False)
            ax[2][1].get_yaxis().set_visible(False)
            ax[2][1].set_title('predict: U_dall')
            fig.tight_layout()
            plt.show()
            #fig.savefig(f'{cfg.PRED.PARTIAL_MAP.SAVED_FOLDER}/img_{count}.jpg')
            #plt.close()
            #'''

            #===================== show the prediction on the frontiers ===================
            '''
			mask_zero = (target == 0)
			output[mask_zero] = 0

			fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(15, 30), dpi=100)
			ax[0][0].imshow(occ_map_Mp, cmap='gray')
			ax[0][0].get_xaxis().set_visible(False)
			ax[0][0].get_yaxis().set_visible(False)
			ax[0][0].set_title('input: occupancy_map_Mp')
			
			ax[0][1].imshow(color_sem_map_Mp)
			ax[0][1].get_xaxis().set_visible(False)
			ax[0][1].get_yaxis().set_visible(False)
			ax[0][1].set_title('input: semantic_map_Mp')
			
			ax[1][0].imshow(target[:, :, 0])
			ax[1][0].get_xaxis().set_visible(False)
			ax[1][0].get_yaxis().set_visible(False)
			ax[1][0].set_title('label: U_a')
			ax[1][1].imshow(output[:, :, 0])
			ax[1][1].get_xaxis().set_visible(False)
			ax[1][1].get_yaxis().set_visible(False)
			ax[1][1].set_title('predict: U_a')
			ax[2][0].imshow(target[:, :, 1])
			ax[2][0].get_xaxis().set_visible(False)
			ax[2][0].get_yaxis().set_visible(False)
			ax[2][0].set_title('label: U_dall')
			ax[2][1].imshow(output[:, :, 1])
			ax[2][1].get_xaxis().set_visible(False)
			ax[2][1].get_yaxis().set_visible(False)
			ax[2][1].set_title('predict: U_dall')
			'''

            #==================== compare each frontier prediction ===============
            '''
			for fron in frons:
				points = fron.points.transpose()
				centroid_y, centroid_x = fron.centroid

				vals = target[points[:, 0], points[:, 1], 0]
				mask_points = points[vals > 0]

				if mask_points.shape[0] > 0:
					# print U_a
					target_val = np.mean(target[mask_points[:, 0], mask_points[:, 1], 0])
					ax[1][0].text(centroid_x, centroid_y, f'{target_val:.2f}', fontsize=20, color='white')

					output_val = np.mean(output[mask_points[:, 0], mask_points[:, 1], 0])
					ax[1][1].text(centroid_x, centroid_y, f'{output_val:.2f}', fontsize=20, color='white')
					# print U_dall
					target_val = np.mean(target[mask_points[:, 0], mask_points[:, 1], 1])
					ax[2][0].text(centroid_x, centroid_y, f'{target_val:.2f}', fontsize=20, color='white')

					output_val = np.mean(output[mask_points[:, 0], mask_points[:, 1], 1])
					ax[2][1].text(centroid_x, centroid_y, f'{output_val:.2f}', fontsize=20, color='white')
			'''

            #fig.tight_layout()
            #plt.show()
            #fig.savefig(f'{cfg.PRED.PARTIAL_MAP.SAVED_FOLDER}/img_{count}_zeroout_occmap_only.jpg')
            #plt.close()
            #'''

            count += 1
            #assert 1==2
