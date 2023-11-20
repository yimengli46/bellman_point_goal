import numpy as np
import numpy.linalg as LA
import cv2
import matplotlib.pyplot as plt
import random
from core import cfg
import torch.utils.data as data
import torch
import torch.nn.functional as F
from random import Random
import os
import glob
import pickle
from modeling.utils.baseline_utils import apply_color_to_map
import bz2
import _pickle as cPickle


class MP3DSceneDataset(data.Dataset):

    def __init__(self, split, scene_name, data_folder=''):
        self.split = split
        self.scene_name = scene_name

        self.saved_folder = f'{data_folder}/{self.split}/{self.scene_name}'

        self.sample_name_list = [os.path.splitext(os.path.basename(x))[0]
                                 for x in sorted(glob.glob(f'{self.saved_folder}/*.pbz2'))]

    def __len__(self):
        return len(self.sample_name_list)

    def __getitem__(self, i):
        with bz2.BZ2File(f'{self.saved_folder}/{self.sample_name_list[i]}.pbz2', 'rb') as fp:
            eps_data = cPickle.load(fp)
            M_p = eps_data['M_p']  # 2 x H x W
            U_PS = eps_data['U_PS']  # H x W
            U_RS = eps_data['U_RS']  # H x W
            U_RE = eps_data['U_RE']  # H x W
            mask_PS = eps_data['mask_PS']  # H x W
            mask_RS = eps_data['mask_RS']  # H x W
            mask_RE = eps_data['mask_RE']  # H x W
            q_G = eps_data['q_G']  # 2 x H x W

        H, W = M_p.shape[1], M_p.shape[2]
        # there are class 99 in the sem map
        M_p[1] = np.where(M_p[1] >= cfg.SEM_MAP.GRID_CLASS_SIZE, 0, M_p[1])

        # =================================== visualize M_p =========================================
        if cfg.PRED.PARTIAL_MAP.FLAG_VISUALIZE_PRED_LABELS:
            occ_map_Mp = M_p[0]
            sem_map_Mp = M_p[1]
            color_sem_map_Mp = apply_color_to_map(sem_map_Mp)

            fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(40, 20))
            ax[0][0].imshow(occ_map_Mp, cmap='gray')
            ax[0][0].get_xaxis().set_visible(False)
            ax[0][0].get_yaxis().set_visible(False)
            ax[0][0].set_title('input: occupancy_map_Mp')

            ax[1][0].imshow(color_sem_map_Mp)
            ax[1][0].get_xaxis().set_visible(False)
            ax[1][0].get_yaxis().set_visible(False)
            ax[1][0].set_title('input: semantic_map_Mp')

            ax[0][1].imshow(U_PS, vmin=0.0)
            ax[0][1].get_xaxis().set_visible(False)
            ax[0][1].get_yaxis().set_visible(False)
            ax[0][1].set_title('U_PS')

            ax[1][1].imshow(U_RS, vmin=0.0)
            ax[1][1].get_xaxis().set_visible(False)
            ax[1][1].get_yaxis().set_visible(False)
            ax[1][1].set_title('U_RS')

            ax[0][2].imshow(U_RE, vmin=0.0)
            ax[0][2].get_xaxis().set_visible(False)
            ax[0][2].get_yaxis().set_visible(False)
            ax[0][2].set_title('U_RE')

            ax[1][2].imshow(mask_PS, vmin=0.0)
            ax[1][2].get_xaxis().set_visible(False)
            ax[1][2].get_yaxis().set_visible(False)
            ax[1][2].set_title('mask_PS')

            ax[0][3].imshow(mask_RS, vmin=0.0)
            ax[0][3].get_xaxis().set_visible(False)
            ax[0][3].get_yaxis().set_visible(False)
            ax[0][3].set_title('mask_RS')

            ax[1][3].imshow(mask_RE, vmin=0.0)
            ax[1][3].get_xaxis().set_visible(False)
            ax[1][3].get_yaxis().set_visible(False)
            ax[1][3].set_title('mask_RE')

            fig.tight_layout()
            plt.show()

            fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(20, 20))

            ax[0][0].imshow(occ_map_Mp, cmap='gray')
            ax[0][0].get_xaxis().set_visible(False)
            ax[0][0].get_yaxis().set_visible(False)
            ax[0][0].set_title('input: occupancy_map_Mp')

            ax[1][0].imshow(color_sem_map_Mp)
            ax[1][0].get_xaxis().set_visible(False)
            ax[1][0].get_yaxis().set_visible(False)
            ax[1][0].set_title('input: semantic_map_Mp')

            ax[0][1].imshow(q_G[0])
            ax[0][1].get_xaxis().set_visible(False)
            ax[0][1].get_yaxis().set_visible(False)
            ax[0][1].set_title('q_G x-axis')

            ax[1][1].imshow(q_G[1])
            ax[1][1].get_xaxis().set_visible(False)
            ax[1][1].get_yaxis().set_visible(False)
            ax[1][1].set_title('q_G, y-axis')

            fig.tight_layout()
            plt.show()

        # ============ combine the input and output
        U_all = np.stack((U_PS, U_RS, U_RE), axis=0)  # 3 x H x W
        mask_all = np.stack((mask_PS, mask_RS, mask_RE), axis=0)  # 3 x H x W
        # rescale the q_G
        q_G = q_G.astype(np.float32)
        q_G[0, :, :] *= 1. / (cfg.PRED.PARTIAL_MAP.INPUT_WH[0] / 2)
        q_G[1, :, :] *= 1. / (cfg.PRED.PARTIAL_MAP.INPUT_WH[1] / 2)

        # ================= convert to tensor=================
        tensor_Mp = torch.tensor(M_p, dtype=torch.long)
        tensor_U = torch.tensor(U_all, dtype=torch.float32)
        tensor_qG = torch.tensor(q_G, dtype=torch.float32)
        tensor_mask = torch.tensor(mask_all, dtype=torch.bool)

        # print(f'tensor_Mp.max = {torch.max(tensor_Mp)}')
        # ================= convert input tensor into one-hot vector===========================
        tensor_Mp_occ = tensor_Mp[0]  # H x W
        tensor_Mp_occ = F.one_hot(tensor_Mp_occ, num_classes=3).permute(2, 0, 1)  # 3 x H x W
        tensor_Mp_sem = tensor_Mp[1]
        tensor_Mp_sem = F.one_hot(tensor_Mp_sem, num_classes=cfg.SEM_MAP.GRID_CLASS_SIZE).permute(
            2, 0, 1)  # num_classes x H x W

        tensor_Mp = torch.cat((tensor_Mp_occ, tensor_Mp_sem), 0).float()
        tensor_input = torch.cat((tensor_Mp, tensor_qG), 0)

        if cfg.PRED.PARTIAL_MAP.INPUT == 'occ_only':
            tensor_Mp = tensor_Mp[0:3]
            tensor_input = torch.cat((tensor_Mp, tensor_qG), 0)

        return {'input': tensor_input, 'output': tensor_U, 'mask': tensor_mask, 'shape': (H, W)}


def get_all_scene_dataset(split, scene_list, data_folder):
    ds_list = []
    for scene in scene_list:
        scene_ds = MP3DSceneDataset(split, scene, data_folder=data_folder)
        ds_list.append(scene_ds)

    concat_ds = data.ConcatDataset(ds_list)
    return concat_ds


def my_collate(batch):
    output_dict = {}
    # ==================================== for input ==================================
    out = None
    batch_input = [dict['input'] for dict in batch]
    output_dict['input'] = torch.stack(batch_input, 0)

    # ==================================== for output ==================================
    out = None
    batch_output = [dict['output'] for dict in batch]
    output_dict['output'] = torch.stack(batch_output, 0)

    out = None
    batch_mask = [dict['mask'] for dict in batch]
    output_dict['mask'] = torch.stack(batch_mask, 0)

    batch_shape = [dict['shape'] for dict in batch]
    output_dict['shape'] = batch_shape

    return output_dict


if __name__ == "__main__":
    cfg.merge_from_file('configs/exp_train_input_partial_map_occ_and_sem_for_pointgoal.yaml')
    cfg.freeze()

    split = 'train'
    if split == 'train':
        scene_list = cfg.MAIN.TRAIN_SCENE_LIST
    elif split == 'val':
        scene_list = cfg.MAIN.VAL_SCENE_LIST
    elif split == 'test':
        scene_list = cfg.MAIN.TEST_SCENE_LIST

    data_folder = cfg.PRED.PARTIAL_MAP.GEN_SAMPLES_SAVED_FOLDER

    ds_list = []
    for scene in scene_list:
        scene_ds = MP3DSceneDataset(split, scene, data_folder=data_folder)
        ds_list.append(scene_ds)

    concat_ds = data.ConcatDataset(ds_list)
