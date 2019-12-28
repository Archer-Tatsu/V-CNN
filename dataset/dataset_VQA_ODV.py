import os
import logging
import numpy as np
import torch.utils.data
import scipy.ndimage.interpolation as interp
import skimage.transform
import warnings
from utils import yuv


class DownSample:
    def __init__(self, down_resolution):
        self.down_resolution = down_resolution

    def __call__(self, Y, U, V):
        half_resolution = [i / 2 for i in self.down_resolution]
        Y_d = skimage.transform.resize(Y.transpose((1, 2, 0)), self.down_resolution, order=1, anti_aliasing=True,
                                       mode='reflect', preserve_range=True)
        U_d = skimage.transform.resize(U.transpose((1, 2, 0)), half_resolution, order=1, anti_aliasing=True,
                                       mode='reflect', preserve_range=True)
        V_d = skimage.transform.resize(V.transpose((1, 2, 0)), half_resolution, order=1, anti_aliasing=True,
                                       mode='reflect', preserve_range=True)

        return Y_d.transpose((2, 0, 1)).round().astype(np.uint8), U_d.transpose((2, 0, 1)).round().astype(np.uint8), \
               V_d.transpose((2, 0, 1)).round().astype(np.uint8)

    def __repr__(self):
        return self.__class__.__name__ + '(down_resolution={0})'.format(self.down_resolution)


class SampleSGrid:
    def __init__(self, bandwidth):
        self.bandwidth = bandwidth
        self.euler_grid, _ = self.make_sgrid(bandwidth)

    def __call__(self, Y, U, V, resolution):
        height, width = resolution
        height_half, width_half = height // 2, width // 2
        theta, phi = self.euler_grid

        pix_height = theta[:, 0] / np.pi * height
        pix_width = phi[0, :] / (np.pi * 2) * width
        pix_width, pix_height = np.meshgrid(pix_width, pix_height)
        pix_height_half = theta[:, 0] / np.pi * height_half
        pix_width_half = phi[0, :] / (np.pi * 2) * width_half
        pix_width_half, pix_height_half = np.meshgrid(pix_width_half, pix_height_half)

        Y_im = interp.map_coordinates(Y[0, ...], [pix_height, pix_width], order=1)
        U_im = interp.map_coordinates(U[0, ...], [pix_height_half, pix_width_half], order=1)
        V_im = interp.map_coordinates(V[0, ...], [pix_height_half, pix_width_half], order=1)

        return Y_im[np.newaxis, ...], U_im[np.newaxis, ...], V_im[np.newaxis, ...]

    @staticmethod
    def make_sgrid(b):
        from lie_learn.spaces import S2

        theta, phi = S2.meshgrid(b=b, grid_type='SOFT')
        sgrid = S2.change_coordinates(np.c_[theta[..., None], phi[..., None]], p_from='S', p_to='C')
        sgrid = sgrid.reshape((-1, 3))

        return (theta, phi), sgrid

    def __repr__(self):
        return self.__class__.__name__ + '(bandwidth={0})'.format(self.bandwidth)


class VQA_ODV_Transform:
    def __init__(self, bandwidth, down_resolution, to_rgb=True):
        self.to_rgb = to_rgb
        self.sgrid_transform = SampleSGrid(bandwidth)
        self.down_transform = DownSample(down_resolution)

    def __call__(self, file_path, resolution, frame_index):
        ori = yuv.yuv_import(file_path, resolution, 1, frame_index)
        im = self.sgrid_transform(*ori, resolution)
        down = self.down_transform(*ori)
        if self.to_rgb:
            im = yuv.yuv2rgb(*im)
            ori = yuv.yuv2rgb(*ori)
            down = yuv.yuv2rgb(*down)
        return im, ori, down


class DS_VQA_ODV(torch.utils.data.Dataset):

    def __init__(self, root, dataset_type, ds_list_file, transform, tr_te_file=None, flow_gap=2, test_start_frame=21,
                 test_interval=45):
        """
        VQA-ODV initialization.
        :param root: Directory of the dataset information files.
        :param dataset_type: Training set or test set. NOTE: ONLY TEST SET IS SUPPORTED AT PRESENT.
        :param ds_list_file: Name of the list file for all impaired sequences.
        :param transform: The class for transformation. Should be an instance of VQA_ODV_Transform.
        :param tr_te_file: Name of the file for splitting training and test scenes.
        :param flow_gap: Frame gap for optical flow extraction.
        :param test_start_frame: The start frame for each sequence in test (for dropping frames and saving time).
        :param test_interval: The interval for each sequence in test (for dropping frames and saving time).
        """

        self.type = dataset_type
        if self.type not in ("train", "test"):
            raise ValueError("Invalid dataset")

        self.logger = logging.getLogger("{}.dataset".format(self.type))
        self.root = os.path.expanduser(root)
        self.ds_list_file = ds_list_file

        assert isinstance(transform, VQA_ODV_Transform)
        self.transform = transform
        self.flow_gap = flow_gap
        self.test_start_frame = test_start_frame
        self.test_interval = test_interval
        if self.test_start_frame < self.flow_gap:
            warnings.warn(
                "The value of test_start_frame should not be less than flow_gap. Set test_start_frame equal to flow_gap.",
                Warning)
            self.test_start_frame = self.flow_gap
        if self.test_interval < 1:
            warnings.warn(
                "The value of test_interval should not be less than 1. Set test_interval equal to 1.",
                Warning)
            self.test_start_frame = 1

        self.scenes = list(range(60))
        self.train_scenes, self.test_scenes = self.divide_tr_te_wrt_ref(self.scenes,
                                                                        tr_te_file=os.path.join(self.root, tr_te_file))

        if self.type == 'train':
            self.data_dict = self.make_video_list(self.train_scenes)
        else:
            self.data_dict = self.make_video_list(self.test_scenes)
            self.frame_num_list = np.array(self.data_dict['frame_num_list'], dtype=np.float32)
            self.frame_num_list = np.ceil((self.frame_num_list - self.test_start_frame) / self.test_interval).astype(
                np.int)
            self.cum_frame_num = np.cumsum(self.frame_num_list)
            self.cum_frame_num_prev = np.zeros_like(self.cum_frame_num)
            self.cum_frame_num_prev[1:] = self.cum_frame_num[:-1]
            self.scores = self.data_dict['score_list']

        self.files = self.data_dict['distort_yuv_list']
        self.ref_files = self.data_dict['ref_yuv_list']

    def __getitem__(self, index):
        if self.type == 'train':
            raise NotImplementedError
        else:
            video_index = np.searchsorted(self.cum_frame_num_prev, index, side='right') - 1
            frame_index = (index - self.cum_frame_num_prev[video_index]) * self.test_interval + self.test_start_frame

        video_path = self.files[video_index]
        ref_path = self.ref_files[video_index]
        resolution = self.data_dict['resolution_list'][video_index]

        img_gap, _, gap_down = self.transform(file_path=video_path, resolution=resolution,
                                              frame_index=frame_index - self.flow_gap)
        img, img_original, img_down = self.transform(file_path=video_path, resolution=resolution,
                                                     frame_index=frame_index)
        _, ref, _ = self.transform(file_path=ref_path, resolution=resolution, frame_index=frame_index)

        target = np.array([self.scores[video_index]])
        self.logger.debug('[DATA] {}, REF:{}, FRAME:{}, SCORE:{}'.format(video_path, ref_path, frame_index, target))

        return img.astype(np.float32), img_original.astype(np.float32), img_down.astype(np.float32), \
               img_gap.astype(np.float32), gap_down.astype(np.float32), ref.astype(np.float32), \
               target.astype(np.float32)

    def __len__(self):
        if self.type == 'train':
            return len(self.files)
        else:
            return self.cum_frame_num[-1]

    def divide_tr_te_wrt_ref(self, scenes, train_size=0.8, tr_te_file=None):
        """
        Divide data with respect to scenes.
        """
        tr_te_file_loaded = False
        if tr_te_file is not None and os.path.isfile(tr_te_file):
            # Load tr_te_file and divide scenes accordingly
            tr_te_file_loaded = True
            with open(tr_te_file, 'r') as f:
                train_scenes = f.readline().strip().split()
                train_scenes = [int(elem) for elem in train_scenes]
                test_scenes = f.readline().strip().split()
                test_scenes = [int(elem) for elem in test_scenes]

            n_train_refs = len(train_scenes)
            n_test_refs = len(test_scenes)
            train_size = (len(train_scenes) /
                          (len(train_scenes) + len(test_scenes)))
        else:
            # Divide scenes randomly
            # Get the numbers of training and testing scenes
            n_scenes = len(scenes)
            n_train_refs = int(np.ceil(n_scenes * train_size))
            n_test_refs = n_scenes - n_train_refs

            # Randomly divide scenes
            rand_seq = np.random.permutation(n_scenes)
            scenes_sh = [scenes[elem] for elem in rand_seq]
            train_scenes = sorted(scenes_sh[:n_train_refs])
            test_scenes = sorted(scenes_sh[n_train_refs:])

            # Write train-test idx list into file
            if tr_te_file is not None:
                fpath, fname = os.path.split(tr_te_file)
                if not os.path.isdir(fpath):
                    os.makedirs(fpath)
                with open(tr_te_file, 'w') as f:
                    for idx in range(n_train_refs):
                        f.write('%d ' % train_scenes[idx])
                    f.write('\n')
                    for idx in range(n_scenes - n_train_refs):
                        f.write('%d ' % test_scenes[idx])
                    f.write('\n')

        self.logger.debug(
            ' - Refs.: training = {} / testing = {} (Ratio = {:.2f})'.format(n_train_refs, n_test_refs, train_size,
                                                                             end=''))
        if tr_te_file_loaded:
            self.logger.debug(' (Loaded {})'.format(tr_te_file))
        else:
            self.logger.debug('')

        return train_scenes, test_scenes

    def make_video_list(self, scenes, show_info=True):
        # Get reference / distorted image file lists:
        # distort_yuv_list and score_list
        distort_yuv_list, ref_yuv_list, ref_index_list, score_list, resolution_list, frame_num_list = [], [], [], [], [], []
        with open(os.path.join(self.root, self.ds_list_file), 'r') as listFile:
            for line in listFile:
                # ref_idx ref_path dist_path DMOS width height frame_number
                scn_idx, _, ref, dis, score, width, height, frame_num = line.split()
                scn_idx = int(scn_idx)
                if scn_idx in scenes:
                    distort_yuv_list.append(dis)
                    ref_yuv_list.append(ref)
                    ref_index_list.append(scn_idx)
                    score_list.append(float(score))
                    resolution_list.append((int(height), int(width)))
                    frame_num_list.append(int(frame_num))

        score_list = np.array(score_list, dtype='float32')
        # DMOS -> reverse subjecive scores by default
        score_list = 1.0 - score_list
        n_videos = len(distort_yuv_list)

        if show_info:
            scenes.sort()
            self.logger.debug(' - Scenes: %s'.format(', '.join([str(i) for i in scenes])))
            self.logger.debug(' - Number of videos: {:,}'.format(n_videos))
            self.logger.debug(
                ' - DMOS range: [{:.2f}, {:.2f}]'.format(np.min(score_list), np.max(score_list)))  # , end='')
            self.logger.debug(' (Scale reversed)')

        return {
            'scenes': scenes,
            'n_videos': n_videos,
            'distort_yuv_list': distort_yuv_list,
            'ref_yuv_list': ref_yuv_list,
            'ref_index_list': ref_index_list,
            'score_list': score_list,
            'resolution_list': resolution_list,
            'frame_num_list': frame_num_list
        }
