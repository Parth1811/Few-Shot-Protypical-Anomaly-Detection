import glob
import os
from collections import OrderedDict

import cv2
import numpy as np
import torch.utils.data as data
rng = np.random.RandomState(2020)


def np_load_frame(filename, resize_height, resize_width):
    """
    Load image path and convert it to numpy.ndarray. Notes that the color channels are BGR and the color space
    is normalized from [0, 255] to [-1, 1].

    :param filename: the full path of image
    :param resize_height: resized height
    :param resize_width: resized width
    :return: numpy.ndarray
    """
    image_decoded = cv2.imread(filename)
    image_resized = cv2.resize(image_decoded, (resize_width, resize_height))
    image_resized = image_resized.astype(dtype=np.float32)
    image_resized = (image_resized / 127.5) - 1.0
    return image_resized


class DataLoader(data.Dataset):
    def __init__(self, video_folder, transform, resize_height, resize_width, time_step=4, num_pred=1):
        self.dir = video_folder
        self.transform = transform
        self.videos = OrderedDict()
        self._resize_height = resize_height
        self._resize_width = resize_width
        self._time_step = time_step
        self._num_pred = num_pred
        self.setup()
        self.samples = self.get_all_samples()

    def setup(self):
        videos = glob.glob(os.path.join(self.dir, '*'))
        for video in sorted(videos):
            video_name = video.split('/')[-1]
            self.videos[video_name] = {}
            self.videos[video_name]['path'] = video
            self.videos[video_name]['frame'] = glob.glob(os.path.join(video, '*.jpg'))
            self.videos[video_name]['frame'].sort()
            self.videos[video_name]['length'] = len(self.videos[video_name]['frame'])

    def get_all_samples(self):
        frames = []
        videos = glob.glob(os.path.join(self.dir, '*'))
        for video in sorted(videos):
            video_name = video.split('/')[-1]
            for i in range(len(self.videos[video_name]['frame']) - self._time_step):
                frames.append(self.videos[video_name]['frame'][i])

        return frames

    def __getitem__(self, index):
        # video_name = self.samples[index].split('/')[-2]
        frame_name = int(self.samples[index].split('/')[-1].split('.')[-2])

        batch = []
        for i in range(self._time_step + self._num_pred):
            try:
                image = np_load_frame(self.samples[index][:-8] + '%04d.jpg' % (frame_name + i), self._resize_height, self._resize_width)
                if self.transform is not None:
                    batch.append(self.transform(image))
            except IndexError:
                print(self.samples[index][:-8] + '%04d.jpg' % (frame_name + i))

        return np.concatenate(batch, axis=0)

    def __len__(self):
        return len(self.samples)


class SceneLoader:

    def __init__(self, scenes_folder, transform, resize_height, resize_width, k_shots=4, time_step=4, num_pred=1):
        self.scene_paths = glob.glob(os.path.join(scenes_folder, '*'))
        self.scenes_dataloader = {}
        self.dataloader_iters = {}
        self.scenes = []
        for scene_path in self.scene_paths:
            scene = scene_path.split('/')[-1]
            self.scenes.append(scene)
            dataset = DataLoader(scene_path, transform, resize_height=resize_height,
                                 resize_width=resize_width, time_step=time_step)
            # train_set, val_set = torch.utils.data.random_split(dataset, [len(dataset)//2, len(dataset) - len(dataset)//2])
            dl_train = data.DataLoader(dataset, batch_size=k_shots, shuffle=True,
                                       num_workers=2, drop_last=True)
            # dl_val = data.DataLoader(val_set, batch_size = k_shots, shuffle=True,
            #                          num_workers=2, drop_last=True)

            self.scenes_dataloader[scene] = dl_train
            self.dataloader_iters[scene] = (scene, iter(dl_train))

    def get_dataloaders_of_N_random_scenes(self, N):
        samples = np.random.choice(self.scenes, N)
        dataloaders = []
        for scene in samples:
            dataloaders.append(self.dataloader_iters[scene])

        return dataloaders
