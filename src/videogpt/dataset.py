import os
import os.path as osp
import math
import random
import warnings
import pickle
from collections import namedtuple

import h5py
import numpy as np
from PIL import Image
from glob import glob

import torch
import torch.nn.functional as F
import torch.utils.data as data
from torchvision.datasets import UCF101
from torchvision.datasets.folder import make_dataset
from torchvision.datasets.video_utils import VideoClips

try:
    from decord import VideoReader, cpu
    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False
    print("Warning: decord not available, falling back to slower methods")

DATA_DIR = os.environ.get("DATA_DIR", "data")


def get_datasets(dataset, **dset_configs):
    name_to_dataset = {
        'bair_pushing': BairPushing,
        'ucf101':  UCF101Wrapper,
        'ucf101_decord': UCF101Decord,
        'uot100': UOT100,
        'druva': DRUVA,
        'uot100_druva': UOT100_DRUVA,
        'all_video': AllVideoDataset,
    }

    # Map dataset names to actual folder names (handles case differences)
    folder_name_map = {
        'uot100': 'UOT100',
        'druva': 'DRUVA',
        'ucf101_decord': 'UCF101',
    }

    Dataset = name_to_dataset[dataset]
    folder_name = folder_name_map.get(dataset, dataset)
    root = osp.join(DATA_DIR, folder_name)
    train_dset = Dataset(root=root, train=True, **dset_configs)
    test_dset = Dataset(root=root, train=False, **dset_configs)
    return train_dset, test_dset


class BairPushing(data.Dataset):
    def __init__(self, root, train, resolution, n_frames):
        super().__init__()
        self.root = root
        self.train = train
        self.resolution = resolution
        self.n_frames = n_frames

        assert resolution == 64, 'BAIR only supports 64 x 64 video'

        fname = 'bair_pushing.hdf5' if train else 'bair_pushing_test.hdf5'
        self.fpath = osp.join(root, fname)

        f = h5py.File(self.fpath, 'r')
        self.size = len(f['frames'])
        f.close()

        self._need_init = True

    @property
    def input_shape(self):
        return (self.n_frames, self.resolution, self.resolution)

    @property
    def n_classes(self):
        raise Exception('BairPushing does not support class conditioning')

    def __len__(self):
        return self.size

    def _init_dset(self):
        f = h5py.File(self.fpath, 'r')
        self.frames = f['frames']
        self._need_init = False

    def __getitem__(self, idx):
        if self._need_init:
            self._init_dset()

        video = self.frames[idx]
        start = np.random.randint(low=0, high=video.shape[1] - self.n_frames + 1)
        video = torch.FloatTensor(video[:, start:start + self.n_frames]) / 255. - 0.5

        return dict(video=video)


class UCF101Wrapper(UCF101):
    def __init__(self, root, train, resolution, n_frames, fold=1):
        video_root = osp.join(root, 'UCF-101')
        super(UCF101, self).__init__(video_root)
        if not 1 <= fold <= 3:
            raise ValueError("fold should be between 1 and 3, got {}".format(fold))

        self.train = train
        self.fold = fold
        self.resolution = resolution
        self.n_frames = n_frames
        self.annotation_path = os.path.join(root, 'ucfTrainTestlist')
        self.classes = list(sorted(p for p in os.listdir(video_root) if osp.isdir(osp.join(video_root, p))))
        class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        self.samples = make_dataset(video_root, class_to_idx, ('avi',), is_valid_file=None)
        video_list = [x[0] for x in self.samples]

        frames_between_clips = 1 if train else 16
        self.video_clips_fname = os.path.join(root, f'ucf_video_clips_{frames_between_clips}_{n_frames}.pkl')
        if not osp.exists(self.video_clips_fname):
            video_clips = VideoClips(
                video_paths=video_list,
                clip_length_in_frames=n_frames,
                frames_between_clips=1,
                num_workers=4
            )
            with open(self.video_clips_fname, 'wb') as f:
                pickle.dump(video_clips, f)
        else:
            with open(self.video_clips_fname, 'rb') as f:
                video_clips = pickle.load(f)
        indices = self._select_fold(video_list, self.annotation_path,
                                    fold, train)
        self.size = video_clips.subset(indices).num_clips()
        self._need_init = True

    @property
    def input_shape(self):
        return (self.n_frames, self.resolution, self.resolution)

    @property
    def n_classes(self):
        return len(self.classes)

    def __len__(self):
        return self.size

    def _init_dset(self):
        with open(self.video_clips_fname, 'rb') as f:
            video_clips = pickle.load(f)
        video_list = [x[0] for x in self.samples]
        self.video_clips_metadata = video_clips.metadata
        self.indices = self._select_fold(video_list, self.annotation_path,
                                         self.fold, self.train)
        self.video_clips = video_clips.subset(self.indices)

        self._need_init = False
        # filter out the pts warnings
        warnings.filterwarnings('ignore')

    def _preprocess(self, video):
        video = resize_crop(video, self.resolution,
                            'random' if self.train else 'center')

        if self.train and random.random() < 0.5:
            video = torch.flip(video, [3])

        video = video.float() / 255
        video = video - 0.5
        return video

    def __getitem__(self, idx):
        if self._need_init:
            self._init_dset()

        video, audio, info, video_idx = self.video_clips.get_clip(idx)
        label = self.samples[self.indices[video_idx]][1]
        video = self._preprocess(video)
        one_hot = torch.zeros(self.n_classes, dtype=torch.float32)
        one_hot[label] = 1.

        return dict(video=video, label=one_hot)


def resize_crop(video, resolution, crop_mode):
    """ Resizes video with smallest axis to `resolution * extra_scale`
        and then crops a `resolution` x `resolution` bock. If `crop_mode == "center"`
        do a center crop, if `crop_mode == "random"`, does a random crop

    Args
        video: a tensor of shape [t, h, w, c] in {0, ..., 255}
        resolution: an int
        crop_mode: 'center', 'random'

    Returns
        a processed video of shape [c, t, h, w]
    """
    # [t, h, w, c] -> [t, c, h, w]
    video = video.permute(0, 3, 1, 2).float()
    _, _, h, w = video.shape

    scale = resolution / min(h, w)
    if h < w:
        target_size = (resolution, math.ceil(w * scale))
    else:
        target_size = (math.ceil(h * scale), resolution)
    video = F.interpolate(video, size=target_size, mode='bilinear')
    t, _, h, w = video.shape

    if crop_mode == 'center':
        w_start = (w - resolution) // 2
        h_start = (h - resolution) // 2
        video = video[:, :, h_start:h_start + resolution, w_start:w_start + resolution]
    elif crop_mode == 'random':
        if w - resolution + 1 <= 0 or h - resolution + 1 <= 0:
            print(video.shape)
        w_start = np.random.randint(low=0, high=w - resolution + 1)
        h_start = np.random.randint(low=0, high=h - resolution + 1)
        video = video[:, :, h_start:h_start + resolution, w_start:w_start + resolution]
    else:
        raise Exception(f"Invalid crop_mode:", crop_mode)

    # [t, c, h, w] -> [c, t, h, w]
    video = video.permute(1, 0, 2, 3).contiguous()
    return video


class DecordVideoDataset(data.Dataset):
    """Base class for video datasets using decord for fast video loading."""

    def __init__(self, root, train, resolution, n_frames, train_split=0.9):
        super().__init__()
        assert DECORD_AVAILABLE, "decord is required for this dataset"
        self.root = root
        self.train = train
        self.resolution = resolution
        self.n_frames = n_frames
        self.train_split = train_split

        # To be populated by subclasses
        self.videos = []  # List of (video_path, n_frames_in_video)
        self.clips = []   # List of (video_idx, start_frame)

    def _build_clips(self, stride=None):
        """Build clip index from video list. stride=None means use n_frames for test, 1 for train."""
        self.clips = []
        if stride is None:
            stride = 1 if self.train else self.n_frames

        for vid_idx, (video_path, total_frames) in enumerate(self.videos):
            if total_frames < self.n_frames:
                continue
            n_clips = total_frames - self.n_frames + 1
            for start in range(0, n_clips, stride):
                self.clips.append((vid_idx, start))

    @property
    def input_shape(self):
        return (self.n_frames, self.resolution, self.resolution)

    @property
    def n_classes(self):
        raise Exception(f'{self.__class__.__name__} does not support class conditioning')

    def __len__(self):
        return len(self.clips)

    def _load_video_clip(self, video_path, start_frame):
        """Load a clip from video using decord."""
        vr = VideoReader(video_path, ctx=cpu(0))
        frame_indices = list(range(start_frame, start_frame + self.n_frames))
        frames = vr.get_batch(frame_indices).asnumpy()  # [T, H, W, C]
        return frames

    def _preprocess(self, video):
        """Common preprocessing: resize, crop, flip, normalize."""
        video = torch.from_numpy(video)  # [T, H, W, C]
        video = resize_crop(video, self.resolution,
                            'random' if self.train else 'center')

        if self.train and random.random() < 0.5:
            video = torch.flip(video, [3])  # horizontal flip

        video = video.float() / 255.0 - 0.5
        return video

    def __getitem__(self, idx):
        vid_idx, start_frame = self.clips[idx]
        video_path, _ = self.videos[vid_idx]

        video = self._load_video_clip(video_path, start_frame)
        video = self._preprocess(video)
        return dict(video=video)


class UOT100(DecordVideoDataset):
    """UOT100 Underwater Object Tracking dataset using decord.

    Structure: root/SequenceName/SequenceName.mp4
    """
    def __init__(self, root, train, resolution, n_frames, train_split=0.9):
        super().__init__(root, train, resolution, n_frames, train_split)

        # Find all MP4 files in sequence directories
        all_videos = []
        for seq_dir in sorted(glob(osp.join(root, '*'))):
            if osp.isdir(seq_dir):
                # Look for MP4 file with same name as directory
                seq_name = osp.basename(seq_dir)
                mp4_path = osp.join(seq_dir, f'{seq_name}.mp4')
                if osp.exists(mp4_path):
                    try:
                        vr = VideoReader(mp4_path, ctx=cpu(0))
                        n_frames_video = len(vr)
                        if n_frames_video >= n_frames:
                            all_videos.append((mp4_path, n_frames_video))
                    except Exception as e:
                        print(f"Warning: Could not read {mp4_path}: {e}")

        # Split into train/test (by video)
        n_train = int(len(all_videos) * train_split)
        if train:
            self.videos = all_videos[:n_train]
        else:
            self.videos = all_videos[n_train:]

        self._build_clips()


class DRUVA(DecordVideoDataset):
    """DRUVA underwater video dataset using decord.

    Structure: root/*.mp4
    """
    def __init__(self, root, train, resolution, n_frames, train_split=0.9):
        super().__init__(root, train, resolution, n_frames, train_split)

        # Find all MP4 files directly in root
        all_videos = []
        for mp4_path in sorted(glob(osp.join(root, '*.mp4'))):
            try:
                vr = VideoReader(mp4_path, ctx=cpu(0))
                n_frames_video = len(vr)
                if n_frames_video >= n_frames:
                    all_videos.append((mp4_path, n_frames_video))
            except Exception as e:
                print(f"Warning: Could not read {mp4_path}: {e}")

        # Split into train/test
        n_train = int(len(all_videos) * train_split)
        if train:
            self.videos = all_videos[:n_train]
        else:
            self.videos = all_videos[n_train:]

        self._build_clips()


class UCF101Decord(DecordVideoDataset):
    """UCF101 dataset using decord for fast video loading.

    Structure: root/Classes/ClassName/*.avi
    Uses official train/test splits from ucfTrainTestlist.
    """
    def __init__(self, root, train, resolution, n_frames, train_split=0.9, fold=1):
        super().__init__(root, train, resolution, n_frames, train_split)
        self.fold = fold

        classes_dir = osp.join(root, 'Classes')
        annotation_path = osp.join(root, 'ucfTrainTestlist')

        # Get class list
        self.classes = sorted([d for d in os.listdir(classes_dir)
                               if osp.isdir(osp.join(classes_dir, d))])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        # Read train/test split
        split_file = 'trainlist{:02d}.txt' if train else 'testlist{:02d}.txt'
        split_path = osp.join(annotation_path, split_file.format(fold))

        video_list = []
        if osp.exists(split_path):
            with open(split_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    # Format: "ClassName/video.avi" or "ClassName/video.avi label"
                    parts = line.split()
                    rel_path = parts[0]
                    video_path = osp.join(classes_dir, rel_path)
                    if osp.exists(video_path):
                        class_name = rel_path.split('/')[0]
                        video_list.append((video_path, class_name))
        else:
            # Fallback: use all videos with train_split ratio
            all_videos = []
            for class_name in self.classes:
                class_dir = osp.join(classes_dir, class_name)
                for vid_file in glob(osp.join(class_dir, '*.avi')):
                    all_videos.append((vid_file, class_name))

            n_train = int(len(all_videos) * train_split)
            if train:
                video_list = all_videos[:n_train]
            else:
                video_list = all_videos[n_train:]

        # Build video index with frame counts
        self.video_classes = []
        for video_path, class_name in video_list:
            try:
                vr = VideoReader(video_path, ctx=cpu(0))
                n_frames_video = len(vr)
                if n_frames_video >= n_frames:
                    self.videos.append((video_path, n_frames_video))
                    self.video_classes.append(self.class_to_idx[class_name])
            except Exception as e:
                print(f"Warning: Could not read {video_path}: {e}")

        self._build_clips()

    @property
    def n_classes(self):
        return len(self.classes)

    def __getitem__(self, idx):
        vid_idx, start_frame = self.clips[idx]
        video_path, _ = self.videos[vid_idx]
        class_idx = self.video_classes[vid_idx]

        video = self._load_video_clip(video_path, start_frame)
        video = self._preprocess(video)

        # One-hot encode label
        one_hot = torch.zeros(self.n_classes, dtype=torch.float32)
        one_hot[class_idx] = 1.0

        return dict(video=video, label=one_hot)


class UOT100_DRUVA(data.Dataset):
    """Combined UOT100 and DRUVA datasets for joint training."""

    def __init__(self, root, train, resolution, n_frames, train_split=0.9):
        super().__init__()
        self.resolution = resolution
        self.n_frames = n_frames

        # root is ignored, we use paths from DATA_DIR
        uot100_root = osp.join(DATA_DIR, 'UOT100')
        druva_root = osp.join(DATA_DIR, 'DRUVA')

        self.uot100 = UOT100(uot100_root, train, resolution, n_frames, train_split)
        self.druva = DRUVA(druva_root, train, resolution, n_frames, train_split)

        self.uot100_len = len(self.uot100)
        self.druva_len = len(self.druva)

    @property
    def input_shape(self):
        return (self.n_frames, self.resolution, self.resolution)

    @property
    def n_classes(self):
        raise Exception('UOT100_DRUVA does not support class conditioning')

    def __len__(self):
        return self.uot100_len + self.druva_len

    def __getitem__(self, idx):
        if idx < self.uot100_len:
            return self.uot100[idx]
        else:
            return self.druva[idx - self.uot100_len]


class AllVideoDataset(data.Dataset):
    """Combined UOT100, DRUVA, and UCF101 datasets for joint training."""

    def __init__(self, root, train, resolution, n_frames, train_split=0.9):
        super().__init__()
        self.resolution = resolution
        self.n_frames = n_frames

        # root is ignored, we use paths from DATA_DIR
        uot100_root = osp.join(DATA_DIR, 'UOT100')
        druva_root = osp.join(DATA_DIR, 'DRUVA')
        ucf101_root = osp.join(DATA_DIR, 'UCF101')

        self.datasets = []
        self.lengths = []

        # Add UOT100
        try:
            uot100 = UOT100(uot100_root, train, resolution, n_frames, train_split)
            if len(uot100) > 0:
                self.datasets.append(uot100)
                self.lengths.append(len(uot100))
        except Exception as e:
            print(f"Warning: Could not load UOT100: {e}")

        # Add DRUVA
        try:
            druva = DRUVA(druva_root, train, resolution, n_frames, train_split)
            if len(druva) > 0:
                self.datasets.append(druva)
                self.lengths.append(len(druva))
        except Exception as e:
            print(f"Warning: Could not load DRUVA: {e}")

        # Add UCF101
        try:
            ucf101 = UCF101Decord(ucf101_root, train, resolution, n_frames, train_split)
            if len(ucf101) > 0:
                self.datasets.append(ucf101)
                self.lengths.append(len(ucf101))
        except Exception as e:
            print(f"Warning: Could not load UCF101: {e}")

        # Compute cumulative lengths for indexing
        self.cumsum = np.cumsum([0] + self.lengths)

    @property
    def input_shape(self):
        return (self.n_frames, self.resolution, self.resolution)

    @property
    def n_classes(self):
        raise Exception('AllVideoDataset does not support class conditioning')

    def __len__(self):
        return sum(self.lengths)

    def __getitem__(self, idx):
        # Find which dataset this index belongs to
        for i, (start, end) in enumerate(zip(self.cumsum[:-1], self.cumsum[1:])):
            if start <= idx < end:
                local_idx = idx - start
                sample = self.datasets[i][local_idx]
                # Remove label if present (for combined training without class conditioning)
                return dict(video=sample['video'])
