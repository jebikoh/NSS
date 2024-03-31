import os
import random

import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import Imath
import numpy as np
import OpenEXR
from util import read_motion, read_depth, get_random_patch_coordinates, extract_patch


NUM_SEQUENCES = 41
NUM_FRAMES = 30


def read_exr_velocities_16bit(file_path):
    exr_file = OpenEXR.InputFile(file_path)
    header = exr_file.header()
    dw = header["dataWindow"]
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    pt = Imath.PixelType(Imath.PixelType.HALF)
    vertical_velocity = np.frombuffer(exr_file.channel("R", pt), dtype=np.float16)
    horizontal_velocity = np.frombuffer(exr_file.channel("G", pt), dtype=np.float16)
    vertical_velocity = vertical_velocity.reshape(size[0], size[0])
    horizontal_velocity = horizontal_velocity.reshape(size[0], size[0])
    exr_file.close()
    return vertical_velocity, horizontal_velocity


class QRISPDataset(Dataset):
    def __init__(
        self,
        data_dir,
        split="train",
        split_ratio=(0.8, 0.1, 0.1),
        sequence_length=5,
        base_resolution=(270, 480),
        trgt_resolution=(540, 960),
        base_crop_size=128,
        trgt_crop_size=256,
    ):
        self.data_dir = data_dir
        self.split = split
        self.split_ratio = split_ratio
        self.sequence_length = sequence_length
        self.base_resolution = base_resolution
        self.trgt_resolution = trgt_resolution
        self.base_crop_size = base_crop_size
        self.trgt_crop_size = trgt_crop_size
        self.scale_factor = trgt_resolution[0] // base_resolution[0]

        self.sequences = sorted(
            os.listdir(os.path.join(data_dir, f"{base_resolution[0]}p", "color"))
        )
        if ".DS_Store" in self.sequences:
            self.sequences.remove(".DS_Store")

        train_ratio, val_ratio, _ = split_ratio

        if split == "train":
            self.sequences = self.sequences[: int(len(self.sequences) * train_ratio)]
        elif split == "val":
            self.sequences = self.sequences[
                int(len(self.sequences) * train_ratio) : int(
                    len(self.sequences) * (train_ratio + val_ratio)
                )
            ]
        else:
            self.sequences = self.sequences[
                int(len(self.sequences) * (train_ratio + val_ratio)) :
            ]

    def __len__(self):
        return len(self.sequences) * (NUM_FRAMES - self.sequence_length)

    def __getitem__(self, idx):
        seq_idx = idx // (NUM_FRAMES - self.sequence_length)
        frame_idx = idx % (NUM_FRAMES - self.sequence_length)
        sequence = self.sequences[seq_idx]

        color_frames = []
        motion_frames = []
        depth_frames = []

        (base_x, base_y), (trgt_x, trgt_y) = get_random_patch_coordinates(
            self.base_resolution, self.base_crop_size, self.scale_factor
        )

        f0 = frame_idx + self.sequence_length - 1

        for i in range(f0, frame_idx - 1, -1):
            color_path = os.path.join(
                self.data_dir,
                f"{self.base_resolution[0]}p",
                "color",
                sequence,
                f"{i:04d}.png",
            )
            motion_path = os.path.join(
                self.data_dir,
                f"{self.base_resolution[0]}p",
                "motion",
                sequence,
                f"{i:04d}.exr",
            )
            depth_path = os.path.join(
                self.data_dir,
                f"{self.base_resolution[0]}p",
                "depth",
                sequence,
                f"{i:04d}.png",
            )

            color_img = np.array(Image.open(color_path).convert("RGB")) / 255.0
            color_img = extract_patch(color_img, base_x, base_y, self.base_crop_size)
            color_frames.append(color_img)

            motion_img = read_motion(motion_path)
            motion_img = extract_patch(motion_img, base_x, base_y, self.base_crop_size)
            motion_frames.append(motion_img)

            depth_img = read_depth(depth_path)
            depth_img = extract_patch(depth_img, base_x, base_y, self.base_crop_size)
            depth_frames.append(depth_img)

        trgt_path = os.path.join(
            self.data_dir,
            f"{self.trgt_resolution[0]}p",
            "color",
            sequence,
            f"{f0:04d}.png",
        )

        trgt_img = np.array(Image.open(trgt_path).convert("RGB")) / 255.0
        trgt_img = extract_patch(trgt_img, trgt_x, trgt_y, self.trgt_crop_size)

        color_frames = np.stack(color_frames, axis=0)
        motion_frames = np.stack(motion_frames, axis=0)
        depth_frames = np.stack(depth_frames, axis=0)

        color_frames = torch.from_numpy(color_frames).permute(0, 3, 1, 2).float()
        motion_frames = torch.from_numpy(motion_frames).permute(0, 3, 1, 2).float()
        depth_frames = torch.from_numpy(depth_frames).unsqueeze(1).float()
        trgt_frames = torch.from_numpy(trgt_img).permute(2, 0, 1).float()

        return (
            color_frames,
            motion_frames,
            depth_frames,
            trgt_frames,
            sequence,
            frame_idx,
            f0,
        )


class NSSDataset(Dataset):
    def __init__(
        self, data_dir, split="train", split_ratio=(0.8, 0.1, 0.1), sequence_length=5
    ):
        self.data_dir = data_dir
        self.split = split
        self.sequence_length = sequence_length
        self.sequences = sorted(os.listdir(os.path.join(data_dir, "270p", "color")))

        # Uncomment this if running on mac
        if ".DS_Store" in self.sequences:
            self.sequences.remove(".DS_Store")

        assert len(self.sequences) == NUM_SEQUENCES

        train_ratio, val_ratio, _ = split_ratio

        if split == "train":
            self.sequences = self.sequences[: int(len(self.sequences) * train_ratio)]
        elif split == "val":
            self.sequences = self.sequences[
                int(len(self.sequences) * train_ratio) : int(
                    len(self.sequences) * (train_ratio + val_ratio)
                )
            ]
        else:
            self.sequences = self.sequences[
                int(len(self.sequences) * (train_ratio + val_ratio)) :
            ]

    def __len__(self):
        return len(self.sequences) * NUM_FRAMES

    def get_random_patch_coordinates(self, img_shape, lr_patch_size=128, hr_scale=2):
        lr_max_x = img_shape[0] - lr_patch_size
        lr_max_y = img_shape[0] - lr_patch_size
        lr_x = random.randint(0, lr_max_x)
        lr_y = random.randint(0, lr_max_y)
        hr_x = lr_x * hr_scale
        hr_y = lr_y * hr_scale
        return (lr_x, lr_y), (hr_x, hr_y)

    def extract_patch(self, img, x, y, size):
        return img[y : y + size, x : x + size]

    def __getitem__(self, idx):
        seq_idx = random.randint(0, len(self.sequences) - 1)
        frame_idx = random.randint(4, NUM_FRAMES - 1)
        sequence = self.sequences[seq_idx]

        color_frames = []
        motion_frames = []
        depth_frames = []

        img_shape = None
        (lr_x, lr_y), (hr_x, hr_y) = (None, None), (None, None)

        for i in range(frame_idx, frame_idx - 5, -1):
            color_path = os.path.join(
                self.data_dir, "270p", "color", sequence, f"{i:04d}.png"
            )
            motion_path = os.path.join(
                self.data_dir, "270p", "motion", sequence, f"{i:04d}.exr"
            )
            depth_path = os.path.join(
                self.data_dir, "270p", "depth", sequence, f"{i:04d}.png"
            )

            color_img = np.array(Image.open(color_path).convert("RGB")) / 255.0

            if img_shape is None:
                img_shape = color_img.shape
                (lr_x, lr_y), (hr_x, hr_y) = self.get_random_patch_coordinates(
                    img_shape
                )

            color_frames.append(self.extract_patch(color_img, lr_x, lr_y, 128))

            vert_vel, hor_vel = read_exr_velocities_16bit(motion_path)
            motion_img = np.stack([hor_vel, vert_vel], axis=-1)
            motion_frames.append(self.extract_patch(motion_img, lr_x, lr_y, 128))

            depth_img = np.array(Image.open(depth_path))
            depth = (
                depth_img[:, :, 0] / 255.0
                + depth_img[:, :, 1] / 255.0**2
                + depth_img[:, :, 2] / 255.0**3
                + depth_img[:, :, 3] / 255.0**4
            )
            depth_frames.append(
                self.extract_patch(depth[:, :, np.newaxis], lr_x, lr_y, 128)
            )
        yhat_img = Image.open(
            os.path.join(
                self.data_dir, "540p", "color", sequence, f"{frame_idx:04d}.png"
            )
        ).convert("RGB")
        yhat = np.array(yhat_img) / 255.0
        yhat = self.extract_patch(yhat, hr_x, hr_y, 256)

        color_frames = np.stack(color_frames, axis=0)
        motion_frames = np.stack(motion_frames, axis=0)
        depth_frames = np.stack(depth_frames, axis=0)

        color_frames = torch.from_numpy(color_frames).permute(0, 3, 1, 2).float()
        motion_frames = torch.from_numpy(motion_frames).permute(0, 3, 1, 2).float()
        depth_frames = torch.from_numpy(depth_frames).permute(0, 3, 1, 2).float()
        yhat = torch.from_numpy(yhat).permute(2, 0, 1).float()

        return color_frames, motion_frames, depth_frames, yhat, sequence, frame_idx
