import os
import random

import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import Imath
import numpy as np
import OpenEXR

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
    vertical_velocity = vertical_velocity.reshape(size[1], size[0])
    horizontal_velocity = horizontal_velocity.reshape(size[1], size[0])
    exr_file.close()
    return vertical_velocity, horizontal_velocity


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
        lr_max_x = img_shape[1] - lr_patch_size
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


if __name__ == "__main__":
    dataset = NSSDataset("data", split="test")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)

    for i, data in enumerate(dataloader):
        color, motion, depth, yhat = data

        color = color.squeeze(0).cpu().numpy()
        yhat = yhat.squeeze(0).permute(1, 2, 0).cpu().numpy()

        fig, axes = plt.subplots(1, 5, figsize=(20, 4))

        for j in range(5):
            axes[j].imshow(color[j].transpose(1, 2, 0))
            axes[j].set_title(f"Color Frame {j+1}")
            axes[j].axis("off")

        plt.figure(figsize=(8, 4))
        plt.imshow(yhat)
        plt.title("Yhat Image")
        plt.axis("off")

        plt.tight_layout()
        plt.show()

        break
