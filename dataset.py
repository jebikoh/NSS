import os
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from util import read_motion, read_depth, get_random_patch_coordinates, extract_patch


class QRISPDataset(Dataset):
    def __init__(
        self,
        data_dir,
        num_frames=30,
        num_sequences=41,
        split="train",
        split_ratio=(0.8, 0.1, 0.1),
        sequence_length=5,
        base_resolution=(270, 480),
        trgt_resolution=(540, 960),
        base_crop_size=128,
        trgt_crop_size=256,
    ):
        self.num_frames = num_frames
        self.num_sequences = num_sequences
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
        return len(self.sequences) * (self.num_frames - self.sequence_length)

    def __getitem__(self, idx):
        seq_idx = idx // (self.num_frames - self.sequence_length)
        frame_idx = idx % (self.num_frames - self.sequence_length)
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
        )


if __name__ == "__main__":
    data_path = "data/"
    dataset = QRISPDataset(data_path, split="train", sequence_length=5)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

    for i, data in enumerate(dataloader):
        (color_frames, motion_frames, depth_frames, trgt_frames) = data

        fig, (axs1, axs2) = plt.subplots(2, 5, figsize=(12, 6))

        # Display color frame 0
        axs1[0].imshow(color_frames[0, 1].permute(1, 2, 0))
        axs1[0].set_title("Color Frame 0")
        axs1[0].axis("off")

        # Display motion frame 0
        axs1[1].imshow(motion_frames[0, 1, 0].unsqueeze(-1), cmap="viridis")
        axs1[1].set_title("Horizontal Motion Frame -1")
        axs1[1].axis("off")

        axs1[2].imshow(motion_frames[0, 1, 1].unsqueeze(-1), cmap="viridis")
        axs1[2].set_title("Vertical Motion Frame -1")
        axs1[2].axis("off")

        # Display depth frame 0
        axs1[3].imshow(depth_frames[0, 0, 0], cmap="gray")
        axs1[3].set_title("Depth Frame 0")
        axs1[3].axis("off")

        # Display target frame
        axs1[4].imshow(trgt_frames[0].permute(1, 2, 0))
        axs1[4].set_title("Target Frame")
        axs1[4].axis("off")

        axs2[0].imshow(color_frames[0, 0].permute(1, 2, 0))
        axs2[0].set_title("Color Frame 0")
        axs2[1].imshow(color_frames[0, 1].permute(1, 2, 0))
        axs2[1].set_title("Color Frame -1")
        axs2[2].imshow(color_frames[0, 2].permute(1, 2, 0))
        axs2[2].set_title("Color Frame -2")
        axs2[3].imshow(color_frames[0, 3].permute(1, 2, 0))
        axs2[3].set_title("Color Frame -3")
        axs2[4].imshow(color_frames[0, 4].permute(1, 2, 0))
        axs2[4].set_title("Color Frame -4")

        plt.tight_layout()
        plt.show()
