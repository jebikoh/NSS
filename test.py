import torch
import matplotlib.pyplot as plt
import os
from dataset import QRISPDataset

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

if __name__ == "__main__":
    data_path = "data/"
    dataset = QRISPDataset(data_path, split="train", sequence_length=5)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    for i, data in enumerate(dataloader):
        (
            color_frames,
            motion_frames,
            depth_frames,
            trgt_frames,
            sequence,
            frame_idx,
            f0,
        ) = data

        fig, (axs1, axs2) = plt.subplots(2, 5, figsize=(12, 6))

        fig.suptitle(
            f"Sequence: {sequence[0]}, Frames: {frame_idx.item()} - {f0.item()}"
        )

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
