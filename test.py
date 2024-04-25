import torch
import matplotlib.pyplot as plt
import os
from model import NeuralSuperSampling
from dataset import QRISPDataset
from loss import WeightedSSIMPerceptualLoss
import kornia as K


os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

if __name__ == "__main__":
    data_path = "data/"
    dataset = QRISPDataset(data_path, split="train", sequence_length=5)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    model = NeuralSuperSampling(2, 5, 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = WeightedSSIMPerceptualLoss(0.1)

    for i, data in enumerate(dataloader):
        (color_frames, motion_frames, depth_frames, trgt_frames) = data

        for _ in range(100):
            outputs = model(color_frames, motion_frames, depth_frames)

            trgt_frames = K.color.rgb_to_ycbcr(trgt_frames)

            optimizer.zero_grad()
            loss = loss_fn(trgt_frames, outputs)
            loss.backward()
            optimizer.step()
            print("Loss:", loss.item())

        with torch.no_grad():
            outputs = model(color_frames, motion_frames, depth_frames)
            outputs = K.color.ycbcr_to_rgb(outputs)

            outputs = outputs.squeeze(0).permute(1, 2, 0).cpu().numpy()

            plt.imshow(outputs)
            plt.show()
        break
