import torch
import matplotlib.pyplot as plt
import os
from model import NeuralSuperSampling
from dataset import QRISPDataset
from loss import WeightedSSIMPerceptualLoss

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

if __name__ == "__main__":
    data_path = "data/"
    dataset = QRISPDataset(data_path, split="train", sequence_length=5)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    model = NeuralSuperSampling(2, 5, 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss = WeightedSSIMPerceptualLoss(0.1)

    for i, data in enumerate(dataloader):
        torch.autograd.set_detect_anomaly(True)
        (color_frames, motion_frames, depth_frames, trgt_frames) = data
        outputs = model(color_frames, motion_frames, depth_frames)

        optimizer.zero_grad()
        loss = loss(trgt_frames, outputs)
        loss.backward()
        optimizer.step()

        print("Loss:", loss.item())

        break
