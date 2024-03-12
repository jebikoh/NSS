import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import NSSDataset
from model import NeuralSuperSamplingNetwork
from loss import NssLoss
from tqdm import tqdm

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model = NeuralSuperSamplingNetwork((480, 270), (960, 540)).to(device)
    criterion = NssLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Create the dataset and dataloader
    dataset = NSSDataset("data")
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)

    # Set the number of epochs
    num_epochs = 10

    # Training loop
    for epoch in range(num_epochs):
        running_loss = 0.0
        progress_bar = tqdm(
            dataloader, desc=f"Epoch [{epoch+1}/{num_epochs}]", unit="batch"
        )

        for i, data in enumerate(progress_bar):
            color, motion, depth, target = data
            color = color.to(device)
            motion = motion.to(device)
            depth = depth.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            output = model(color, motion, depth)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    torch.save(model.state_dict(), "model_weights.pth")
    print("Model weights saved as 'model_weights.pth'")
    print("Done")
