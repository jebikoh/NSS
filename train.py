import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import NSSDataset
from model import NeuralSuperSamplingNetwork
from loss import NssLoss
from tqdm import tqdm
import wandb

DATA_DIR = "data"
NUM_WORKERS = 4

BATCH_SIZE = 2
LR = 1e-4
NUM_EPOCHS = 1

if __name__ == "__main__":
    wandb.init(
        project="NSS",
        config={"batch_size": BATCH_SIZE, "lr": LR, "num_epochs": NUM_EPOCHS},
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_dataset = NSSDataset(DATA_DIR, split="train")
    train_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )
    val_dataset = NSSDataset(DATA_DIR, split="val")
    val_dataloader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )

    model = NeuralSuperSamplingNetwork((480, 270), (960, 540)).to(device)
    criterion = NssLoss()
    criterion.to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_val_loss = float("inf")

    # Training loop
    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        train_bar = tqdm(
            train_dataloader,
            desc=f"Epoch [{epoch+1}/{NUM_EPOCHS}] (Train)",
            unit="batch",
        )

        for i, data in enumerate(train_bar):
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
            train_bar.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(train_dataloader)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Train Loss: {epoch_loss:.4f}")

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            eval_bar = tqdm(
                val_dataloader,
                desc=f"Epoch [{epoch+1}/{NUM_EPOCHS}] (Val)",
                unit="batch",
            )
            for i, data in enumerate(eval_bar):
                color, motion, depth, target = data
                color = color.to(device)
                motion = motion.to(device)
                depth = depth.to(device)
                target = target.to(device)

                output = model(color, motion, depth)
                loss = criterion(output, target)
                val_loss += loss.item()
        val_loss /= len(val_dataloader)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Val Loss: {val_loss:.4f}")

        wandb.log({"train_loss": epoch_loss, "val_loss": val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "weights/best_model_weights.pth")
            print("Best model weights saved as 'best_model_weights.pth'")
    wandb.finish()
    torch.save(model.state_dict(), "weights/final_weights.pth")
    print("Done")
