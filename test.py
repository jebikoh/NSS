import torch
import matplotlib.pyplot as plt
from dataset import NSSDataset
from model import NeuralSuperSamplingNetwork


if __name__ == "__main__":
    test_data = NSSDataset("data", split="test")
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True)

    model = NeuralSuperSamplingNetwork((480, 270), (960, 540))
    checkpoint = torch.load(
        "weights/best_model_weights.pth", map_location=torch.device("cpu")
    )
    model.load_state_dict(checkpoint)
    model.eval()

    for i, data in enumerate(test_loader):
        color, motion, depth, yhat = data
        yhat_pred = model(color, motion, depth)

        print(torch.max(yhat_pred), torch.min(yhat_pred))

        color = color.squeeze(0)[0]
        yhat_pred = yhat_pred.squeeze(0)
        yhat = yhat.squeeze(0)

        input = color.permute(1, 2, 0)
        yhat_pred = yhat_pred.permute(1, 2, 0)
        yhat = yhat.permute(1, 2, 0)

        input = input.detach().cpu().numpy()
        yhat_pred = yhat_pred.detach().cpu().numpy()
        yhat = yhat.detach().cpu().numpy()

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

        ax1.imshow(input)
        ax1.axis("off")
        ax1.set_title("Input 270p")

        ax2.imshow(yhat)
        ax2.axis("off")
        ax2.set_title("Rendered 540p")

        ax3.imshow(yhat_pred)
        ax3.axis("off")
        ax3.set_title("NSS 540p")

        plt.tight_layout()
        plt.show()
