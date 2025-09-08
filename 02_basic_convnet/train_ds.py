"""Train a basic CNN on simulated MNIST-like data using DeepSpeed."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import deepspeed


class CNNModel(nn.Module):
    """
    A basic convolutional neural network for MNIST-like classification.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for CNN.
        """
        x = self.pool(F.relu(self.conv1(x)))  # [batch, 16, 14, 14]
        x = self.pool(F.relu(self.conv2(x)))  # [batch, 32, 7, 7]
        x = torch.flatten(x, 1)               # [batch, 32*7*7]
        x = F.relu(self.fc1(x))               # [batch, 128]
        x = self.fc2(x)                       # [batch, 10]
        return x


def get_data_loader(batch_size: int) -> DataLoader:
    """
    Generates a random dataset that simulates MNIST:
    - 28x28 grayscale images (1 channel)
    - Integer labels 0–9
    """
    num_samples = 10000  # Simulated training size
    x_data = torch.randn(num_samples, 1, 28, 28)
    y_data = torch.randint(0, 10, (num_samples,))
    dataset = TensorDataset(x_data, y_data)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def main() -> None:
    """
    Initializes a CNN model with DeepSpeed and trains it on synthetic data.
    """
    model = CNNModel()

    model_engine, _, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config="ds_config.json"
    )

    data_loader = get_data_loader(batch_size=model_engine.train_micro_batch_size_per_gpu())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dtype = next(model_engine.module.parameters()).dtype

    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(30):
        for step, (x_batch, y_batch) in enumerate(data_loader):
            x_batch = x_batch.to(device).to(model_dtype)
            y_batch = y_batch.to(device)

            outputs = model_engine(x_batch)
            loss = loss_fn(outputs, y_batch)

            model_engine.backward(loss)
            model_engine.step()

            if step % 100 == 0:
                print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.6f}")


if __name__ == "__main__":
    main()
