"""Train a simple linear model using DeepSpeed on dummy data."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import deepspeed


class SimpleModel(nn.Module):
    """
    A simple linear regression model: y = Wx + b
    """

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the model.
        """
        return self.linear(x)


def get_data_loader(batch_size: int) -> DataLoader:
    """
    Generates a dummy dataset y = 2x + 1 and returns a DataLoader.
    """
    x_data = torch.randn(1000, 1)
    y_data = 2 * x_data + 1
    dataset = TensorDataset(x_data, y_data)
    return DataLoader(dataset, batch_size=batch_size)


def main() -> None:
    """
    Initializes the model with DeepSpeed, trains it over a dummy dataset,
    and prints loss at intervals.
    """
    model = SimpleModel()

    model_engine, _, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config="ds_config.json"
    )

    data_loader = get_data_loader(batch_size=model_engine.train_micro_batch_size_per_gpu())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_dtype = next(model_engine.module.parameters()).dtype

    for epoch in range(30):
        for step, (x_batch, y_batch) in enumerate(data_loader):
            x_batch = x_batch.to(device).to(model_dtype)
            y_batch = y_batch.to(device).to(model_dtype)

            outputs = model_engine(x_batch)
            loss = nn.functional.mse_loss(outputs, y_batch)

            model_engine.backward(loss)
            model_engine.step()

            if step % 10 == 0:
                print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.6f}")


if __name__ == "__main__":
    main()
