# cifar10_deepspeed.py

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import argparse
import deepspeed

# ----- Argument Parser -----
def add_argument():
    parser = argparse.ArgumentParser(description='CIFAR10 with DeepSpeed')
    parser.add_argument('--deepspeed_config', type=str, default='ds_config.json')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser = deepspeed.add_config_arguments(parser)
    return parser.parse_args()

# ----- CIFAR-10 Dataset -----
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5),
                                                     (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

# ----- Model Definition -----
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# ----- Main -----
def main():
    args = add_argument()
    model = Net()
    criterion = nn.CrossEntropyLoss()

    parameters = filter(lambda p: p.requires_grad, model.parameters())

    model_engine, optimizer, trainloader, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=parameters,
        training_data=trainset
    )

    for epoch in range(args.epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(model_engine.device), data[1].to(model_engine.device)
            outputs = model_engine(inputs)
            loss = criterion(outputs, labels)
            model_engine.backward(loss)
            model_engine.step()

            running_loss += loss.item()
            if i % 2000 == 1999:
                print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}")
                running_loss = 0.0

    print("Finished Training")

if __name__ == "__main__":
    main()
