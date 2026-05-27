"""
冻结部分网络层，只更新剩下网络层
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, 10)

        self.relu = nn.ReLU()

    def forward(self, x):

        x = x.view(x.size(0), -1)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        x = self.fc6(x)

        return x


def freeze_all():

    for param in model.parameters():
        param.requires_grad = False


def unfreeze_all():

    for param in model.parameters():
        param.requires_grad = True


def unfreeze_layers(layers_name):

    for name in layers_name:
        
        layer = getattr(model, name)

        for param in layer.parameters():
            param.requires_grad = True


transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = MLP().to(device)

criterion = nn.CrossEntropyLoss()

epochs = 20

for epoch in range(epochs):
    print(f'epoch: {epoch}')

    if epoch == 0:

        freeze_all()
        unfreeze_layers(['fc6'])
        print(f'训练 fc6')

    elif epoch == 5:

        freeze_all()
        unfreeze_layers(['fc5', 'fc6'])
        print(f'训练 fc5，fc6')
    
    elif epoch == 10:

        unfreeze_all()
        print(f'训练所有层')

    # 关键：optimizer必须重新构建
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3
    )

    model.train()

    total_loss = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss:.4f}")

