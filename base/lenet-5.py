# coding:UTF-8
# RuHe  2025/3/24 21:41
# lenet-5实现

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

torch.manual_seed(2025)
writer = SummaryWriter('logs')


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(16 * 5 * 5, 120),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x


model = LeNet()
input = torch.ones(64, 1, 32, 32)
out = model(input)
print(out.shape)
writer.add_graph(model, input)

transform = transforms.Compose([
    transforms.Resize([32, 32]),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
train_data = MNIST('./data', train=True, transform=transform, download=True)
test_data = MNIST('./data', train=False, transform=transform, download=False)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

for i, (img, label) in enumerate(train_loader):
    writer.add_image('train_loader', img, i, dataformats='NCHW')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

model = LeNet().to(device)

loss_fn = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))


for epoch in range(15):
    model.train()
    train_loss = 0
    train_acc = 0
    for image, label in train_loader:
        image = image.to(device)
        label = label.to(device)
        out = model(image)
        optimizer.zero_grad()
        loss = loss_fn(out, label)
        loss.backward()
        optimizer.step()

        _, pres = out.max(1)
        num_true = (pres == label).sum().item()
        acc = num_true / len(label)
        train_loss += loss.item()
        train_acc += acc

    train_loss /= len(train_loader)
    train_acc /= len(train_loader)

    model.eval()
    eval_loss = 0
    eval_acc = 0
    with torch.no_grad():
        for image, label in test_loader:
            image = image.to(device)
            label = label.to(device)
            out = model(image)
            loss = loss_fn(out, label)

            _, pres = out.max(1)
            num_true = (pres == label).sum().item()
            acc = num_true / len(label)
            eval_loss += loss.item()
            eval_acc += acc
    eval_loss /= len(test_loader)
    eval_acc /= len(test_loader)

    writer.add_scalar('train_loss', train_loss, epoch)
    writer.add_scalar('train_acc', train_acc, epoch)
    writer.add_scalar('eval_loss', eval_loss, epoch)
    writer.add_scalar('eval_acc', eval_acc, epoch)
    print(f'第{epoch + 1}轮训练结束, train_loss: {train_loss: .3f}, train_acc: {train_acc: .3f}, eval_loss: {eval_loss: .3f}, eval_acc: {eval_acc: .3f}')

writer.close()
