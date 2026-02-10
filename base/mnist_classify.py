import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('./logs')

torch.manual_seed(2025)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

train_data = MNIST('./data', train=True, transform=transform, download=True)
test_data = MNIST('./data', train=False, transform=transform, download=False)

train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 6, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 120, 5),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = CNN().to(device)

loss_fuc = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=9e-1)

train_losses = []
train_acces = []
eval_losses = []
eval_acces = []

for epoch in range(15):
    train_loss = 0
    train_acc = 0
    model.train()
    for data, label in train_dataloader:
        data = data.to(device)
        label = label.to(device)
        out = model(data)
        loss = loss_fuc(out, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, pres = out.max(1)
        num_true = (pres == label).sum().item()
        acc = num_true / len(data)
        train_loss += loss.item()
        train_acc += acc 
    train_losses.append(train_loss / len(train_dataloader))
    train_acces.append(train_acc / len(train_dataloader))

    eval_loss = 0
    eval_acc = 0
    model.eval()
    with torch.no_grad():
        for data, label in test_dataloader:
            data = data.to(device)
            label = label.to(device)
            out = model(data)
            loss = loss_fuc(out, label)

            _, pres = out.max(1)
            num_true = (pres == label).sum().item()
            acc = num_true / len(data)
            eval_loss += loss.item()
            eval_acc += acc
        eval_losses.append(eval_loss / len(test_dataloader))
        eval_acces.append(eval_acc / len(test_dataloader))
    print(f'epoch: {epoch + 1}, train_losses: {train_losses[-1]: .3f}, train_acces: {train_acces[-1]: .3f}, eval_losses: {eval_losses[-1]: .3f}, eval_acces: {eval_acces[-1]: .3f}')

image = torch.ones(64, 1, 28, 28).to(device)
writer.add_graph(model, image)

for i, (data, _) in enumerate(test_dataloader):
    writer.add_image('image', data, i, dataformats='NCHW')

for i in range(len(train_losses)):
    writer.add_scalars('loss', {'train':train_losses[i], 'eval':eval_losses[i]}, i)
    writer.add_scalars('acc', {'train':train_acces[i], 'eval':eval_acces[i]}, i)
writer.close()