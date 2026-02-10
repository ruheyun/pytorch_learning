import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter


class Discriminator(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.disc(x)
    

class Generator(nn.Module):
    def __init__(self, z_dim, image_dim):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, image_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.gen(x)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lr=3e-4
z_dim = 64
image_dim = 28 * 28 * 1
batch_size = 128
num_epochs = 100

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

Disc = Discriminator(image_dim).to(device)
Gen = Generator(z_dim, image_dim).to(device)
opt_disc = optim.Adam(Disc.parameters(), lr=lr)
opt_gen = optim.Adam(Gen.parameters(), lr=lr)
criterion = nn.BCELoss()

fixed_noise = torch.randn((batch_size, z_dim)).to(device)
writer_fake = SummaryWriter('logs/fake')
writer_real = SummaryWriter('logs/real')
step = 0

for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.view(-1, 28 * 28).to(device)
        batch_size = real.shape[0]

        disc_real = Disc(real).view(-1)
        lossD_real = criterion(disc_real, torch.ones_like(disc_real))

        noise = torch.randn((batch_size, z_dim)).to(device)
        fake = Gen(noise)
        disc_fake = Disc(fake).view(-1)
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        lossD = (lossD_real + lossD_fake) / 2

        Disc.zero_grad()
        lossD.backward(retain_graph=True)
        opt_disc.step()

        output = Disc(fake).view(-1)
        lossG = criterion(output, torch.ones_like(output))
        Gen.zero_grad()
        lossG.backward()
        opt_gen.step()

        if batch_idx == 0:
            print(f'epoch: {epoch} / {num_epochs}, loss D: {lossD.item():.4f}, loss G: {lossG.item():.4f}')
            with torch.no_grad():
                fake = Gen(fixed_noise).reshape(-1, 1, 28, 28)
                data = real.reshape(-1, 1, 28, 28)
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(data, normalize=True)

                writer_fake.add_image('MNIST Fake Image', img_grid_fake, global_step=step)
                writer_real.add_image('MNIST Real Image', img_grid_real, global_step=step)
                step += 1
