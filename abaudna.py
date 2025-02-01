import torch  
import torch.nn as nn  
import torchvision.transforms as transforms  
import torchvision.datasets as datasets  
from torch.utils.data import DataLoader  
import numpy as np  
import cv2  
 
class Generator(nn.Module):  
    def __init__(self):  
        super(Generator, self).init()  
        self.model = nn.Sequential(  
            nn.Linear(100, 128),  
            nn.ReLU(True),  
            nn.Linear(128, 256),  
            nn.ReLU(True),  
            nn.Linear(256, 512),  
            nn.ReLU(True),  
            nn.Linear(512, 784),  
            nn.Tanh()  
        )  

    def forward(self, z):  
        return self.model(z).view(-1, 1, 28, 28)  
class Discriminator(nn.Module):  
    def __init__(self):  
        super(Discriminator, self).init()  
        self.model = nn.Sequential(  
            nn.Linear(784, 512),  
            nn.LeakyReLU(0.2, inplace=True),  
            nn.Linear(512, 256),  
            nn.LeakyReLU(0.2, inplace=True),  
            nn.Linear(256, 1),  
            nn.Sigmoid()  
        )  

    def forward(self, img):  
        return self.model(img.view(-1, 784))  

def train_gan(generator, discriminator, dataloader, num_epochs):  
    criterion = nn.BCELoss()  
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))  
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))  

    for epoch in range(num_epochs):  
        for i, (imgs, _) in enumerate(dataloader):  
            batch_size = imgs.size(0)  
            valid = torch.ones(batch_size, 1)  
            fake = torch.zeros(batch_size, 1)  
            optimizer_D.zero_grad()  
            real_loss = criterion(discriminator(imgs), valid)  
            z = torch.randn(batch_size, 100)  
            gen_imgs = generator(z)  
            fake_loss = criterion(discriminator(gen_imgs.detach()), fake)  
            d_loss = real_loss + fake_loss  
            for param in discriminator.parameters():  
                d_loss += 0.01 * torch.norm(param, p=1)   
            
            d_loss.backward()  
            optimizer_D.step()   
            optimizer_G.zero_grad()  
            g_loss = criterion(discriminator(gen_imgs), valid)  
            g_loss.backward()  
            optimizer_G.step()  

        print(f'Epoch [{epoch+1}/{num_epochs}], D Loss: {d_loss.item()}, G Loss: {g_loss.item()}')  
        if (epoch + 1) % 10 == 0:  
            z = torch.randn(16, 100)  
            gen_imgs = generator(z).detach().cpu().numpy()  
            gen_imgs = (gen_imgs * 255).astype(np.uint8)  
            for i in range(16):  
                cv2.imwrite(f'generated_{epoch+1}_{i}.png', gen_imgs[i][0])    
transform = transforms.Compose([  
    transforms.ToTensor(),  
    transforms.Normalize((0.5,), (0.5,))  
])  

dataset = datasets.MNIST('data', train=True, download=True, transform=transform)  
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)  
 
generator = Generator()  
discriminator = Discriminator()  
train_gan(generator, discriminator, dataloader, num_epochs= 100)
