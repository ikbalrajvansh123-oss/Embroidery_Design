import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(i, o, 3, padding=1),
            nn.BatchNorm2d(o),
            nn.ReLU(inplace=True),
            nn.Conv2d(o, o, 3, padding=1),
            nn.BatchNorm2d(o),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.conv(x)

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = ConvBlock(3, 64)
        self.c2 = ConvBlock(64,128)
        self.c3 = ConvBlock(128,256)
        self.pool = nn.MaxPool2d(2)
        self.b  = ConvBlock(256,512)
        self.u3 = nn.ConvTranspose2d(512,256,2,2); self.d3 = ConvBlock(512,256)
        self.u2 = nn.ConvTranspose2d(256,128,2,2); self.d2 = ConvBlock(256,128)
        self.u1 = nn.ConvTranspose2d(128,64,2,2); self.d1 = ConvBlock(128,64)
        self.out = nn.Conv2d(64,1,1)

    def forward(self,x):
        c1 = self.c1(x)
        c2 = self.c2(self.pool(c1))
        c3 = self.c3(self.pool(c2))
        b  = self.b(self.pool(c3))
        d3 = self.d3(torch.cat([self.u3(b), c3],1))
        d2 = self.d2(torch.cat([self.u2(d3),c2],1))
        d1 = self.d1(torch.cat([self.u1(d2),c1],1))
        return self.out(d1)