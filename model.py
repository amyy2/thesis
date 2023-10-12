import torch
import torch.nn as nn

def double_conv_3d(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True)
    )

class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=4):
        super(UNet3D, self).__init__()
        self.enc1 = double_conv_3d(in_channels, 32)
        self.enc2 = double_conv_3d(32, 64)
        self.enc3 = double_conv_3d(64, 128)
        self.enc4 = double_conv_3d(128, 256)

        self.pool = nn.MaxPool3d(2)
        self.upconv3 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.upconv1 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)

        self.dec3 = double_conv_3d(256, 128)
        self.dec2 = double_conv_3d(128, 64)
        self.dec1 = double_conv_3d(64, 32)

        self.out_conv = nn.Conv3d(32, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        dec3 = self.dec3(torch.cat([self.upconv3(enc4), enc3], dim=1))
        dec2 = self.dec2(torch.cat([self.upconv2(dec3), enc2], dim=1))
        dec1 = self.dec1(torch.cat([self.upconv1(dec2), enc1], dim=1))

        return torch.sigmoid(self.out_conv(dec1))