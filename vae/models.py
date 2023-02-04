import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self, latent_size=64):
        super().__init__()

        image_size = (80, 160)

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2),
            nn.ReLU(),
        )

        (self.encoded_H, self.encoded_W), size_hist = self._calculate_spatial_size(image_size, self.encoder)

        self.mean = nn.Linear(self.encoded_H * self.encoded_W * 256, latent_size)
        self.logstd = nn.Linear(self.encoded_H * self.encoded_W * 256, latent_size)

        # latent
        self.latent = nn.Linear(latent_size, self.encoded_H * self.encoded_W * 256)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2),
            nn.Sigmoid(),
        )

    def encode(self, x):
        x = self.encoder(x)
        x = x.flatten(start_dim=1)
        return self.mean(x), self.logstd(x)

    def decode(self, z):
        z = self.latent(z)
        z = z.view(-1, 256, self.encoded_H, self.encoded_W)
        z = self.decoder(z)
        return z

    def reparameterize(self, mu, logvar):
        sigma = logvar.exp()
        eps = torch.randn_like(sigma)
        return eps.mul(sigma).add_(mu)

    def forward(self, x, encode=False, mean=False):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x = self.decode(z)
        return x, mu, logvar

    def _calculate_spatial_size(self, image_size, conv_layers):
        ''' Calculate spatial size after convolution layers '''
        H, W = image_size
        size_hist = []
        size_hist.append((H, W))

        for layer in conv_layers:
            if layer.__class__.__name__ != 'Conv2d':
                continue
            conv = layer
            H = int((H + 2 * conv.padding[0] - conv.dilation[0] * (conv.kernel_size[0] - 1) - 1) / conv.stride[0] + 1)
            W = int((W + 2 * conv.padding[1] - conv.dilation[1] * (conv.kernel_size[1] - 1) - 1) / conv.stride[1] + 1)

            size_hist.append((H, W))

        return (H, W), size_hist
