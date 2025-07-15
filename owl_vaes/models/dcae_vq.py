
from .dcae import DCAE
from ..nn.lfq import LFQModule

class DCVQVAE(DCAE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.lfq = LFQModule(self.config.codebook_size)

    def forward(self, x):
        z = self.encoder(x)[0]
        dec_in = self.lfq(z)
        x_rec = self.decoder(dec_in)
        return x_rec, z, None
