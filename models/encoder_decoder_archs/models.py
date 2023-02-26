from .blocks import *


class EncDecBaselineDS(nn.Module):
    def __init__(self, cardinality_of_params, is_encoder_as_baseline=False, is_decoder_as_baseline=False, with_discriminator=False):
        super(EncDecBaselineDS, self).__init__()
        self.cardinality_of_params = cardinality_of_params
        if not is_encoder_as_baseline:
            self.encoder = MyEncoder(cardinality_of_params)
        else:
            self.encoder = EncoderAsBaseline(cardinality_of_params)

        if not is_decoder_as_baseline:
            self.decoder = MyDecoder(cardinality_of_params)
        else:
            self.decoder = DecoderAsBaseline()

        self.with_discriminator = with_discriminator
        if with_discriminator:
            if not is_encoder_as_baseline:
                self.discriminator = MyEncoder(cardinality_of_params)
            else:
                self.discriminator = EncoderAsBaseline(cardinality_of_params, is_discriminator=True)
            self.discriminator_result = 0

    def forward(self, spec_in, return_bottleneck=True):
        bottleneck = self.encoder(spec_in)
        spec_out = self.decoder(bottleneck)
        if self.with_discriminator:
            self.discriminator_result = self.discriminator(spec_out)
        return spec_out, bottleneck
