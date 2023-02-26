from torch import nn


class EncoderAsBaseline(nn.Module):
    def __init__(self, cardinality_list, input_shape='4sec', is_discriminator=False):
        super(EncoderAsBaseline, self).__init__()

        self.input_shape = input_shape
        self.cardinality_list = cardinality_list
        self.ch = 1

        self.encoder_out_dim = 0

        if not is_discriminator:
            if self.input_shape == '4sec':
                for d in cardinality_list:
                    if d == -1:
                        self.encoder_out_dim += 1
                    elif d > 1:
                        self.encoder_out_dim += d
        else:
            self.encoder_out_dim = 1

        self.enc_nn = nn.Sequential(
            nn.Conv2d(self.ch, 8, (5, 5), stride=2, padding=2),
            nn.LeakyReLU(negative_slope=0.1),

            nn.Conv2d(8, 16, (4, 4), stride=2, padding=2),
            nn.LeakyReLU(negative_slope=0.1),
            nn.BatchNorm2d(16, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),

            nn.Conv2d(16, 32, (4, 4), stride=2, padding=2),
            nn.LeakyReLU(negative_slope=0.1),
            nn.BatchNorm2d(32, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),

            nn.Conv2d(32, 64, (4, 4), stride=2, padding=2),
            nn.LeakyReLU(negative_slope=0.1),
            nn.BatchNorm2d(64, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),

            nn.Conv2d(64, 128, (4, 4), stride=2, padding=2),
            nn.LeakyReLU(negative_slope=0.1),
            nn.BatchNorm2d(128, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),

            nn.Conv2d(128, 256, (4, 4), stride=2, padding=2),
            nn.LeakyReLU(negative_slope=0.1),
            nn.BatchNorm2d(256, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        )

        self.features_mixer_cnn = nn.Sequential(
            nn.Conv2d(256, 512, (4, 4), stride=2, padding=2),
            nn.LeakyReLU(negative_slope=0.1),
            nn.BatchNorm2d(512, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),

            nn.Conv2d(512, 2048, (1, 1), stride=1),
            nn.LeakyReLU(negative_slope=0.1),
        )

        self.mlp = nn.Sequential(
            nn.Dropout(p=0.3, inplace=False),
            nn.Linear(in_features=24576, out_features=self.encoder_out_dim),
            nn.BatchNorm1d(self.encoder_out_dim, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        )

    def forward(self, x):
        x = self.enc_nn(x)
        x = self.features_mixer_cnn(x)
        x = x.view(x.shape[0], -1)
        x = self.mlp(x)
        return x


class DecoderAsBaseline(nn.Module):
    def __init__(self):
        super(DecoderAsBaseline, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_features=610, out_features=24576, bias=True),
            nn.Dropout(p=0.3, inplace=False)
        )

        self.features_unmixer_cnn = nn.Sequential(
            nn.ConvTranspose2d(2048, 512, kernel_size=(1, 1), stride=(1, 1)),
            nn.LeakyReLU(negative_slope=0.1),
            nn.BatchNorm2d(512, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        )

        self.single_ch_cnn = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=(4, 4), stride=(2, 2), padding=(2, 2), output_padding=(1, 1)),
            nn.LeakyReLU(negative_slope=0.1),
            nn.BatchNorm2d(256, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),

            nn.ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(2, 2), output_padding=(1, 0)),
            nn.LeakyReLU(negative_slope=0.1),
            nn.BatchNorm2d(128, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),

            nn.ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(2, 2), output_padding=(1, 1)),
            nn.LeakyReLU(negative_slope=0.1),
            nn.BatchNorm2d(64, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),

            nn.ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=(2, 2), padding=(2, 2), output_padding=(1, 1)),
            nn.LeakyReLU(negative_slope=0.1),
            nn.BatchNorm2d(32, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),

            nn.ConvTranspose2d(32, 16, kernel_size=(4, 4), stride=(2, 2), padding=(2, 2), output_padding=(1, 0)),
            nn.LeakyReLU(negative_slope=0.1),
            nn.BatchNorm2d(16, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),

            nn.ConvTranspose2d(16, 8, kernel_size=(4, 4), stride=(2, 2), padding=(2, 2), output_padding=(1, 0)),
            nn.LeakyReLU(negative_slope=0.1),
            nn.BatchNorm2d(8, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),

            nn.ConvTranspose2d(8, 1, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
            nn.Hardtanh(min_val=-1.0, max_val=1.0)
        )
        #
        # self.dense = nn.Sequential(
        #     nn.Linear(self.gen_latent_dim, 512 * 8 * 11),
        #     nn.BatchNorm1d(512 * 8 * 11),
        #     nn.ReLU(),
        # )
        # self.l1 = nn.Sequential(
        #     nn.utils.spectral_norm(
        #         nn.ConvTranspose2d(8 * self.gf, 8 * self.gf, (12, 3), stride=2, padding=(5, 1), output_padding=(0, 1))),
        #     # nn.LeakyReLU(cfg.alpha, inplace=True),
        #     nn.BatchNorm2d(8 * self.gf),
        #     nn.ReLU(),
        # )
        # self.l2 = nn.Sequential(
        #     nn.utils.spectral_norm(
        #         nn.ConvTranspose2d(8 * self.gf, 4 * self.gf, (12, 3), stride=2, padding=(5, 1), output_padding=(0, 1))),
        #     # nn.LeakyReLU(cfg.alpha, inplace=True),
        #     nn.BatchNorm2d(4 * self.gf),
        #     nn.ReLU(),
        # )
        # self.l3 = nn.Sequential(
        #     nn.utils.spectral_norm(
        #         nn.ConvTranspose2d(4 * self.gf, 2 * self.gf, (12, 3), stride=2, padding=(5, 1), output_padding=(0, 0))),
        #     # nn.LeakyReLU(cfg.alpha, inplace=True),
        #     nn.BatchNorm2d(2 * self.gf),
        #     nn.ReLU(),
        # )
        # self.l4 = nn.Sequential(
        #     nn.utils.spectral_norm(
        #         nn.ConvTranspose2d(2 * self.gf, self.gf, (12, 3), stride=2, padding=(5, 1), output_padding=(0, 1))),
        #     # nn.LeakyReLU(cfg.alpha, inplace=True),
        #     nn.BatchNorm2d(self.gf),
        #     nn.ReLU(),
        # )
        # self.final = nn.Sequential(
        #     nn.utils.spectral_norm(
        #         nn.ConvTranspose2d(self.gf, self.ch, (12, 3), stride=2, padding=(5, 1), output_padding=(1, 0))),
        #     # nn.BatchNorm2d(cfg.ch),
        # )
        #
        # # before_relu_dim = 50
        # # second_fc_out_dim = 75
        # before_relu_dim = 100
        # second_fc_out_dim = 150
        #
        # self.cardinality_list = cardinality_list
        # self.linear_layers = nn.ModuleList()
        # vector_len = 0
        # for d in cardinality_list:
        #     if d > 1:
        #         self.linear_layers.append(nn.Linear(d, self.mapper_fc_out_dim, bias=False))
        #         vector_len += self.mapper_fc_out_dim
        #     if d == -1:
        #         vector_len += 1
        #
        # self.concat2relu = nn.Linear(vector_len, before_relu_dim)
        # self.concat2fc = nn.Linear(before_relu_dim, second_fc_out_dim)
        # self.concat2latent = nn.Linear(second_fc_out_dim, self.gen_latent_dim)
        # self.relu = nn.ReLU()

    def forward(self, v_in):

        x = self.mlp(v_in)
        x = x.view(-1, 2048, 3, 4)
        x = self.features_unmixer_cnn(x)
        x = self.single_ch_cnn(x)
        return x
