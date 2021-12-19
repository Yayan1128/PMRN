
from torch import nn

from utils1.afnb2 import PMR


class PMRN(nn.Module):
    def __init__(self, cfg, device):
        super(PMRN, self).__init__()

        #
        self.image_height = cfg.image_height
        self.image_width = cfg.image_width
        self.image_channel_size = cfg.image_channel_size
        self.co_w=cfg.co_w
        self.lo_w=cfg.lo_w
        self.lc=cfg.lc
        self.lf=cfg.lf

        self.conv_channel_size = cfg.conv_channel_size#16
        self.encoder = Encoder(image_channel_size=self.image_channel_size,
                               conv_channel_size=self.conv_channel_size,
                              )
        self.encoder2 = Encoder(image_channel_size=self.image_channel_size,
                               conv_channel_size=self.conv_channel_size,
                               )

        self.decoder = Decoder(image_height=self.image_height,
                               image_width=self.image_width,
                               image_channel_size=self.image_channel_size,
                               conv_channel_size=self.conv_channel_size,
                              )

        self.fusion = PMR(128, 128, 128, 128, 128, dropout=0.05, co=self.co_w,lo=self.lo_w, sizes=([1]),#0.005
                           norm_type='batchnorm')
        self.fusion2 = PMR(128, 128, 128, 128, 128, dropout=0.05, co=self.co_w,lo=self.lo_w, sizes=([1]),
                           norm_type='batchnorm')
        self.fc1 = nn.Linear(128 * 7 * 7, 128)#7,4,5
        self.fc_bn1 = nn.BatchNorm1d(128)
        self.fc21 = nn.Linear(128, 128)

        # DECODER
        self.fc3 = nn.Linear(128, 128)
        self.fc_bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 7 * 7 * 128) #7,4,5
        self.fc_bn4 = nn.BatchNorm1d(7 * 7 * 128)#7,4,5
        self.relu = nn.ReLU()
        #
        self.latent_dim=128

    def forward(self, x):


        output0, output1,output2 = self.encoder(x)
        o = self.fusion(output0, output1, output2)
        # print(o.size())
        z3 = self.fc21(self.relu(self.fc_bn1(self.fc1(o.view(-1, self.latent_dim * 7 * 7)))))#7,4,5
        lin_z = self.relu(self.fc_bn3(self.fc3(z3)))

        z4 = self.relu(self.fc_bn4(self.fc4(lin_z))).view(-1, self.latent_dim, 7,7)#7,4,5
        z_f = self.lc*o + self.lf*z4#1,1
        # decoder
        rec_x = self.decoder(z_f)
        re0, re1, re2=self.encoder2(rec_x)
        o2 = self.fusion2( re0, re1, re2)
        z7 = self.fc21(self.relu(self.fc_bn1(self.fc1(o2.view(-1, self.latent_dim * 7 * 7)))))#7,4,5


        return rec_x,z3,z7,


class Encoder(nn.Module):
    def __init__(self, image_channel_size, conv_channel_size):
        super(Encoder, self).__init__()
        self.image_channel_size = image_channel_size
        self.conv_channel_size = conv_channel_size

        self.conv1 = nn.Conv2d(in_channels=self.image_channel_size,
                               out_channels=self.conv_channel_size,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                              )

        self.bn1 = nn.BatchNorm2d(num_features=self.conv_channel_size,)

        self.conv2 = nn.Conv2d(in_channels=self.conv_channel_size,
                               out_channels=self.conv_channel_size*2,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                              )

        self.bn2 = nn.BatchNorm2d(num_features=self.conv_channel_size*2,)

        self.conv3 = nn.Conv2d(in_channels=self.conv_channel_size*2,
                               out_channels=self.conv_channel_size*2,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                              )

        self.bn3 = nn.BatchNorm2d(num_features=self.conv_channel_size*2,)

        self.conv4 = nn.Conv2d(in_channels=self.conv_channel_size * 2,
                               out_channels=self.conv_channel_size * 2,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               )

        self.bn4 = nn.BatchNorm2d(num_features=self.conv_channel_size * 2, )

        self.conv5 = nn.Conv2d(in_channels=self.conv_channel_size * 2,
                               out_channels=self.conv_channel_size * 2,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               )

        self.bn5 = nn.BatchNorm2d(num_features=self.conv_channel_size * 2, )



        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)



        x = self.conv3(x)
        x = self.bn3(x)
        x0 = self.relu(x)

        x = self.conv4(x0)
        x = self.bn4(x)
        x1 = self.relu(x)#16

        x2 = self.conv5(x1)
        x2 = self.bn5(x2)
        x2 = self.relu(x2)#


        return x0,x1, x2


class Decoder(nn.Module):
    def __init__(self, image_height, image_width, image_channel_size, conv_channel_size):
        super(Decoder, self).__init__()
        self.image_height = image_height
        self.image_width = image_width
        self.image_channel_size = image_channel_size
        self.conv_channel_size = conv_channel_size

        self.deconv2 = nn.ConvTranspose2d(in_channels=self.conv_channel_size*2,
                                          out_channels=self.conv_channel_size*2,
                                          kernel_size=3,
                                          stride=2,
                                          padding=1,
                                          output_padding=1,
                                         )

        self.bn2 = nn.BatchNorm2d(num_features=self.conv_channel_size*2,)

        self.deconv3 = nn.ConvTranspose2d(in_channels=self.conv_channel_size * 2,
                                          out_channels=self.conv_channel_size* 2,
                                          kernel_size=3,
                                          stride=2,
                                          padding=1,
                                          output_padding=1,
                                          )

        self.bn3 = nn.BatchNorm2d(num_features=self.conv_channel_size* 2, )

        self.deconv4 = nn.ConvTranspose2d(in_channels=self.conv_channel_size * 2,
                                          out_channels=self.conv_channel_size * 2,
                                          kernel_size=3,
                                          stride=2,
                                          padding=1,
                                          output_padding=1,
                                          )

        self.bn4 = nn.BatchNorm2d(num_features=self.conv_channel_size * 2, )

        self.deconv5 = nn.ConvTranspose2d(in_channels=self.conv_channel_size * 2,
                                          out_channels=self.conv_channel_size ,
                                          kernel_size=3,
                                          stride=2,
                                          padding=1,
                                          output_padding=1,
                                          )

        self.bn5 = nn.BatchNorm2d(num_features=self.conv_channel_size, )


        self.deconv6 = nn.ConvTranspose2d(in_channels=self.conv_channel_size,
                                          out_channels=self.image_channel_size,
                                          kernel_size=3,
                                          stride=2,
                                          padding=1,
                                          output_padding=1,
                                         )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        x = self.deconv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        # print(x.size())

        x = self.deconv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        # print(x.size())


        x = self.deconv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        # print(x.size())


        x = self.deconv5(x)
        x = self.bn5(x)
        x = self.relu(x)
        # print(x.size())

        x = self.deconv6(x)

        # print(x.size())
        return x
