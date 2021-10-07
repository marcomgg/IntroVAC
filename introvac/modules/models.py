import torch.nn as nn
import logging
from torchsummary import summary

activations = dict(relu=nn.ReLU, tanh=nn.Tanh, sigmoid=nn.Sigmoid)


def get_config(model_name):
    config = dict(
        Mlp=dict(num_hidden=3, hidden_sizes=128, activation='relu')
    )
    return config.get(model_name, dict())


def get_model(model_name):
    models = dict(
        Softmax=soft_reg,
        Logistic=log_reg,
        Encoder=Encoder,
        Decoder=Decoder
    )
    return models[model_name]


def num_parameters(model):
    return sum([w.numel() for w in model.parameters()])


class View(nn.Module):
    def __init__(self, o):
        super().__init__()
        self.o = o

    def forward(self, x):
        return x.view(-1, self.o)


class Linear(nn.Module):
    name = 'Linear classifier'

    def __init__(self, input_size, num_classes):
        super().__init__()

        num_classes = num_classes
        input_size = input_size

        self.m = nn.Sequential(
            nn.Linear(input_size, num_classes, bias=False)
        )

        self.N = num_parameters(self.m)
        s = '[%s] Num parameters: %d' % (self.name, self.N)
        print(s)
        logging.info(s)

    def forward(self, x):
        return self.m(x)


# Needs nn.BCEWithLogitsLoss
def log_reg(opt):
    return Linear(opt['dim'], opt['num_classes'] - 1)


def log_reg_with_logits(opt):
    return nn.Sequential(Linear(opt['dim'], opt['num_classes'] - 1), nn.Sigmoid())


def soft_reg(opt):
    return Linear(opt['dim'], opt['num_classes'])


class Classifier(nn.Module):
    def __init__(self, encoder, classifier):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier

    def forward(self, x):
        mu, var = self.encoder(x)
        return self.classifier(mu)


class _Residual_Block(nn.Module):
    def __init__(self, inc=64, outc=64, groups=1, scale=1.0):
        super(_Residual_Block, self).__init__()

        midc = int(outc * scale)

        if inc is not outc:
          self.conv_expand = nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        else:
          self.conv_expand = None

        self.conv1 = nn.Conv2d(in_channels=inc, out_channels=midc, kernel_size=3, stride=1, padding=1, groups=groups, bias=False)
        self.bn1 = nn.BatchNorm2d(midc)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=midc, out_channels=outc, kernel_size=3, stride=1, padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(outc)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        if self.conv_expand is not None:
          identity_data = self.conv_expand(x)
        else:
          identity_data = x

        output = self.relu1(self.bn1(self.conv1(x)))
        output = self.conv2(output)
        output = self.relu2(self.bn2(output))
        output = identity_data + output
        return output


class Encoder(nn.Module):
    def __init__(self, opt):
        super(Encoder, self).__init__()

        cdim = 3
        hdim = opt['dim']
        channels = opt.get('channels', [64, 128, 256, 512, 512, 512])
        image_size = opt.get('image_size', 256)

        assert (2 ** len(channels)) * 4 == image_size

        self.hdim = hdim
        cc = channels[0]
        self.main = nn.Sequential(
            nn.Conv2d(cdim, cc, 5, 1, 2, bias=False),
            nn.BatchNorm2d(cc),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(2)  #nn.MaxPool2d(2),
        )

        sz = image_size // 2
        for ch in channels[1:]:
            self.main.add_module('res_in_{}'.format(sz), _Residual_Block(cc, ch, scale=1.0))
            self.main.add_module('down_to_{}'.format(sz // 2), nn.AvgPool2d(2))
            cc, sz = ch, sz // 2

        self.main.add_module('res_in_{}'.format(sz), _Residual_Block(cc, cc, scale=1.0))
        self.fc = nn.Linear((cc) * 4 * 4, 2 * hdim)

    def forward(self, x):
        y = self.main(x).view(x.size(0), -1)
        y = self.fc(y)
        mu, logvar = y.chunk(2, dim=1)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, opt):
        super(Decoder, self).__init__()
        cdim = 3
        hdim = opt['dim']
        channels = opt.get('channels', [64, 128, 256, 512, 512, 512])
        image_size = opt.get('image_size', 256)

        assert (2 ** len(channels)) * 4 == image_size

        cc = channels[-1]
        self.fc = nn.Sequential(
            nn.Linear(hdim, cc * 4 * 4),
            nn.ReLU(True),
        )

        sz = 4

        self.main = nn.Sequential()
        for ch in channels[::-1]:
            self.main.add_module('res_in_{}'.format(sz), _Residual_Block(cc, ch, scale=1.0))
            self.main.add_module('up_to_{}'.format(sz * 2), nn.Upsample(scale_factor=2, mode='nearest'))
            cc, sz = ch, sz * 2

        self.main.add_module('res_in_{}'.format(sz), _Residual_Block(cc, cc, scale=1.0))
        self.main.add_module('predict', nn.Conv2d(cc, cdim, 5, 1, 2))

    def forward(self, z):
        z = z.view(z.size(0), -1)
        y = self.fc(z)
        y = y.view(z.size(0), -1, 4, 4)
        y = self.main(y)
        sigmoid = nn.Sigmoid()
        return sigmoid(y)


if __name__ == '__main__':
    args = dict(dim=256, channels=[32, 64, 128, 256, 512], image_size=128)
    model = Decoder(args).cuda()
    summary(model, input_size=(256,))
    model = Encoder(args).cuda()
    summary(model, input_size=(3, 128, 128))
