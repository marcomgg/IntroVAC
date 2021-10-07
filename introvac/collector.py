import sys
sys.path.append(".")
from argparse import ArgumentParser
import introvac.modules.utils as ut
import torch
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from introvac.modules.models import get_model
from types import SimpleNamespace
import matplotlib.pyplot as plt
from os.path import join
import seaborn as sns
import numpy as np
import torch.nn as nn
import json
from pathlib import Path
from tqdm import tqdm
from torchvision.utils import save_image
parser = ArgumentParser("Results Collector")

add = parser.add_argument
add('--root', type=str, default='results/save', help="Folder containing all executions results")
add('--folder', type=str, default=None, help="Folder inside root containing the checkpoints")
add('--save_folder', type=str, default='results/collect', help='Folder where collected results are saved')
add('--distance', type=float, default=4.0, help='Distance to move in the latent space')
add('--ratio', type=int, default=2, help='How much data of the test set to keep')
add('--direction', type=int, default=[1, -1], nargs='+', help="Direction to move for every logit in the classifier")
add('--best', type=int, default=0, help='Use best model?')
args = parser.parse_args()


def show(tensor, nrow, x_tick_labels=[str(i) for i in range(12)]):
    plt.figure(dpi=200)
    img = make_grid(tensor, nrow=nrow)
    img = img.permute(1, 2, 0)
    plt.imshow(img)


def save_tensor_images(images, paths):
    for i, image in enumerate(images):
        save_image(image, paths[i])


def recontruct(model, x):
    z, _ = model.encoder(x)
    x_hat = model.decoder(z)
    return x_hat


def langevin(classifier, eps, steps, cl=1, size=1024):
    bce = torch.nn.BCEWithLogitsLoss()
    mse = torch.nn.MSELoss()
    z_start = torch.randn(size, 256).to(device)
    z_start.requires_grad = True
    for i in range(steps):
        y_hat = classifier(z_start)
        y = torch.ones_like(y_hat) * cl
        y[:, -1] = 0.0
        loss = bce(y_hat, y) + mse(z_start, torch.zeros_like(z_start))
        loss.backward()
        z_start.data = z_start.data - eps * z_start.grad.data + torch.randn(size, 256).to(device) * np.sqrt(2 * eps)
    return z_start


def langevin2(classifier, eps, steps, cl=1):
    bce = torch.nn.BCEWithLogitsLoss()
    mse = torch.nn.MSELoss()
    z_start = torch.randn(1, 256).to(device)
    z_start.requires_grad = True
    z_s = []
    for i in range(steps):
        y_hat = classifier(z_start)
        y = torch.ones_like(y_hat) * cl
        y[:, -1] = 0.0
        loss = bce(y_hat, y) + mse(z_start, torch.zeros_like(z_start))
        loss.backward()
        z_start.data = z_start.data - eps * z_start.grad.data + torch.randn(1, 256).to(device) * eps
        if i >= steps - 1024:
            z_s.append(z_start.data.clone())
    print(y)
    return torch.cat(z_s)


class SpaceManipulator:
    def __init__(self, model, class_type='Logit', num_classes=2):
        self.model = model
        self.class_type = class_type
        self.num_classes = num_classes
        self.eval()

    def eval(self):
        self.model.encoder.eval()
        self.model.decoder.eval()
        self.model.classifier.eval()

    def get_top_n(self, n, save_folder=None):
        for name, p in self.model.classifier.named_parameters():
            if 'weight' in name:
                if save_folder:
                    sns.set_style('darkgrid')
                    sns.scatterplot(np.arange(p[0, :].shape[0]), p[0, :].data.cpu())
                    plt.title('Weights')
                    sns.set_style('white')
                    name = join(save_folder, 'weights.pdf')
                    plt.savefig(name)
                idx = torch.argsort(torch.abs(p[0, :]))

        top_n = idx[-n:]
        return top_n

    @torch.no_grad()
    def add_attribute_linear(self, x, direction=[1, -1], top_n=2, distance=20):
        mu_z, _ = self.model.encoder(x)
        mu_z_1 = self.linear_transfer(mu_z.data.clone(), direction=direction, top_n=top_n, distance=distance)

        x_hat, _ = self.model.decoder(mu_z), self.model.classifier(mu_z)
        x_aug, y_aug = self.model.decoder(mu_z_1), self.model.classifier(mu_z_1)

        return x, x_hat, x_aug, y_aug

    def linear_transfer(self, z, direction=[1, -1], distance=2000, top_n=3):
        top_n = self.get_top_n(top_n)
        w = None
        for name, p in self.model.classifier.named_parameters():
            if 'weight' in name:
                for i, d in enumerate(direction):
                    w = d * p[i, :] / p[i, :].norm() if w is None else w + d * p[i, :] / p[i, :].norm()
        direction = w[top_n].squeeze() / w[top_n].norm()
        z[:, top_n] = z[:, top_n] + distance * direction
        return z


class WrappedModel(nn.Module):
    def __init__(self, module):
        super(WrappedModel, self).__init__()
        self.module = module

    def forward(self, x):
        return self.module(x)


if __name__ == '__main__':
    print(args.direction)
    folder_path = join(args.root, args.folder)
    best = '_best' if args.best == 1 else ''
    checkpoint_file = join(folder_path, f'checkpoint{best}.pkl')
    checkpoint = SimpleNamespace()
    ut.load_opts(checkpoint, checkpoint_file)
    opt = checkpoint.opt
    print(f"Epoch: {checkpoint.epoch} Attributes: {opt['attributes']} Ratio {opt['ratio']} Weight decay: {opt['wd_class']} Wd: {opt['wd']} Betas: {opt['betas']} ")

    attributes = opt['attributes']
    dir_str = '' if len(attributes) < 2 else '_'.join([str(d) for d in args.direction])
    save_folder = Path(args.save_folder, '-'.join(attributes), dir_str)
    save_folder.mkdir(parents=True, exist_ok=True)
    save_trans = Path(save_folder, 'transfer')
    save_trans.mkdir(parents=True, exist_ok=True)
    save_gen = Path(save_folder, 'gen')
    save_gen.mkdir(parents=True, exist_ok=True)
    save_condgen = Path(save_folder, 'condgen')
    save_condgen.mkdir(parents=True, exist_ok=True)
    save_reconst = Path(save_folder, 'reconst')
    save_reconst.mkdir(parents=True, exist_ok=True)

    ratio = opt['ratio']
    dataset = opt['dataset']

    model = SimpleNamespace()
    classifier = opt['classifier']
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    model.encoder = WrappedModel(get_model("Encoder")(opt=opt).to(device))  # get_model(f"{opt['model']}Encoder")(opt=args).to(device)
    model.decoder = WrappedModel(get_model("Decoder")(opt=opt).to(device))  # get_model(f"CustomDecoder")(args).to(device)
    model.classifier = WrappedModel(get_model(classifier)(opt).to(device))
    ut.load_models(model, checkpoint_file, False)

    bs = 64
    train_set, test_set, validation_set, _, _, _ = ut.get_dataset(dataset, opt['dataset_root'], False)
    test_loader = DataLoader(test_set, batch_size=bs, shuffle=False)
    print("Test set length all:", len(test_set))
    for i, data in tqdm(enumerate(test_loader), total=len(test_loader)):
        x = data[0].to(device)
        x_hat = recontruct(model, x)
        paths = [Path(save_reconst, f'img_{j}.png') for j in range(i * bs, i * bs + bs)]
        save_tensor_images(x, paths)
        paths = [Path(save_reconst, f'img_{j}_r.png') for j in range(i * bs, i * bs + bs)]
        save_tensor_images(x_hat, paths)

    test_set = torch.utils.data.Subset(test_set, list(range(len(test_set) // args.ratio)))
    mask = None
    if dataset.lower() == 'celeba':
        mask = ut.get_mask(attributes, train_set.attr_names)

    test_loader = DataLoader(test_set, batch_size=bs, shuffle=False)
    print("Test set length:", len(test_set))

    mn = SpaceManipulator(model)
    distances = np.linspace(0.5, 10, len(test_loader))
    for i, data in tqdm(enumerate(test_loader), total=len(test_loader)):
        x = data[0].to(device)
        y = data[0].to(device)
        x, x_hat, x_aug, y_hat = mn.add_attribute_linear(x=x, direction=args.direction, distance=args.distance, top_n=opt['dim'])
        # print(f"Distance: {distances[i]:.4f} Error:{torch.abs(x_hat - x_aug).mean():.4f} Confidence: {torch.sigmoid(y_hat[:, 0]).mean():.4f}")
        paths = [Path(save_trans, f'img_{j}.png') for j in range(i * bs, i * bs + bs)]
        save_tensor_images(x, paths)
        paths = [Path(save_trans, f'img_{j}_r.png') for j in range(i * bs, i * bs + bs)]
        save_tensor_images(x_hat, paths)
        paths = [Path(save_trans, f'img_{j}_a.png') for j in range(i * bs, i * bs + bs)]
        save_tensor_images(x_aug, paths)

    torch.manual_seed(300)
    torch.backends.cudnn.deterministic = True

    size = 1024
    thr = 1.2
    bs = 64
    for i in tqdm(range(0, size, bs)):
        z_gen = torch.randn(bs, 256).to(device)
        x_gen = model.decoder(z_gen)
        paths = [Path(save_gen, f'img_{j}.png') for j in range(i, i + bs)]
        save_tensor_images(x_gen, paths)

    z = langevin(model.classifier, 0.0002, 5000, 1)
    z_path = Path(save_condgen, 'z_cond.pkl')
    torch.save(z, z_path)
    ys = []
    print(z.shape)
    with torch.no_grad():
        for i in tqdm(range(0, size, bs)):
            z_gen = z[i:i + bs]
            x_gen = model.decoder(z_gen)
            mu, _ = model.encoder(x_gen.detach())
            y = torch.round(torch.sigmoid(model.classifier(mu)))
            ys.append(y)
            paths = [Path(save_condgen, f'img_{j}.png') for j in range(i, i + bs)]
            save_tensor_images(x_gen, paths)
    ys = torch.cat(ys).cpu().numpy()
    print("Correct conditional:", np.sum((ys[:, 0] == 1.)) / ys.shape[0])
