import time
import json
from torchvision import transforms
from torchvision.utils import make_grid
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns


def get_gradients(models, model_names):
    grads = {}
    for model, model_name in zip(models, model_names):
        for name, p in model.named_parameters():
            grads.update({f"{model_name} grad {name}": p.grad.data.cpu().numpy()})
    return grads


def to_tf_images(images):
    return images.data.cpu().numpy().transpose(0, 2, 3, 1)


def accuracy(matrix):
    return ((matrix[:, 0, 0].sum() + matrix[:, 1, 1].sum()) / matrix.sum()) * 100


def plot_confusion(mat, title, cbar=False):
    fig = plt.figure(dpi=200)
    ax = sns.heatmap(mat, cmap='Blues', annot=True, cbar=cbar)
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Real Class")
    b, t = plt.ylim()  # discover the values for bottom and top
    b += 0.5  # Add 0.5 to the bottom
    t -= 0.5  # Subtract 0.5 from the top
    plt.ylim(b, t)
    plt.title(title)
    plt.tight_layout()
    return fig


def create_confusion_grid(mat, attributes):
    data = []
    for i, m in enumerate(mat):
        fig = plot_confusion(m, attributes[i])
        array = fig2data(fig)
        data.append(array[None, :, :, :].transpose(0, 3, 1, 2))
        plt.close(fig)
    nrow = min(2, mat.shape[0])
    return make_grid(torch.tensor(np.concatenate(data, axis=0)), nrow=nrow).numpy().transpose(1, 2, 0)


def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGB values
    """
    # draw the renderer
    fig.canvas.draw()
    return np.array(fig.canvas.renderer.buffer_rgba())[:, :, 0:3]


def schedule(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_label(mask, y, long=False, keepdim=True, binary=True):
    if binary:
        label = ((y * torch.tensor(mask)).sum(dim=1, keepdim=keepdim) >= 1).long()
    else:
        label = y[:, mask.astype(bool)]
    label = label if long else label.float()
    return label


def decay(optimizer, decay):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay


def init_opt(ctx):
    cfg = ctx.ex.current_run.config
    opt = dict()
    for k, v in cfg.items():
        opt[k] = v
    return opt


def get_subset_indices(dataset, mask, ratio=2):
    n = ratio + 1
    has_attr = np.sum(np.logical_and(np.array(dataset.attr), mask), axis=1) >= 1
    # Indexes with the attribute
    attr_samples = np.argwhere(has_attr == 1)
    # Indexes withou the attribute
    no_attr_samples = np.argwhere(has_attr == 0)
    n_attr = np.sum(has_attr)
    return np.concatenate((attr_samples, no_attr_samples[0:(n - 1) * n_attr])).astype(np.int).squeeze()


def get_mask(attributes, attribute_names):
    attribute_names = np.array(attribute_names)
    mask = np.zeros(*attribute_names.shape).astype(int)
    for att in attributes:
        mask = np.logical_or(mask, attribute_names == att)
    return mask.astype(int)


def build_filename(ctx, included_opts=('model', 'attributes')):
    opt = ctx.opt
    o = {k: opt[k] for k in included_opts if k in opt}
    if "attributes" in o.keys():
        o['attributes'] = ",".join(o['attributes'])
    t = time.strftime('%b_%d_%H_%M_%S')
    opt['time'] = t
    opt['filename'] = f"({t})_opts_{json.dumps(o, sort_keys=True, separators=(',', ':'))}"


def get_dataset(dataset, root, normalize=True, image_size=128):
    from introvac.modules.datasets import CelebA
    trans = transforms.ToTensor()
    if dataset.lower() == 'celeba':
        trans = transforms.Compose([transforms.Resize(image_size), trans])
        train_set = CelebA(root, "train", transform=trans, download=True)
        test_set = CelebA(root, "test", transform=trans, download=True)
        validation_set = CelebA(root, "valid", transform=trans, download=True)
        return train_set, test_set, validation_set, np.array(train_set.attr_names), None, None
    else:
        raise Exception("Dataset not available")


def load_opts(ctx, filename):
    data = torch.load(filename)
    ctx.opt = data['opt']
    ctx.iter = data['iter'] + ctx.opt['log_freq']
    ctx.epoch = data['epoch'] + 1


def save_checkpoint(ctx, best=False):
    opt = ctx.opt
    folder = opt['save_folder']
    filename = os.path.join(folder, 'checkpoint.pkl')
    torch.save(dict(opt=opt, iter=ctx.iter, epoch=ctx.epoch,
                    encoder=ctx.encoder.state_dict(),
                    decoder=ctx.decoder.state_dict(),
                    classifier=ctx.classifier.state_dict() if hasattr(ctx, 'classifier') else None,
                    optimizer_enc=ctx.optimizer_enc.state_dict(),
                    optimizer_dec=ctx.optimizer_dec.state_dict(),
                    optimizer_class=ctx.optimizer_class.state_dict() if hasattr(ctx, 'optimizer_class') else None),
               filename)
    if best:
        filename = os.path.join(folder, 'checkpoint_best.pkl')
        torch.save(dict(opt=opt, iter=ctx.iter, epoch=ctx.epoch,
                        encoder=ctx.encoder.state_dict(),
                        decoder=ctx.decoder.state_dict(),
                        classifier=ctx.classifier.state_dict() if hasattr(ctx, 'classifier') else None,
                        optimizer_enc=ctx.optimizer_enc.state_dict(),
                        optimizer_dec=ctx.optimizer_dec.state_dict(),
                        optimizer_class=ctx.optimizer_class.state_dict() if hasattr(ctx, 'optimizer_class') else None),
                   filename)


def load_models(ctx, filename, optimizer=True):
    data = torch.load(filename)
    ctx.encoder.load_state_dict(data['encoder'])
    ctx.decoder.load_state_dict(data['decoder'])
    if hasattr(ctx, 'classifier'):
        ctx.classifier.load_state_dict(data['classifier'])
    if optimizer:
        ctx.optimizer.load_state_dict(data['optimizer'])


def kl_divergence(p):
    (mu, var) = p
    # We average over the latent dimension so it's invariant to the size
    return - 0.5 * torch.mean(torch.mean(1 + torch.log(var) - mu**2 - var))
