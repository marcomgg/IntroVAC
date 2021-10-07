import torch
import torch.nn as nn
import numpy as np
import seaborn as sns
import introvac.modules.utils as ut
from introvac.modules.models import get_model
from torch.utils.data import DataLoader
from torchnet.meter import AverageValueMeter
from sklearn.metrics import multilabel_confusion_matrix
import introvac.modules.log as log
from tqdm import tqdm
torch.autograd.set_detect_anomaly(True)
sns.set_style("darkgrid")


def compute_loss_vac(ctx, x, y, evaluate=False):
    mse = nn.L1Loss()
    ce = nn.BCEWithLogitsLoss()
    y = torch.cat((y, torch.zeros((y.shape[0], 1)).to(ctx.device)), dim=1)
    encoder, decoder, classifier = ctx.encoder, ctx.decoder, ctx.classifier

    mu_z, log_var_z = encoder(x)
    var_z = torch.exp(log_var_z)
    std_z = torch.exp(0.5 * log_var_z)
    noise = torch.randn_like(mu_z) * std_z
    sample = mu_z + noise

    if evaluate:
        sample = mu_z

    x_hat, y_hat = decoder(sample), classifier(sample)

    loss_reconst = mse(x_hat, x)
    loss_class = ce(y_hat, y)
    loss_kl = ut.kl_divergence((mu_z, var_z))

    return loss_class, loss_reconst, loss_kl, x_hat, y_hat, sample


def step_intro(ctx, x, y):
    optimizer_enc = ctx.optimizer_enc
    optimizer_dec = ctx.optimizer_dec
    optimizer_class = ctx.optimizer_class
    encoder, decoder, classifier = ctx.encoder, ctx.decoder, ctx.classifier
    beta_cl, beta_rec, beta_kl, beta_e, beta_d = ctx.opt['betas']
    ce = nn.BCEWithLogitsLoss()

    out = compute_loss_vac(ctx, x, y)
    loss_class, loss_reconst, loss_kl, x_hat, y_hat, sample = out
    loss = beta_cl * loss_class + beta_rec * loss_reconst + beta_kl * loss_kl

    optimizer_enc.zero_grad()
    optimizer_dec.zero_grad()
    optimizer_class.zero_grad()
    loss.backward(retain_graph=True)

    mu_hat_d, _ = encoder(x_hat.detach())
    y_hat_d = classifier(mu_hat_d)

    z_gen = torch.randn_like(mu_hat_d)
    x_gen = decoder(z_gen)
    mu_gen, _ = encoder(x_gen.detach())
    y_hat_gen = classifier(mu_gen)

    y_e = torch.ones((y.shape[0], 1)).to(ctx.device)
    loss_e = ce(y_hat_d[:, -1:], y_e) * beta_e + beta_e * ce(y_hat_gen[:, -1:], y_e)
    loss_e.backward()
    optimizer_enc.step()
    optimizer_class.step()

    x_hat = decoder(sample.detach())
    x_gen = decoder(z_gen)
    mu_hat, _ = encoder(x_hat)
    mu_gen, _ = encoder(x_gen)
    y_hat_gen = classifier(mu_gen)
    y_hat_r = classifier(mu_hat)
    y_d = torch.cat((y, 1 - y_e), dim=1)
    loss_d = ce(y_hat_r, y_d) * beta_d + beta_d * ce(y_hat_gen[:, -1:], 1 - y_e)
    loss_d.backward()
    optimizer_dec.step()
    return loss_class, loss_reconst, loss_kl, loss_d, loss_e, x_hat


def step_vac(ctx, x, y):
    optimizer_enc = ctx.optimizer_enc
    optimizer_dec = ctx.optimizer_dec
    optimizer_class = ctx.optimizer_class
    beta_cl, beta_rec, beta_kl, _, _ = ctx.opt['betas']

    out = compute_loss_vac(ctx, x, y)
    loss_class, loss_reconst, loss_kl, x_hat, y_hat, sample = out
    loss = beta_cl * loss_class + beta_rec * loss_reconst + beta_kl * loss_kl

    optimizer_enc.zero_grad()
    optimizer_dec.zero_grad()
    optimizer_class.zero_grad()
    loss.backward()
    optimizer_class.step()
    optimizer_enc.step()
    optimizer_dec.step()
    return loss_class, loss_reconst, loss_kl, x_hat


@log.batch_log
def train_batch(ctx, x, y):
    pretrain = ctx.opt['pretrain']
    images, scalars = {}, {}

    if ctx.epoch >= pretrain:
        if ctx.epoch == pretrain:
            init_optim(ctx)
        loss_class, loss_reconst, loss_kl, loss_d, loss_e, x_hat = step_intro(ctx, x, y)
        scalars = {'Loss d': loss_d.item(),
                   'Loss e': loss_e.item()}

    else:
        loss_class, loss_reconst, loss_kl, x_hat = step_vac(ctx, x, y)

    vectors = ut.get_gradients((ctx.encoder, ctx.decoder, ctx.classifier),
                               ("Enc", "Dec", "Class")) if ctx.opt['log_gradients'] else {}

    scalars.update({'Loss class': loss_class.item(),
                    'Loss reconst': loss_reconst.item(),
                    'Loss kl': loss_kl.item()})

    images.update({'Reconstruction': ut.to_tf_images(x_hat[0:1]),
                   'Real': ut.to_tf_images(x[0:1])})

    return ctx.iter, scalars, vectors, images


def train_epoch(ctx, train_loader):
    opt = ctx.opt
    for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
        x = data[0].to(ctx.device)
        y = ut.get_label(ctx.mask, data[1], binary=opt['binary']).to(ctx.device)
        train_batch(ctx, x, y)
        ctx.iter += 1


def train(ctx, train_loader, val_loader):
    epochs = ctx.opt['epochs']
    start_epoch = ctx.epoch
    sched_cl = torch.optim.lr_scheduler.MultiStepLR(ctx.optimizer_class, ctx.opt['lrs'], gamma=ctx.opt['gamma'])
    sched_e = torch.optim.lr_scheduler.MultiStepLR(ctx.optimizer_enc, ctx.opt['lrs'], gamma=ctx.opt['gamma'])
    sched_d = torch.optim.lr_scheduler.MultiStepLR(ctx.optimizer_dec, ctx.opt['lrs'], gamma=ctx.opt['gamma'])
    best_reconst = 1000
    best = False
    for e in range(start_epoch, epochs):
        print(f'Epoch {e} / {epochs}')
        ctx.epoch = e

        train_epoch(ctx, train_loader)

        if e % log.epoch_log_freq == 0:
            _, s, _, _ = evaluate(ctx, val_loader, True)
            reconst = s['Val reconst']
            if reconst < best_reconst:
                best_reconst = reconst
                best = True

        sched_cl.step()
        sched_e.step()
        sched_d.step()
        ut.save_checkpoint(ctx, best)
        best = False


@log.epoch_log
def evaluate(ctx, test_loader, validation=True):
    name = 'Val' if validation else 'Test'
    avg = AverageValueMeter()
    avg_class = AverageValueMeter()
    x_wrong, x_wrong_rec = [], []
    binary = ctx.opt['binary']
    encoder, decoder, classifier = ctx.encoder, ctx.decoder, ctx.classifier
    encoder.eval()
    decoder.eval()
    classifier.eval()
    matrix = None
    for data in test_loader:
        x = data[0].to(ctx.device)

        if hasattr(ctx, 'mask'):
            y = ut.get_label(ctx.mask, data[1], binary=binary).to(ctx.device)
        else:
            y = data[1].to(ctx.device)[:, None].float()

        with torch.no_grad():
            loss_class, loss_reconst, loss_kl, x_hat, y_hat, _ = compute_loss_vac(ctx, x, y, True)
            y = torch.cat((y, torch.zeros((y.shape[0], 1)).to(ctx.device)), dim=1)
            y, y_hat = y.cpu().numpy(), torch.round(torch.sigmoid(y_hat)).cpu().numpy()
            m = multilabel_confusion_matrix(y, y_hat).astype(np.float)
            if matrix is None:
                matrix = m
            else:
                matrix = np.add(matrix, m)

            error = ((y_hat != y).sum(axis=1) > 0).squeeze()

            if error.sum() > 0:
                x_wrong.append(x[error.astype(np.bool)][0:1])
                x_wrong_rec.append(x_hat[error.astype(np.bool)][0:1])
            avg.add(loss_reconst.item())
            avg_class.add(loss_class.item())

    n_attr = len(ctx.label_names) + 1
    matrix = matrix[-n_attr:] if n_attr < matrix.shape[0] else matrix
    accuracy = ut.accuracy(matrix)
    grid = ut.create_confusion_grid(matrix, ctx.label_names + ['Fake'])

    scalars = {f'{name} ce': avg_class.value()[0],
               f'{name} error': 100 - accuracy,
               f'{name} reconst': avg.value()[0]}
    images = {f'{name} Confusion': grid[None, :, :, :],
              f'{name} Error': ut.to_tf_images(x_wrong[0]),
              f'{name} Error rec': ut.to_tf_images(x_wrong_rec[0])}

    print(f"[{name}] Loss class: {avg_class.value()[0]:.3f}, Error: {100-accuracy:.3f}, Loss reconst: {avg.value()[0]:.3f}")

    encoder.train()
    decoder.train()
    classifier.train()

    return ctx.epoch, scalars, {}, images


def init_optim(ctx):
    opt = ctx.opt
    ctx.optimizer_enc = torch.optim.Adam(ctx.encoder.parameters(), lr=opt['lr'], weight_decay=opt['wd'])
    ctx.optimizer_dec = torch.optim.Adam(ctx.decoder.parameters(), lr=opt['lr'], weight_decay=opt['wd'])
    ctx.optimizer_class = torch.optim.Adam(list(ctx.classifier.parameters()), lr=opt['lr'], weight_decay=opt['wd_class'])


def run(ctx):
    opt = ctx.opt
    ratio = opt['ratio']

    attributes = opt['attributes']
    opt['num_classes'] = 2 if opt['binary'] else len(attributes) + 1
    opt['num_classes'] += 1

    train_set, test_set, validation_set, names, _, _ = ut.get_dataset(opt['dataset'], opt['dataset_root'], False)

    mask = ut.get_mask(attributes, train_set.attr_names)
    ctx.mask = mask
    names = names[mask.astype(bool)].tolist()

    # Get the indexes to create the dataset as specified in the paper in particular for the Eye
    # dataset we use ratio = 2 that  means that the images without attribute are two times the one
    # with the attribute, for the Facial we use ratio = 1 and same for the EyeFacial

    train_idx = ut.get_subset_indices(train_set, mask, ratio=ratio)
    train_set = torch.utils.data.Subset(train_set, train_idx)

    val_idx = ut.get_subset_indices(validation_set, mask, ratio=ratio)
    validation_set = torch.utils.data.Subset(validation_set, val_idx)

    ctx.label_names = [','.join(names)] if opt['binary'] else names
    train_loader = DataLoader(train_set, batch_size=opt['batch_size'], shuffle=True, num_workers=8)
    val_loader = DataLoader(validation_set, batch_size=opt['batch_size'], shuffle=True)

    print(f"Ratio: {ratio}, "
          f"Train length: {len(train_set)}, "
          f"Val length: {len(validation_set)}")

    ctx.encoder = nn.DataParallel(get_model("Encoder")(opt=opt).to(ctx.device))
    ctx.decoder = nn.DataParallel(get_model("Decoder")(opt=opt).to(ctx.device))
    ctx.classifier = nn.DataParallel(get_model(opt['classifier'])(opt=opt).to(ctx.device))

    init_optim(ctx)
    train(ctx, train_loader, val_loader)
