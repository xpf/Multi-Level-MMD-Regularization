from opts import get_opts
from utils.utils import get_name
from utils.settings import DATASETTINGS
from models import build_model
from datasets import build_transform, build_data
from triggers import build_trigger
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import tqdm, torch, os
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def visualize(opts):
    name = get_name(opts)
    print('visualize', name)
    if not os.path.isdir(opts.figure_path): os.mkdir(opts.figure_path)
    DSET = DATASETTINGS[opts.data_name]

    model = build_model(opts.model_name, DSET['num_classes']).to(opts.device).eval()
    model.load_state_dict(torch.load(os.path.join(opts.weight_path, '{}.pt'.format(name)), map_location=opts.device))
    val_transform = build_transform(False, DSET['img_size'], DSET['crop_pad'], DSET['flip'])
    trigger = build_trigger(opts.trigger, opts.prob, DSET['img_size'])
    val_data = build_data(opts.data_name, opts.data_path, False, val_transform, trigger)
    val_loader = DataLoader(val_data, DSET['batch_size'], shuffle=False, num_workers=2)

    features_out = []
    levels = [4, 6, 8]

    def hook(module, feature_in, feature_out):
        features_out.append(feature_out.view(feature_out.shape[0], -1).detach())

    count = 0
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            count = count + 1
            if count in levels:
                module.register_forward_hook(hook=hook)

    trigger.set_prob(prob=0.0)
    correct, total = 0, 0
    desc = 'val   - acc: {:.3f}'
    features_nor = [[] for _ in levels]
    run_tqdm = tqdm.tqdm(val_loader, desc=desc.format(0, 0), disable=opts.disable_bar)
    for x, y, _, _ in run_tqdm:
        x, y = x.to(opts.device), y.to(opts.device)
        with torch.no_grad():
            p = model(x)
        _, p = torch.max(p, dim=1)
        for i in range(len(levels)):
            features_nor[i].append(features_out[i][p == 0, :].cpu().numpy())
        features_out = []
        correct += (p == y).sum().item()
        total += y.shape[0]
        run_tqdm.set_description(desc.format(correct / total))
    val_acc = correct / total

    if opts.disable_bar:
        print(desc.format(val_acc))

    trigger.set_prob(prob=1.0)
    correct, total = 0, 0
    desc = 'back  - acc: {:.3f}'
    features_back = [[] for _ in levels]
    run_tqdm = tqdm.tqdm(val_loader, desc=desc.format(0, 0), disable=opts.disable_bar)
    for x, y, b, s in run_tqdm:
        x, y, b, s = x.to(opts.device), y.to(opts.device), b.to(opts.device), s.to(opts.device)
        ind_back = torch.where(b == 1)[0]
        x, y, b, s = x[ind_back, :, :, :], y[ind_back], b[ind_back], s[ind_back]
        if x.shape[0] == 0: continue
        with torch.no_grad():
            p = model(x)
        _, p = torch.max(p, dim=1)
        for i in range(len(levels)):
            features_back[i].append(features_out[i][p == 0, :].cpu().numpy())
        features_out = []
        correct += (p == y).sum().item()
        total += y.shape[0]
        run_tqdm.set_description(desc.format(correct / total))
    back_acc = correct / total

    if opts.disable_bar:
        print(desc.format(back_acc))

    visual = PCA(n_components=2)
    plt.figure(figsize=(3.2 * len(levels), 2.4))
    for i, (feature_nor, feature_back) in enumerate(zip(features_nor, features_back)):
        feature_nor = np.concatenate(feature_nor, axis=0)[:100]
        feature_back = np.concatenate(feature_back, axis=0)[:100]
        feature_total = np.concatenate([feature_nor, feature_back], axis=0)
        feature_vis = visual.fit_transform(feature_total)
        plt.subplot(1, len(levels), i + 1)
        plt.scatter(feature_vis[:100, 0], feature_vis[:100, 1])
        plt.scatter(feature_vis[100:, 0], feature_vis[100:, 1])
        if opts.mlmmdr_lamb > 0 and opts.mlmmdr_layer == 'all':
            plt.title(r's{}, ML-MMDR, $\lambda$={}'.format(i + 1, opts.mlmmdr_lamb))
        elif opts.mlmmdr_lamb > 0 and opts.mlmmdr_layer == 'last':
            plt.title(r's{}, SL-MMDR, $\lambda$={}'.format(i + 1, opts.mlmmdr_lamb))
        else:
            plt.title(r's{}, RBT'.format(i + 1))
    plt.tight_layout()
    plt.savefig(os.path.join(opts.figure_path, '{}.png'.format(name)))
    plt.show()


if __name__ == '__main__':
    opts = get_opts()
    opts.mlmmdr_layer = 'all'
    visualize(opts)
