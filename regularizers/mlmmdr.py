import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, grad


class MultiLevelMMDReg(object):
    def __init__(self, model, levels):
        super(MultiLevelMMDReg, self).__init__()
        self.features_out = []
        self.hook_model(model, levels)

    def hook_model(self, model, levels):
        def hook(module, feature_in, feature_out):
            self.features_out.append(feature_out)

        count = 0
        for module in model.modules():
            if isinstance(module, nn.ReLU):
                count = count + 1
                if count in levels:
                    module.register_forward_hook(hook=hook)

    def calculate(self, ind_nor, ind_back):
        if min(ind_nor.shape[0], ind_back.shape[0]) == 0:
            return 0
        loss = 0
        for feature in self.features_out:
            feature = feature.view(feature.shape[0], -1)
            loss = loss + self.mmd(feature[ind_nor, :], feature[ind_back, :])
        self.features_out = []
        return loss

    def mmd(self, source, target):
        source_size = source.shape[0]
        kernel_mul, kernel_num = 2.0, 5
        n_samples = int(source.shape[0]) + int(target.shape[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / temp) for temp in bandwidth_list]
        kernels = sum(kernel_val)
        XX = torch.mean(kernels[:source_size, :source_size])
        YY = torch.mean(kernels[source_size:, source_size:])
        XY = torch.mean(kernels[:source_size, source_size:])
        YX = torch.mean(kernels[source_size:, :source_size])
        loss = torch.mean(XX + YY - XY - YX)
        return loss
