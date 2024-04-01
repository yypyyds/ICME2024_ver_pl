import torch
from torch.nn import functional as F
import numpy as np

def linear_rampup(current, rampup_length):
    """_summary_

    Args:
        rampup_length (_type_): config.max_epoch
    """
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)
    

class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, lambda_u, max_epochs):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)

        return Lx, Lu, lambda_u * linear_rampup(epoch, max_epochs)
    
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger(object):
    '''Save training process to log file with simple plot function.'''
    def __init__(self, fpath, title=None, resume=False): 
        self.file = None
        self.resume = resume
        self.title = '' if title == None else title
        if fpath is not None:
            if resume: 
                self.file = open(fpath, 'r') 
                name = self.file.readline()
                self.names = name.rstrip().split('\t\t')
                self.numbers = {}
                for _, name in enumerate(self.names):
                    self.numbers[name] = []

                for numbers in self.file:
                    numbers = numbers.rstrip().split('\t\t')
                    for i in range(0, len(numbers)):
                        self.numbers[self.names[i]].append(numbers[i])
                self.file.close()
                self.file = open(fpath, 'a')  
            else:
                self.file = open(fpath, 'w')

    def set_names(self, names):
        if self.resume: 
            pass
        # initialize numbers as empty list
        self.numbers = {}
        self.names = names
        for _, name in enumerate(self.names):
            self.file.write(name)
            self.file.write('\t')
            self.numbers[name] = []
        self.file.write('\n')
        self.file.flush()


    def append(self, numbers):
        assert len(self.names) == len(numbers), 'Numbers do not match names'
        for index, num in enumerate(numbers):
            self.file.write("{0:.6f}".format(num))
            self.file.write('\t')
            self.numbers[self.names[index]].append(num)
        self.file.write('\n')
        self.file.flush()

    def plot(self, names=None):   
        names = self.names if names == None else names
        numbers = self.numbers
        for _, name in enumerate(names):
            x = np.arange(len(numbers[name]))
            plt.plot(x, np.asarray(numbers[name]))
        plt.legend([self.title + '(' + name + ')' for name in names])
        plt.grid(True)

    def close(self):
        if self.file is not None:
            self.file.close()


import torch

def random_temporal_shift(mel_spectrogram, a):
    batch_size, num_channels, T, F = mel_spectrogram.shape

    # 计算要移动的时间步数
    shift_steps = int(a * T)

    # 将前半段和后半段切分出来
    first_half = mel_spectrogram[:, :, :shift_steps, :]
    second_half = mel_spectrogram[:, :, shift_steps:, :]

    # 将前半段移动到后半段
    shifted_mel_spectrogram = torch.cat((second_half, first_half), dim=2)

    return shifted_mel_spectrogram


class_2_index = {
    "Bus": 0,
    "Airport": 1,
    "Metro": 2,
    "Restaurant": 3,
    "Shopping mall": 4,
    "Public square": 5,
    "Urban park": 6,
    "Traffic street": 7,
    "Construction site": 8,
    "Bar": 9,
}

indoor = ["Airport", "Restaurant", "Shopping mall", "Bar"]
outdoor = ["Public square", "Urban park", "Traffic street"]
vehicle = ["Bus", "Metro", "Construction site"]

def get_threeclass_mask():
    mask = torch.zeros((3,10))
    for name in indoor:
        mask[0][class_2_index[name]] = 1
    for name in outdoor:
        mask[1][class_2_index[name]] = 1
    for name in vehicle:
        mask[2][class_2_index[name]] = 1
    
    return mask.unsqueeze(0)

def threeclass_loss(label, preds, lossfunc):
    batch = label.shape[0]
    mask = get_threeclass_mask()
    mask = mask.repeat(batch, 1, 1).cuda()
    res1 = torch.matmul(mask, label.unsqueeze(-1)).squeeze()
    res2 = torch.matmul(mask, preds.unsqueeze(-1)).squeeze()
    loss = lossfunc(res1, res2)
    return loss


import torch.nn.functional as F
# z.shape = [8, 768, 128, 27]
def DG_loss(z: torch.Tensor, targets_x, tou):
    z = torch.sum(z, dim=(2, 3)).cuda()
    batch_size = z.shape[0]
    dimension_size = z.shape[1]

    # 对张量进行 L2 归一化
    latent_z = F.normalize(z, p=2, dim=1)
    classes = [[] for _ in range(10)]   # 放的是每个类别在batch中对应的序号
    for idx in range(batch_size):
        classes[targets_x[idx]].append(idx)
    
    # Learning domain-invariant representation, CID loss(classwise instance discrimination loss)
    CID_loss = torch.sum(torch.Tensor([0])).cuda()
    for idx_i in range(batch_size):
        up = 0
        down = 0
        if len(classes[targets_x[idx_i]]) < 2:
            continue
        for idx_j in range(batch_size):
            if idx_j == idx_i:
                continue
            else:
                tmp = torch.dot(latent_z[idx_i], latent_z[idx_j]) / tou
                down += torch.exp(tmp)
                if idx_j in classes[targets_x[idx_i]]:
                    up += torch.exp(tmp)
            
        l_i = -torch.log(up/down)
        CID_loss += l_i

    # 维度间相关性
    mean_z = z.mean(dim=0, keepdim=True)
    std_z = z.std(dim=0, keepdim=True)
    latent_z = (z - mean_z)/std_z
    L_decorre = torch.sum(torch.Tensor([0])).cuda()
    for idx in range(batch_size):
        G_dd = torch.mul(latent_z[idx].unsqueeze(1), latent_z[idx])
        tmp = torch.sum(torch.mul(latent_z[idx], latent_z[idx]))
        G_dd = torch.sum(G_dd) - tmp
        L_decorre += G_dd * G_dd

    # 维度内相关性
    maxval, _ = torch.max(z, dim=0, keepdim=True)
    latent_z = z / maxval
    std_z = latent_z.std(dim = 0)
    L_uniform = -torch.sum(std_z)/dimension_size

    return CID_loss, L_decorre, L_uniform

def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets

def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res