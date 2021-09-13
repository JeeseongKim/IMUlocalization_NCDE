from torch.utils.data import DataLoader
import time
from utils import *

from tqdm import tqdm

from model.IMUTransformer import *
from loss import *
from torchviz import make_dot

torch.multiprocessing.set_start_method('spawn', force=True)

import visdom
import utils_dataloader_KITTI

import os
from torch.utils.data import DataLoader

from utils import *
from torchdiffeq import odeint_adjoint as odeint

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('nn.Conv2d') != -1:
        #torch.nn.init.kaiming_normal_(m.weight)
        torch.nn.init.xavier_normal_(m.weight)
    elif classname.find('nn.Linear') != -1:
       torch.nn.init.xavier_normal_(m.weight)
    #torch.nn.init.xavier_normal_(m.weight)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def norm(dim):
    return nn.GroupNorm(min(32, dim), dim)

class ConcatConv2d(nn.Module):

    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
        super(ConcatConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in + 1, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )

    def forward(self, t, x):
        tt = torch.ones_like(x[:, :1, :, :]) * t
        ttx = torch.cat([tt, x], 1)
        return self._layer(ttx)

class ODEfunc(nn.Module):

    def __init__(self, dim):
        super(ODEfunc, self).__init__()
        self.norm1 = norm(dim)
        self.relu = nn.ReLU(inplace=True)
        self.elu = nn.ELU(inplace=True)
        self.conv1 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm2 = norm(dim)
        self.conv2 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm3 = norm(dim)
        self.nfe = 0

    def forward(self, t, x):
        #print(x)
        self.nfe += 1
        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(t, out)
        out = self.norm2(out)
        out = self.elu(out)
        out = self.conv2(t, out)
        out = self.norm3(out)
        return out

class ODEfunc_iacc_gyro(nn.Module):

    def __init__(self, dim):
        super(ODEfunc_iacc_gyro, self).__init__()
        self.norm1 = norm(dim)
        self.relu = nn.ReLU(inplace=True)
        self.elu = nn.ELU(inplace=True)
        self.conv1 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm2 = norm(dim)
        self.conv2 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm3 = norm(dim)
        self.nfe = 0

    def forward(self, t, x):
        #print(x)
        self.nfe += 1
        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(t, out)
        out = self.norm2(out)
        out = self.elu(out)
        out = self.conv2(t, out)
        out = self.norm3(out)
        return out

class ODEfunc_acc2(nn.Module):

    def __init__(self, dim):
        super(ODEfunc_acc2, self).__init__()
        self.norm1 = norm(dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm2 = norm(dim)
        self.conv2 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm3 = norm(dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        #out = self.norm1(x)
        #out = self.relu(x)
        #out = self.relu(out)
        #out = self.conv1(t, out)
        out = self.conv1(t, x)
        #out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(t, out)
        #out = self.norm3(out)
        return out

class ODEfunc_gyro(nn.Module):

    def __init__(self, dim):
        super(ODEfunc_gyro, self).__init__()
        self.norm1 = norm(dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm2 = norm(dim)
        self.conv2 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm3 = norm(dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        #out = self.norm1(x)
        #out = self.relu(out)
        #out = self.relu(x)
        #out = self.conv1(t, out)
        out = self.conv1(t, x)
        #out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(t, out)
        #out = self.norm3(out)
        return out

class ODEBlock(nn.Module):

    def __init__(self, odefunc):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, 1]).float()

    def forward(self, x):
        #time = t.float()
        self.integration_time = self.integration_time.type_as(x)
        #out = odeint(self.odefunc, x, self.integration_time, rtol=1e-7, atol=1e-9, method='rk4')
        #out = odeint(self.odefunc, x, self.integration_time, rtol=1e-7, atol=1e-9)
        #out = odeint(self.odefunc, x, self.integration_time, rtol=1e-7, atol=1e-9, method='implicit_adams')
        #out = odeint(self.odefunc, x, self.integration_time, rtol=1e-7, atol=1e-9, method='dopri5')
        out = odeint(self.odefunc, x, self.integration_time, rtol=1e-3, atol=1e-4, method='implicit_adams')
        #out = odeint(self.odefunc, x, time, rtol=1e-7, atol=1e-9, method='implicit adams')
        return out[1]

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value

class ODEBlock_s(nn.Module):

    def __init__(self, odefunc):
        super(ODEBlock_s, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, 1]).float()

    def forward(self, x):
        #time = t.float()
        self.integration_time = self.integration_time.type_as(x)
        #out = odeint(self.odefunc, x, self.integration_time, rtol=1e-7, atol=1e-9, method='rk4')
        #out = odeint(self.odefunc, x, self.integration_time, rtol=1e-7, atol=1e-9)
        #out = odeint(self.odefunc, x, self.integration_time, rtol=1.5e-8, atol=1.5e-8, method='rk4')
        #out = odeint(self.odefunc, x, self.integration_time, rtol=1e-3, atol=1.5e-8, method='rk4')
        #out = odeint(self.odefunc, x, self.integration_time, rtol=1e-7, atol=1e-9, method='dopri5')
        out = odeint(self.odefunc, x, self.integration_time, rtol=1e-7, atol=1e-9, method='implicit_adams')
        #out = odeint(self.odefunc, x, time, rtol=1e-7, atol=1e-9, method='implicit adams')
        return out[1]

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value

class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)

class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val

def one_hot(x, K):
    return np.array(x[:, None] == np.arange(K)[None, :], dtype=int)

class ind_fc_layers_acc(nn.Module):
    def __init__(self):
        super(ind_fc_layers_acc, self).__init__()
        self.norm = norm(64)
        self.norm_2 = torch.nn.BatchNorm1d(100)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = Flatten()
        self.relu = nn.ReLU()
        self.conv1d_1 = torch.nn.Conv1d(64, 64, kernel_size=3, padding=1, dilation=1, stride=1)
        self.conv1d_2 = torch.nn.Conv1d(64, 16, kernel_size=3, padding=1, dilation=1, stride=1)
        self.conv1d_3 = torch.nn.Conv1d(16, 2, kernel_size=3, padding=1, dilation=1, stride=1)
        #self.conv1d_2 = nn.Conv1d(1024, 256, kernel_size=3, padding=1, dilation=1, stride=1)
        #self.conv1d_3 = nn.Conv1d(256, 64, kernel_size=3, padding=1, dilation=1, stride=1)
        self.linear1 = nn.Linear(100, 32)
        self.linear2 = nn.Linear(32, 8)
        self.linear3 = nn.Linear(8, 2)

    def forward(self, x):
        x = self.norm(x)
        #x = self.pool(x)
        #x = self.flatten(x)

        x = self.relu(self.conv1d_1(x.squeeze(2)))
        x = self.relu(self.conv1d_2(x))
        x = self.relu(self.conv1d_3(x))

        x = self.norm_2(x.transpose(1, 2)).transpose(1, 2)

        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.linear3(x).squeeze(2)

        return x

class ind_fc_layers_gyro(nn.Module):
    def __init__(self):
        super(ind_fc_layers_gyro, self).__init__()
        self.norm = norm(64)
        self.norm_2 = torch.nn.BatchNorm1d(100)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = Flatten()
        self.relu = nn.ReLU()
        self.conv1d_1 = torch.nn.Conv1d(64, 64, kernel_size=3, padding=1, dilation=1, stride=1)
        self.conv1d_2 = torch.nn.Conv1d(64, 16, kernel_size=3, padding=1, dilation=1, stride=1)
        self.conv1d_3 = torch.nn.Conv1d(16, 2, kernel_size=3, padding=1, dilation=1, stride=1)
        #self.conv1d_2 = nn.Conv1d(1024, 256, kernel_size=3, padding=1, dilation=1, stride=1)
        #self.conv1d_3 = nn.Conv1d(256, 64, kernel_size=3, padding=1, dilation=1, stride=1)
        self.linear1 = nn.Linear(100, 32)
        self.linear2 = nn.Linear(32, 8)
        self.linear3 = nn.Linear(8, 1)

    def forward(self, x):
        x = self.norm(x)
        #x = self.pool(x)
        #x = self.flatten(x)

        x = self.relu(self.conv1d_1(x.squeeze(2)))
        x = self.relu(self.conv1d_2(x))
        x = self.relu(self.conv1d_3(x))

        x = self.norm_2(x.transpose(1, 2)).transpose(1, 2)

        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.linear3(x).squeeze(2)

        return x

class ind_fc_layers(nn.Module):
    def __init__(self):
        super(ind_fc_layers, self).__init__()
        self.norm = norm(64)
        #self.norm_2 = torch.nn.BatchNorm1d(100)
        #self.pool = nn.AdaptiveAvgPool2d((1, 1))
        #self.flatten = Flatten()
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.conv2d_1 = torch.nn.Conv2d(64, 64, kernel_size=3, padding=1, dilation=1, stride=1)
        self.conv2d_2 = torch.nn.Conv2d(64, 16, kernel_size=3, padding=1, dilation=1, stride=1)
        self.conv2d_3 = torch.nn.Conv2d(16, 1, kernel_size=3, padding=1, dilation=1, stride=1)
        #self.conv1d_2 = nn.Conv1d(1024, 256, kernel_size=3, padding=1, dilation=1, stride=1)
        #self.conv1d_3 = nn.Conv1d(256, 64, kernel_size=3, padding=1, dilation=1, stride=1)
        self.linear1 = nn.Linear(100, 32)
        self.linear2 = nn.Linear(32, 8)
        self.linear3 = nn.Linear(8, 1)

        self.bn1 = torch.nn.BatchNorm2d(64)
        self.bn2 = torch.nn.BatchNorm2d(16)

    def forward(self, x):
        x = self.norm(x)
        #x = self.pool(x)
        #x = self.flatten(x)

        x = self.conv2d_1(x)
        x = self.bn1(x)
        x = self.elu(x)

        x = self.conv2d_2(x)
        x = self.bn2(x)
        x = self.elu(x)

        x = self.conv2d_3(x)
        x = self.elu(x)

        x = self.linear1(x)
        x = self.elu(x)

        x = self.linear2(x)
        x = self.elu(x)

        x = self.linear3(x)

        return x.squeeze(3)


class ind_fc_layers_small(nn.Module):
    def __init__(self):
        super(ind_fc_layers_small, self).__init__()
        self.norm = norm(256)
        #self.norm_2 = torch.nn.BatchNorm1d(100)
        #self.pool = nn.AdaptiveAvgPool2d((2, 1)).cuda()
        #self.flatten = Flatten()
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.conv2d_1 = torch.nn.Conv2d(256, 64, kernel_size=3, padding=1, dilation=1, stride=1)
        self.conv2d_2 = torch.nn.Conv2d(64, 16, kernel_size=3, padding=1, dilation=1, stride=1)
        self.conv2d_3 = torch.nn.Conv2d(16, 1, kernel_size=3, padding=1, dilation=1, stride=1)

        #self.linear0 = nn.Linear(1, 1)
        self.linear1 = nn.Linear(100, 64)
        self.linear2 = nn.Linear(64, 3)
        self.linear3 = nn.Linear(3, 1)

        self.bn0 = torch.nn.BatchNorm2d(256)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.bn2 = torch.nn.BatchNorm2d(16)
        self.bn3 = torch.nn.BatchNorm2d(1)

    def forward(self, x):
        x = self.norm(x)
        #x = self.pool(x)
        x = self.bn0(x)
        x = self.relu(x)

        #x = self.flatten(x)

        x = self.conv2d_1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2d_2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv2d_3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.linear1(x)
        x = self.relu(x)

        x = self.linear2(x)
        x = self.relu(x)

        x = self.linear3(x)

        return x.squeeze(3)

class NODE_acc_2_pos(torch.nn.Module):
    def __init__(self):
        super(NODE_acc_2_pos, self).__init__()
        #self.conv1d_2_32 = torch.nn.Conv1d(2, 32, kernel_size=3, padding=1, dilation=1, stride=1)
        self.conv1d_4_32 = torch.nn.Conv1d(4, 32, kernel_size=3, padding=1, dilation=1, stride=1)
        #torch.nn.init.xavier_uniform_(self.conv1d_2_32.weight)
        #torch.nn.init.xavier_normal_(self.conv1d_2_32.weight)

        self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.ReLU()
        self.elu = torch.nn.ELU()
        #self.conv2d_32_64 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1, dilation=1, stride=1)
        self.conv1d_32_64 = torch.nn.Conv1d(32, 64, kernel_size=3, padding=1, dilation=1, stride=1)

        self.feature_layers = [ODEBlock(ODEfunc(64))][0]

        #self.batchnorm_32 = torch.nn.BatchNorm2d(32)
        self.bn1 = torch.nn.BatchNorm1d(32)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.bn3 = torch.nn.BatchNorm2d(64)

    def forward(self, x):
        #x = imu value
        ##embedding
        x = self.conv1d_4_32(x.squeeze(2))
        x = self.bn1(x)
        x = self.elu(x)

        x = self.conv1d_32_64(x)
        x = self.bn2(x)
        x = self.elu(x)

        ##ODE
        x = self.feature_layers(x.unsqueeze(2))
        x = self.bn3(x)

        return x


class NODE_pose(torch.nn.Module):
    def __init__(self):
        super(NODE_pose, self).__init__()

        self.norm = norm(256)

        self.conv1d_5_64 = torch.nn.Conv1d(5, 64, kernel_size=3, padding=1, dilation=1, stride=1)
        self.conv1d_64_256 = torch.nn.Conv1d(64, 256, kernel_size=3, padding=1, dilation=1, stride=1)

        self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.ReLU()
        self.elu = torch.nn.ELU()

        #self.conv2d_32_64 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1, dilation=1, stride=1)
        #self.conv1d_32_64 = torch.nn.Conv1d(32, 64, kernel_size=3, padding=1, dilation=1, stride=1)

        self.feature_layers = [ODEBlock_s(ODEfunc(256))][0]
        #self.residual = Residual(256, 256).cuda()
        #self.batchnorm_32 = torch.nn.BatchNorm2d(32)
        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(256)
        #self.bn3 = torch.nn.BatchNorm2d(64)

        #self.linear = torch.nn.Linear(100, 32)

        self.linear1 = nn.Linear(100, 64)
        self.linear2 = nn.Linear(64, 3)
        self.linear3 = nn.Linear(3, 1)

        self.conv2d_1 = torch.nn.Conv2d(256, 64, kernel_size=3, padding=1, dilation=1, stride=1)
        self.conv2d_2 = torch.nn.Conv2d(64, 16, kernel_size=3, padding=1, dilation=1, stride=1)
        self.conv2d_3 = torch.nn.Conv2d(16, 2, kernel_size=3, padding=1, dilation=1, stride=1)

        #self.bn0 = torch.nn.BatchNorm2d(256)
        #self.bn1 = torch.nn.BatchNorm2d(64)
        #self.bn2 = torch.nn.BatchNorm2d(16)
        #self.bn3 = torch.nn.BatchNorm2d(1)

        #self.pool = torch.nn.AdaptiveAvgPool2d((2, 1)).cuda()

    def forward(self, x):
        ##embedding
        x = self.conv1d_5_64(x.squeeze(2))
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv1d_64_256(x.squeeze(2))
        x = self.bn2(x)
        x = self.tanh(x)

        ##ODE
        #out = self.linear(x)
        #out = self.tanh(out)

        x = self.feature_layers(x.unsqueeze(2))
        x = self.norm(x)
        x = self.relu(x)

        #x = self.pool(x.permute(0, 3, 1, 2))

        x1 = self.conv2d_1(x)
        x1 = self.elu(x1)

        x1 = self.conv2d_2(x1)
        x1 = self.elu(x1)

        x1 = self.conv2d_3(x1)
        x1 = self.elu(x1)

        x = self.linear1(x1)
        x = self.elu(x)

        x = self.linear2(x)
        x = self.elu(x)

        x = self.linear3(x).squeeze(2)

        ans = x.transpose(1, 2)
        #ans = self.pool(x).permute(0, 2, 1)
        #x = self.bn1(x)

        return ans

class NODE_acc_2_pos_small(torch.nn.Module):
    def __init__(self):
        super(NODE_acc_2_pos_small, self).__init__()
        #self.conv1d_2_32 = torch.nn.Conv1d(2, 32, kernel_size=3, padding=1, dilation=1, stride=1)
        self.conv1d_4_64 = torch.nn.Conv1d(4, 64, kernel_size=3, padding=1, dilation=1, stride=1)
        self.conv1d_64_256 = torch.nn.Conv1d(64, 256, kernel_size=3, padding=1, dilation=1, stride=1)

        #torch.nn.init.xavier_uniform_(self.conv1d_2_32.weight)
        #torch.nn.init.xavier_normal_(self.conv1d_2_32.weight)

        self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.ReLU()
        self.elu = torch.nn.ELU()

        #self.conv2d_32_64 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1, dilation=1, stride=1)
        #self.conv1d_32_64 = torch.nn.Conv1d(32, 64, kernel_size=3, padding=1, dilation=1, stride=1)

        self.feature_layers = [ODEBlock_s(ODEfunc(256))][0]
        #self.residual = Residual(256, 256).cuda()
        #self.batchnorm_32 = torch.nn.BatchNorm2d(32)
        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(256)
        #self.bn3 = torch.nn.BatchNorm2d(64)

    def forward(self, x):
        #x = imu value
        ##embedding
        x = self.conv1d_4_64(x.squeeze(2))
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv1d_64_256(x.squeeze(2))
        x = self.bn2(x)
        #x = self.relu(x)

        #x = self.residual(x.unsqueeze(2)).squeeze(2)
        x = self.tanh(x)
        ##ODE
        x = self.feature_layers(x.unsqueeze(2))
        #x = self.bn1(x)

        return x

class NODE_gyro_2_pos(torch.nn.Module):
    def __init__(self):
        super(NODE_gyro_2_pos, self).__init__()
        self.conv1d_1_32 = torch.nn.Conv1d(1, 32, kernel_size=3, padding=1, dilation=1, stride=1)
        #self.conv1d_3_32 = torch.nn.Conv1d(3, 32, kernel_size=3, padding=1, dilation=1, stride=1)
        self.relu = torch.nn.ReLU()
        self.elu = torch.nn.ELU()
        self.tanh = torch.nn.Tanh()
        # self.conv2d_32_64 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1, dilation=1, stride=1)
        self.conv1d_32_64 = torch.nn.Conv1d(32, 64, kernel_size=3, padding=1, dilation=1, stride=1)

        self.bn1 = torch.nn.BatchNorm1d(32)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.bn3 = torch.nn.BatchNorm2d(64)

        self.feature_layers_iacc_gyro = [ODEBlock(ODEfunc_iacc_gyro(64))][0]

    def forward(self, gyro):
        ##embedding
        x = self.conv1d_1_32(gyro.squeeze(2))
        #x = self.conv1d_3_32(gyro.squeeze(2))
        x = self.bn1(x)
        x = self.elu(x)

        x = self.conv1d_32_64(x)
        x = self.bn2(x)
        x = self.elu(x)

        out = self.feature_layers_iacc_gyro(x.unsqueeze(2))
        out = self.bn3(out)

        return out

class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size - 1) // 2, bias=True)
        self.relu = None
        self.bn = None

        if relu:
            self.relu = nn.ReLU()
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        self.conv.apply(weights_init)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Residual(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(Residual, self).__init__()
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(inp_dim)
        self.conv1 = Conv(inp_dim, int(out_dim / 2), 1, relu=False)  # kernel= 1, stride=1
        self.bn2 = nn.BatchNorm2d(int(out_dim / 2))
        self.conv2 = Conv(int(out_dim / 2), int(out_dim / 2), 3, relu=False)
        self.bn3 = nn.BatchNorm2d(int(out_dim / 2))
        self.conv3 = Conv(int(out_dim / 2), out_dim, 1, relu=False)
        self.skip_layer = Conv(inp_dim, out_dim, 1, relu=False)
        if inp_dim == out_dim:
            self.need_skip = False
        else:
            self.need_skip = True

    def forward(self, x):
        if self.need_skip:
            residual = self.skip_layer(x)
        else:
            residual = x
        out = self.bn1(x)
        out = self.relu(out)
        self.conv1.apply(weights_init)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        self.conv2.apply(weights_init)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        self.conv3.apply(weights_init)
        out = self.conv3(out)
        out += residual
        return out

class NODE_gyro_2_pos_small(torch.nn.Module):
    def __init__(self):
        super(NODE_gyro_2_pos_small, self).__init__()
        self.conv1d_1_32 = torch.nn.Conv1d(1, 32, kernel_size=3, padding=1, dilation=1, stride=1)
        self.conv1d_32_256 = torch.nn.Conv1d(32, 256, kernel_size=3, padding=1, dilation=1, stride=1)
        self.relu = torch.nn.ReLU()
        self.elu = torch.nn.ELU()
        self.tanh = torch.nn.Tanh()
        # self.conv2d_32_64 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1, dilation=1, stride=1)
        #self.conv1d_32_64 = torch.nn.Conv1d(32, 64, kernel_size=3, padding=1, dilation=1, stride=1)

        #self.residual = Residual(256, 256).cuda()
        self.bn1 = torch.nn.BatchNorm1d(32)
        self.bn2 = torch.nn.BatchNorm1d(256)
        #self.bn3 = torch.nn.BatchNorm2d(64)

        self.feature_layers_iacc_gyro = [ODEBlock_s(ODEfunc_iacc_gyro(256))][0]

    def forward(self, gyro):
        ##embedding
        x = self.conv1d_1_32(gyro.squeeze(2))
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv1d_32_256(x)
        x = self.bn2(x)
        #x = self.relu(x)

        #x = self.residual(x.unsqueeze(2)).squeeze(2)
        x = self.tanh(x)
        #x = self.conv1d_32_64(x)
        #x = self.bn2(x)
        #x = self.elu(x)

        out = self.feature_layers_iacc_gyro(x.unsqueeze(2))
        #out = self.bn3(out)

        return out

class NODE_acc(torch.nn.Module):
    def __init__(self):
        super(NODE_acc, self).__init__()
        #self.conv2d_2_32 = torch.nn.Conv2d(2, 32, kernel_size=3, padding=1, dilation=1, stride=1)
        self.conv1d_2_32 = torch.nn.Conv1d(2, 32, kernel_size=3, padding=1, dilation=1, stride=1)
        #torch.nn.init.xavier_uniform_(self.conv1d_2_32.weight)
        #torch.nn.init.xavier_normal_(self.conv1d_2_32.weight)

        self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.ReLU()
        self.elu = torch.nn.ELU()
        #self.conv2d_32_64 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1, dilation=1, stride=1)
        self.conv1d_32_64 = torch.nn.Conv1d(32, 64, kernel_size=3, padding=1, dilation=1, stride=1)

        self.feature_layers = [ODEBlock(ODEfunc(64))][0]

        #self.batchnorm_32 = torch.nn.BatchNorm2d(32)
        self.bn1 = torch.nn.BatchNorm1d(32)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.bn3 = torch.nn.BatchNorm2d(64)

    def forward(self, x):
        #x = imu value
        ##embedding
        x = self.conv1d_2_32(x.squeeze(2))
        x = self.bn1(x)
        x = self.elu(x)

        x = self.conv1d_32_64(x)
        x = self.bn2(x)
        x = self.elu(x)

        ##ODE
        x = self.feature_layers(x.unsqueeze(2))
        x = self.bn3(x)

        return x

class NODE_iacc_gyro(torch.nn.Module):
    def __init__(self):
        super(NODE_iacc_gyro, self).__init__()
        self.conv1d_1_32 = torch.nn.Conv1d(1, 32, kernel_size=3, padding=1, dilation=1, stride=1)
        self.relu = torch.nn.ReLU()
        self.elu = torch.nn.ELU()
        # self.conv2d_32_64 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1, dilation=1, stride=1)
        self.conv1d_32_64 = torch.nn.Conv1d(32, 64, kernel_size=3, padding=1, dilation=1, stride=1)

        self.bn1 = torch.nn.BatchNorm1d(32)
        self.bn2 = torch.nn.BatchNorm1d(64)

        self.feature_layers_iacc_gyro = [ODEBlock(ODEfunc_iacc_gyro(64))][0]


    def forward(self, gyro):
        ##embedding
        x = self.conv1d_1_32(gyro.squeeze(2))
        x = self.bn1(x)
        x = self.elu(x)

        x = self.conv1d_32_64(x)
        x = self.bn2(x)
        x = self.elu(x)

        out = self.feature_layers_iacc_gyro(x.unsqueeze(2))

        return out

class NODE_acc2(torch.nn.Module):
    def __init__(self):
        super(NODE_acc2, self).__init__()

        self.feature_layers_acc2 = [ODEBlock(ODEfunc_acc2(64))][0]

        self.batchnorm_64 = torch.nn.BatchNorm2d(64)

    def forward(self, x):
        ##ODE
        x = self.feature_layers_acc2(x)
        #x = self.batchnorm_64(x)
        return x

class NODE_gyro(torch.nn.Module):
    def __init__(self):
        super(NODE_gyro, self).__init__()
        #self.conv2d_1_32 = torch.nn.Conv2d(1, 32, kernel_size=3, padding=1, dilation=1, stride=1)
        self.conv1d_1_32 = torch.nn.Conv1d(1, 32, kernel_size=3, padding=1, dilation=1, stride=1)
        self.relu = torch.nn.ReLU()
        #self.conv2d_32_64 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1, dilation=1, stride=1)
        self.conv1d_32_64 = torch.nn.Conv1d(32, 64, kernel_size=3, padding=1, dilation=1, stride=1)

        self.batchnorm_32 = torch.nn.BatchNorm2d(32)
        self.batchnorm_64 = torch.nn.BatchNorm2d(64)

        self.feature_layers_acc2 = [ODEBlock(ODEfunc_gyro(64))][0]

    def forward(self, x):
        #x = imu value
        ##embedding
        x = self.conv1d_1_32(x.squeeze(2))
        #x = self.batchnorm_32(x)
        x = self.relu(x)
        x = self.conv1d_32_64(x)
        #x = self.batchnorm_64(x)
        #x = self.relu(x)

        ##ODE
        x = self.feature_layers_acc2(x.unsqueeze(2))

        return x
