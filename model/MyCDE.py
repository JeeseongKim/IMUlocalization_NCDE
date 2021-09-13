import copy
import json
import math
import numpy as np
import os
import pathlib
import sklearn.metrics

import torch
import tqdm
#import controldiffeq
import math
import CDE_model

from CDE_model import vector_fields, metamodel, other

import torchcde

class CDEFunc_gyro(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels):
        super(CDEFunc_gyro, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.linear1 = torch.nn.Linear(hidden_channels, 64)
        self.linear2 = torch.nn.Linear(64, 128)
        self.linear3 = torch.nn.Linear(128, 256)
        self.linear4 = torch.nn.Linear(256, 256)
        self.linear5 = torch.nn.Linear(256, 64)
        self.linear6 = torch.nn.Linear(64, input_channels * hidden_channels)

        self.relu = torch.nn.ReLU()
        self.elu = torch.nn.ELU()

        self.bn1 = torch.nn.BatchNorm1d(3)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.bn3 = torch.nn.BatchNorm1d(128)
        self.bn4 = torch.nn.BatchNorm1d(256)

    ######################
    # For most purposes the t argument can probably be ignored; unless you want your CDE to behave differently at
    # different times, which would be unusual. But it's there if you need it!
    ######################
    def forward(self, t, z):
        # z has shape (batch, hidden_channels)
        z = self.bn1(z)

        z = self.linear1(z)
        z = self.bn2(z)
        # z = z.relu()
        z = self.elu(z)

        z = self.linear2(z)
        z = self.bn3(z)
        z = self.elu(z)

        z = self.linear3(z)
        z = self.bn4(z)
        z = self.elu(z)

        z = self.linear4(z)
        z = self.bn4(z)
        z = self.elu(z)

        z = self.linear5(z)
        z = self.bn2(z)
        z = self.elu(z)

        z = self.linear6(z)

        ######################
        # Easy-to-forget gotcha: Best results tend to be obtained by adding a final tanh nonlinearity.
        ######################
        z = z.tanh()

        ######################
        # Ignoring the batch dimension, the shape of the output tensor must be a matrix,
        # because we need it to represent a linear map from R^input_channels to R^hidden_channels.
        ######################
        z = z.view(z.size(0), self.hidden_channels, self.input_channels)
        return z

class CDEFunc_vel(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels):
        super(CDEFunc_vel, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.linear1 = torch.nn.Linear(hidden_channels, 64)
        self.linear2 = torch.nn.Linear(64, 128)
        self.linear3 = torch.nn.Linear(128, 256)
        self.linear4 = torch.nn.Linear(256, 256)
        self.linear5 = torch.nn.Linear(256, 64)
        self.linear6 = torch.nn.Linear(64, input_channels * hidden_channels)

        self.relu = torch.nn.ReLU()
        self.elu = torch.nn.ELU()

        self.bn1 = torch.nn.BatchNorm1d(8)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.bn3 = torch.nn.BatchNorm1d(128)
        self.bn4 = torch.nn.BatchNorm1d(256)

    ######################
    # For most purposes the t argument can probably be ignored; unless you want your CDE to behave differently at
    # different times, which would be unusual. But it's there if you need it!
    ######################
    def forward(self, t, z):
        # z has shape (batch, hidden_channels)
        z = self.bn1(z)

        z = self.linear1(z)
        z = self.bn2(z)
        # z = z.relu()
        z = self.elu(z)

        z = self.linear2(z)
        z = self.bn3(z)
        z = self.elu(z)

        z = self.linear3(z)
        z = self.bn4(z)
        z = self.elu(z)

        z = self.linear4(z)
        z = self.bn4(z)
        z = self.elu(z)

        z = self.linear5(z)
        z = self.bn2(z)
        z = self.elu(z)

        z = self.linear6(z)

        ######################
        # Easy-to-forget gotcha: Best results tend to be obtained by adding a final tanh nonlinearity.
        ######################
        z = z.tanh()

        ######################
        # Ignoring the batch dimension, the shape of the output tensor must be a matrix,
        # because we need it to represent a linear map from R^input_channels to R^hidden_channels.
        ######################
        z = z.view(z.size(0), self.hidden_channels, self.input_channels)
        return z

class CDEFunc_imu(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels):
        super(CDEFunc_imu, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.linear1 = torch.nn.Linear(hidden_channels, 64)
        self.linear2 = torch.nn.Linear(64, 128)
        self.linear3 = torch.nn.Linear(128, 256)
        self.linear4 = torch.nn.Linear(256, 256)
        self.linear5 = torch.nn.Linear(256, 64)
        self.linear6 = torch.nn.Linear(64, input_channels * hidden_channels)

        self.relu = torch.nn.ReLU()
        self.elu = torch.nn.ELU()

        #self.bn0 = torch.nn.BatchNorm1d(16)
        self.bn1 = torch.nn.BatchNorm1d(8)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.bn3 = torch.nn.BatchNorm1d(128)
        self.bn4 = torch.nn.BatchNorm1d(256)
    ######################
    # For most purposes the t argument can probably be ignored; unless you want your CDE to behave differently at
    # different times, which would be unusual. But it's there if you need it!
    ######################
    def forward(self, t, z):
        # z has shape (batch, hidden_channels)
        z = self.bn1(z)
        #z = self.bn0(z)

        z = self.linear1(z)
        z = self.bn2(z)
        #z = z.relu()
        z = self.elu(z)

        z = self.linear2(z)
        z = self.bn3(z)
        z = self.elu(z)

        z = self.linear3(z)
        z = self.bn4(z)
        z = self.elu(z)

        z = self.linear4(z)
        z = self.bn4(z)
        z = self.elu(z)

        z = self.linear5(z)
        z = self.bn2(z)
        z = self.elu(z)

        z = self.linear6(z)

        ######################
        # Easy-to-forget gotcha: Best results tend to be obtained by adding a final tanh nonlinearity.
        ######################
        z = z.tanh()

        ######################
        # Ignoring the batch dimension, the shape of the output tensor must be a matrix,
        # because we need it to represent a linear map from R^input_channels to R^hidden_channels.
        ######################
        z = z.view(z.size(0), self.hidden_channels, self.input_channels)
        return z


class CDEFunc(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels):
        super(CDEFunc, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.linear1 = torch.nn.Linear(hidden_channels, 128)
        self.linear2 = torch.nn.Linear(128, input_channels * hidden_channels)
        #self.linear1 = torch.nn.Linear(hidden_channels, 64)
        #self.linear2 = torch.nn.Linear(64, input_channels * hidden_channels)
        #self.linear2 = torch.nn.Linear(128, 5 * hidden_channels)

        #torch.nn.init.xavier_normal(self.linear1.weight)
        #torch.nn.init.xavier_normal(self.linear2.weight)
        #torch.nn.init.xavier_normal(self.linear3.weight)

        torch.nn.init.kaiming_uniform_(self.linear1.weight)
        torch.nn.init.kaiming_uniform_(self.linear2.weight)
        #torch.nn.init.kaiming_uniform_(self.linear3.weight)

        self.softplus = torch.nn.Softplus()

    ######################
    # For most purposes the t argument can probably be ignored; unless you want your CDE to behave differently at
    # different times, which would be unusual. But it's there if you need it!
    ######################
    def forward(self, t, z):
        # z has shape (batch, hidden_channels)
        z = self.linear1(z)
        z = self.softplus(z)

        z = self.linear2(z)

        ######################
        # Easy-to-forget gotcha: Best results tend to be obtained by adding a final tanh nonlinearity.
        ######################
        z = z.tanh()

        ######################
        # Ignoring the batch dimension, the shape of the output tensor must be a matrix,
        # because we need it to represent a linear map from R^input_channels to R^hidden_channels.
        ######################
        z = z.view(z.size(0), self.hidden_channels, self.input_channels)
        #z = z.view(z.size(0), self.hidden_channels, 5)
        return z

class CDEFunc_P(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels):
        super(CDEFunc_P, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.linear1 = torch.nn.Linear(hidden_channels, 32)
        self.linear2 = torch.nn.Linear(32, input_channels * hidden_channels)
        #self.linear1 = torch.nn.Linear(hidden_channels, 64)
        #self.linear2 = torch.nn.Linear(64, input_channels * hidden_channels)

        #torch.nn.init.kaiming_uniform_(self.linear1.weight)
        #torch.nn.init.kaiming_uniform_(self.linear2.weight)

        self.softplus = torch.nn.Softplus()

    ######################
    # For most purposes the t argument can probably be ignored; unless you want your CDE to behave differently at
    # different times, which would be unusual. But it's there if you need it!
    ######################
    def forward(self, t, z):
        # z has shape (batch, hidden_channels)
        z = self.linear1(z)
        z = self.softplus(z)

        z = self.linear2(z)

        ######################
        # Easy-to-forget gotcha: Best results tend to be obtained by adding a final tanh nonlinearity.
        ######################
        z = z.tanh()

        ######################
        # Ignoring the batch dimension, the shape of the output tensor must be a matrix,
        # because we need it to represent a linear map from R^input_channels to R^hidden_channels.
        ######################
        z = z.view(z.size(0), self.hidden_channels, self.input_channels)
        #z = z.view(z.size(0), self.hidden_channels, 5)
        return z

class model_cde(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, interpolation="cubic", **kwargs):
        super(model_cde, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.func = CDEFunc(input_channels, hidden_channels)

        self.initial = torch.nn.Linear(input_channels, hidden_channels)
        #self.initial = torch.nn.Linear(5, hidden_channels) #linear interpolation
        #self.initial_1 = torch.nn.Linear(12, hidden_channels)

        self.softplus = torch.nn.Softplus()
        self.relu = torch.nn.ReLU()

        self.linear = torch.nn.Linear(hidden_channels, output_channels)
        #self.linear1 = torch.nn.Linear(hidden_channels, 32)
        #self.linear2 = torch.nn.Linear(32, output_channels)

        #torch.nn.init.xavier_normal(self.initial.weight)
        #torch.nn.init.xavier_normal(self.linear1.weight)
        #torch.nn.init.xavier_normal(self.linear2.weight)

        torch.nn.init.kaiming_uniform_(self.initial.weight)
        #torch.nn.init.kaiming_uniform_(self.linear.weight)

        self.kwargs = kwargs

    def forward(self, cde_src):
        #interpolate
        #my_src = torch.cat([time.unsqueeze(2), src[:, :, 3].unsqueeze(2), src[:, :, 4].unsqueeze(2), src[:, :, 2].unsqueeze(2), vel[:, :, 0].unsqueeze(2), vel[:, :, 1].unsqueeze(2)], dim=2)

        coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(cde_src)
        #coeffs = torchcde.linear_interpolation_coeffs(cde_src[:, :, 1:6])
        #coeffs = torchcde.linear_interpolation_coeffs(x=cde_src.float(), rectilinear=True)
        myX = torchcde.CubicSpline(coeffs)
        myX = myX.float()
        #myX = torchcde.LinearInterpolation(coeffs)

        myX0 = myX.evaluate(myX.interval[0])
        z0 = self.initial(myX0.float()) #initial observation
        #z0 = self.softplus(z0) #initial observation
        #z0 = self.initial_1(z0) #initial observation

        # solve cde
        zT = torchcde.cdeint(X=myX, z0=z0, adjoint=False, func=self.func, t=myX.interval, **self.kwargs)
        #get_ans = self.batchnorm(zT)
        #ans = self.readout(zT[:, 1])

        #ans = self.linear1(zT[:, 1])
        #ans = self.softplus(ans)

        #ans = self.linear2(ans)
        ans = self.linear(zT[:, 1])

        return ans

class model_cde_P(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, interpolation="cubic", **kwargs):
        super(model_cde_P, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.func = CDEFunc_P(input_channels, hidden_channels)

        self.initial = torch.nn.Linear(input_channels, hidden_channels)
        #self.initial = torch.nn.Linear(5, hidden_channels) #linear interpolation
        #self.initial_1 = torch.nn.Linear(12, hidden_channels)

        self.softplus = torch.nn.Softplus()

        self.linear1 = torch.nn.Linear(hidden_channels, 16)
        self.linear2 = torch.nn.Linear(16, output_channels)
        #self.linear1 = torch.nn.Linear(hidden_channels, 32)
        #self.linear2 = torch.nn.Linear(32, output_channels)

        #torch.nn.init.xavier_normal(self.initial.weight)
        #torch.nn.init.xavier_normal(self.linear1.weight)
        #torch.nn.init.xavier_normal(self.linear2.weight)

        #torch.nn.init.kaiming_uniform_(self.initial.weight)
        #torch.nn.init.kaiming_uniform_(self.linear1.weight)
        #torch.nn.init.kaiming_uniform_(self.linear2.weight)
        #torch.nn.init.kaiming_uniform_(self.linear.weight)

        self.kwargs = kwargs

    def forward(self, cde_src):
        coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(cde_src)
        myX = torchcde.CubicSpline(coeffs)

        #coeffs = torchcde.linear_interpolation_coeffs(x=cde_src.float(), rectilinear=True)
        #myX = torchcde.LinearInterpolation(coeffs)
        myX = myX.float()

        myX0 = myX.evaluate(myX.interval[0])
        z0 = self.initial(myX0.float()) #initial observation
        #z0 = self.softplus(z0) #initial observation
        #z0 = self.initial_1(z0) #initial observation

        # solve cde
        zT = torchcde.cdeint(X=myX, z0=z0, adjoint=False, func=self.func, t=myX.interval, **self.kwargs)
        #get_ans = self.batchnorm(zT)
        #ans = self.readout(zT[:, 1])

        ans = self.linear1(zT[:, 1])
        ans = self.softplus(ans)

        ans = self.linear2(ans)
        #ans = self.linear(zT[:, 1])

        return ans

class model_cde_multi(torch.nn.Module):
    def __init__(self, **kwargs):
        super(model_cde_multi, self).__init__()

        self.func_imu = CDEFunc_imu(input_channels=3, hidden_channels=8)
        #self.func_imu = CDEFunc_imu(input_channels=3, hidden_channels=16)
        self.func_vel = CDEFunc_vel(input_channels=3, hidden_channels=8)
        self.func_gyro = CDEFunc_gyro(input_channels=2, hidden_channels=3)

        self.initial_imu = torch.nn.Linear(3, 8)
        #self.initial_imu = torch.nn.Linear(12, 16)
        self.initial_vel = torch.nn.Linear(3, 8)
        self.initial_gyro = torch.nn.Linear(2, 3)

        self.linear1 = torch.nn.Linear(19, 8)
        self.linear2 = torch.nn.Linear(8, 2)
        self.linear3 = torch.nn.Linear(2, 2)

        self.kwargs = kwargs

        self.relu = torch.nn.ReLU()
        self.elu = torch.nn.ELU()

        self.bn0 = torch.nn.BatchNorm1d(19)
        self.bn1 = torch.nn.BatchNorm1d(8)
        self.bn2 = torch.nn.BatchNorm1d(2)

    def forward(self, src, time, vel):
        imu_src = torch.cat([time.unsqueeze(2), src[:, :, 3].unsqueeze(2), src[:, :, 4].unsqueeze(2)], dim=2)
        vel_src = torch.cat([time.unsqueeze(2), vel[:, :, 0].unsqueeze(2), vel[:, :, 1].unsqueeze(2)], dim=2)
        gyro_src = torch.cat([time.unsqueeze(2), src[:, :, 2].unsqueeze(2)], dim=2)

        imu_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(imu_src)
        imu_X = torchcde.CubicSpline(imu_coeffs)
        #imu_X = torchcde.LinearInterpolation(imu_coeffs)
        imu_X0 = imu_X.evaluate(imu_X.interval[0])
        imu_z0 = self.initial_imu(imu_X0)  # initial observation
        imu_zT = torchcde.cdeint(X=imu_X, z0=imu_z0, func=self.func_imu, t=imu_X.interval, **self.kwargs)

        vel_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(vel_src)
        vel_X = torchcde.CubicSpline(vel_coeffs)
        #vel_X = torchcde.LinearInterpolation(vel_coeffs)
        vel_X0 = vel_X.evaluate(vel_X.interval[0])
        vel_z0 = self.initial_vel(vel_X0)  # initial observation
        vel_zT = torchcde.cdeint(X=vel_X, z0=vel_z0, func=self.func_vel, t=vel_X.interval, **self.kwargs)

        gyro_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(gyro_src)
        gyro_X = torchcde.CubicSpline(gyro_coeffs)
        #gyro_X = torchcde.LinearInterpolation(gyro_coeffs)
        gyro_X0 = gyro_X.evaluate(gyro_X.interval[0])
        gyro_z0 = self.initial_gyro(gyro_X0)  # initial observation
        gyro_zT = torchcde.cdeint(X=gyro_X, z0=gyro_z0, func=self.func_gyro, t=gyro_X.interval, **self.kwargs)

        get_zT = torch.cat([imu_zT[:, 1], vel_zT[:, 1], gyro_zT[:, 1]], dim=1)

        get_zT = self.bn0(get_zT)

        ans = self.linear1(get_zT)
        ans = self.bn1(ans)
        ans = self.elu(ans)

        ans = self.linear2(ans)
        ans = self.bn2(ans)
        ans = self.elu(ans)

        ans = self.linear3(ans)

        return ans

