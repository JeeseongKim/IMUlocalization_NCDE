import torch
import torchcde

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



