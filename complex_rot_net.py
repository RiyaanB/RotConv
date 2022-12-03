import torch
import torch.nn as nn
from complex_rot_conv import ComplexRotConv2d

class ComplexRotNet(nn.Module):

	def __init__(self, c_dim=128):
		super().__init__()

		self.conv0  = ComplexRotConv2d(3, 16, 5, mode='real')
		self.conv1  = ComplexRotConv2d(16, 32, 3, stride=2)
		self.conv2  = ComplexRotConv2d(32, 64, 5)
		self.conv3  = ComplexRotConv2d(64, 128, 5)
		self.conv4  = ComplexRotConv2d(128, 256, 5, stride=2)
		self.linear1 = nn.Linear(256, 256).to(torch.cfloat)
		self.activation = nn.ReLU()
		self.linear2 = nn.Linear(256, c_dim)

	def forward(self, x):
		batch_size = x.shape[0]
		net = x.unsqueeze(1)
		net = self.conv0.forward(net)
		net = self.conv1.forward(net)
		net = self.conv2.forward(net)
		net = self.conv3.forward(net)
		net = self.conv4.forward(net)
		net = (net[:,0] + net[:,1]*1j).view(batch_size, self.linear1.in_features, -1).mean(2)
		out = self.activation(torch.real(self.linear1(net)))
		out = self.linear2(out)
		print(out.shape)
		return out

model = ComplexRotNet(128)
x = torch.randn(4,3,32,32)
model.forward(x)