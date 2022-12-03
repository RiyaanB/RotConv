import torch
import torch.nn as nn
from complex_rot_conv import ComplexRotConv2d

class ComplexRotNet(nn.Module):

	def __init__(self, c_dim=128):
		super().__init__()

		self.conv0  = ComplexRotConv2d(3, 16, 3, stride=2, mode='real')
		self.conv1  = ComplexRotConv2d(16, 32, 3, stride=2)
		self.conv2  = ComplexRotConv2d(32, 64, 3, stride=2)
		self.conv3  = ComplexRotConv2d(64, 128, 3, stride=2)
		self.conv4  = ComplexRotConv2d(128, 256, 3, stride=2)
		self.fc_out = nn.Linear(256, c_dim).to(torch.cfloat)

	def forward(self, x):
		batch_size = x.shape[0]
		net = x.unsqueeze(1)
		net = self.conv0.forward(net)
		net = self.conv1.forward(net)
		net = self.conv2.forward(net)
		net = self.conv3.forward(net)
		net = self.conv4.forward(net)
		net = (net[:,0] + net[:,1]*j).view(batch_size, self.fc_out.in_features, -1).mean(2)
		out = nn.ReLU(torch.real(self.fc_out(net)))
		return out
