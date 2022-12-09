import torch
import torch.nn as nn
from torch.nn import functional as F
import time
from .utils import *

def complex_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
	assert groups == 1	# DOES NOT SUPPORT MULTIPLE GROUPS YET
	assert input.shape[1] == 2 					# complexity=2
	assert weight.shape[1] == 2 				# complexity=2
	assert input.shape[2] == weight.shape[2]	# in_channels
	assert weight.shape[-1] == weight.shape[-2]	# square kernel (optional)

	batch, complexity, in_channels, H, W = input.shape
	out_channels, complexity, in_channels, k, _ = weight.shape

	p, q = weight[:, 0], weight[:, 1]
	assert p.shape == (out_channels, in_channels, k, k)
	assert q.shape == (out_channels, in_channels, k, k)

	ab = input.view(batch, 2*in_channels, H, W)
	pp = torch.cat((p,p), dim=0)
	qq = torch.cat((q,q), dim=0)
	apbp = F.conv2d(ab, pp, bias, stride, padding, dilation, groups=2)
	aqbq = F.conv2d(ab, qq, bias, stride, padding, dilation, groups=2)
	
	Hnew, Wnew = apbp.shape[-2], apbp.shape[-1]

	apbp = apbp.view(batch, 2, out_channels, Hnew, Wnew)
	aqbq = aqbq.view(batch, 2, out_channels, Hnew, Wnew)

	apbp[:,0] -= aqbq[:,1]
	apbp[:,1] += aqbq[:,0]

	assert True not in torch.isnan(apbp)

	return apbp


class ComplexConv2d(nn.Module):

	def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
		padding=0, dilation=1):
		super(ComplexConv2d, self).__init__()

		self.kernel_size = ntuple(2)(kernel_size)
		self.k = self.kernel_size[0]
		self.stride = ntuple(2)(stride)
		self.padding = ntuple(2)(padding)
		self.dilation = ntuple(2)(dilation)

		self.in_channels = in_channels
		self.out_channels = out_channels

		self.weight = Parameter(torch.randn(self.out_channels, 2, self.in_channels, *self.kernel_size))

	def reset_parameters(self):
		n = self.in_channels
		for k in self.kernel_size:
			n *= k
		stdv = 1. / n**0.5
		for i in range(self.complexity):
			self.weight[i].data.uniform_(-stdv, stdv)
		self.weight.requires_grad = True

	def forward(self, input):
		assert True not in torch.isnan(input)
		# assert True not in torch.isnan(self.weight)
		output = complex_conv2d(input, self.weight, None, self.stride, self.padding, self.dilation)
		assert True not in torch.isnan(output)
		return output
