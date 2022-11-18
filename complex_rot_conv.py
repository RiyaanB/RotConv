import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
import math
import torch


def ntuple(n):
	""" Ensure that input has the correct number of elements """
	def parse(x):
		if isinstance(x, collections.Iterable):
			return x
		return tuple(itertools.repeat(x, n))
	return parse


class ComplexRotConv2d(nn.Module):

	kind = {
		"real": 1,
		"complex": 2,
	}

	def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, n_angles=8, mode=1):

		# Conv2d Parameters
		kernel_size = ntuple(2)(kernel_size)
		stride = ntuple(2)(stride)
		padding = ntuple(2)(padding)
		dilation = ntuple(2)(dilation)

		# Init Conv2d Parameters
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.kernel_size = kernel_size
		self.stride = stride
		self.padding = padding
		self.dilation = dilation

		# Set Angles and corresponding Interpolation variables
		self.reset_angles_and_interps(n_angles)

		#Input tensor datatype
		assert mode in self.kind
		self.mode = mode

		# [kind, out, in, *kernel_size]
		self.weight = Parameter(torch.empty(self.kind[mode], out_channels, in_channels, *kernel_size))



