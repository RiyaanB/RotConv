import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
import math
import torch
import torchvision.transforms.functional as TF

def ntuple(n):
	def parse(x):
		if isinstance(x, collections.Iterable):
			return x
		return tuple(itertools.repeat(x, n))
	return parse


class ComplexRotConv(nn.Module):

	modes = {
		'real': 1,
		'complex': 2,
	}

	def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
		padding=0, dilation=1, n_angles=8, mode='complex'):
		super(ComplexRotConv, self).__init__()

		self.kernel_size = ntuple(2)(kernel_size)
		self.stride = ntuple(2)(stride)
		self.padding = ntuple(2)(padding)
		self.dilation = ntuple(2)(dilation)

		self.in_channels = in_channels
		self.out_channels = out_channels

		self.n_angles = n_angles
		angles = np.linspace(0, 2*torch.pi, n_angles, endpoint=False)
		self.angles = torch.tensor(angles, requires_grad=False)
		
		self.mode = mode
		self.complexity = self.modes[mode]

		# weight is [complexity, out_channels, in_channels, k, k]
		self.weight = Parameter(torch.tensor(complexity, out_channels, in_channels, *kernel_size))

		self.reset_parameters()

	def reset_parameters(self):
		n = self.in_channels
		for k in self.kernel_size:
			n *= k
		stdv = 1. / n**0.5
		for i in range(self.complexity):
			self.weight[i].data.uniform_(-stdv, stdv)

	def forward(self, input):
		# input : [batch, 1 or 2, in_channels, 	   axis1, axis2   ]
		# output: [batch,      2, out_channels, newaxis1, newaxis2]

		# input_real is [batch_size, in_channels, axis1, axis2] always
		input_real = input[:, 0]

		# stacked_input_real is [batch_size, n_angles*in_channels, axis1, axis2]
		stacked_input_real = torch.cat([input_real] * self.n_angles, 1)

		# kernel_real is [out_channels, in_channels, k, k]
		kernel_real = self.weight[0]

		# rotated_kernels is [n_angles * out_channels, in_channels, k, k]
		rotated_kernels_real = torch.cat([TF.rotate(kernel_real, angle) for angle in self.angles], 0)

		if self.mode == 'real':

			# raw_activations is [n_angles * out_channels, in_channels, k , k]
			raw_activations = F.Conv2d(stacked_input_real, rotated_kernels_real, None, self.stride, self.padding, self.dilation, n_angles)
			magnitudes = torch.relu(torch.max(raw_activations, axis=1))
			arguments = torch.argmax(raw_activations, axis=1) * 2 * torch.pi / self.n_angles
			out = torch.cat((torch.cos(arguments)*magnitudes, torch.sin(arguments)*magnitudes), axis=1)
			return out

		if self.mode == 'complex':

			# input_imag is [batch_size, in_channels, axis1, axis2] if it exists
			input_imag = input[:, 1] if self.mode == 'complex' else None

			# stacked_input_real is [batch_size, n_angles*in_channels, axis1, axis2]
			stacked_input_imag = torch.cat([input_imag] * self.n_angles, 1)

			# kernel_real is [out_channels, in_channels, k, k]
			kernel_imag = self.weight[1]

			# rotated_kernels is [n_angles * out_channels, in_channels, k, k]
			rotated_kernels_imag = torch.cat([TF.rotate(kernel_imag, angle) for angle in self.angles], 0)

			# costhetas and sinthetas are [n_angles * out_channels,]
			costhetas = torch.cat([torch.cos(self.angles)]*self.out_channels, axis=0)
			sinthetas = torch.cat([torch.sin(self.angles)]*self.out_channels, axis=0)

			# both should be [n_angles * out_channels, in_channels, k, k]
			weight_real = costhetas*rotated_kernels_real - sinthetas*rotated_kernels_imag
			weight_imag = sinthetas*rotated_kernels_real - costhetas*rotated_kernels_imag

			



if __name__ == '__main__':
	main()