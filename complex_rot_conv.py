import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
import math
import torch
import torchvision.transforms.functional as TF
from complex_conv import complex_conv2d

def ntuple(n):
	def parse(x):
		if isinstance(x, collections.Iterable):
			return x
		return tuple(itertools.repeat(x, n))
	return parse


def rotate_kernel(kernel, angle):
	angle = (180/torch.pi) * angle
	new_weight = TF.rotate(kernel, angle)
	if kernel.shape[1] == 1:
		return new_weight
	if kernel.shape[1] == 2:
		real_kernels = new_weight[:, 0]
		imag_kernels = new_weight[:, 1]
		sintheta, costheta = torch.sin(angle), torch.cos(angle)
		new_real_kernels = costheta*real_kernels - sintheta*imag_kernels
		new_imag_kernels = sintheta*real_kernels - costheta*imag_kernels
		return torch.stack((new_real_kernels, new_imag_kernels), axis=1)

	# return is always [out_channels, complexity, in_channels, k, k]
	raise Exception()

class ComplexRotConv2d(nn.Module):

	modes = {
		'real': 1,
		'complex': 2,
	}

	def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
		padding=0, dilation=1, n_angles=8, mode='complex'):
		super(ComplexRotConv2d, self).__init__()

		self.kernel_size = ntuple(2)(kernel_size)
		self.stride = ntuple(2)(stride)
		self.padding = ntuple(2)(padding)
		self.dilation = ntuple(2)(dilation)

		self.in_channels = in_channels
		self.out_channels = out_channels

		self.n_angles = n_angles
		self.angle_increment = (2*torch.pi)/self.n_angles
		
		self.mode = mode
		self.complexity = self.modes[mode]

		# canon weight is [out_channels, complexity, in_channels, k, k]
		self.canonical_weight = Parameter(torch.tensor(out_channels, complexity, in_channels, *kernel_size))

		self.reset_parameters()

	def reset_parameters(self):
		n = self.in_channels
		for k in self.kernel_size:
			n *= k
		stdv = 1. / n**0.5
		for i in range(self.complexity):
			self.weight[i].data.uniform_(-stdv, stdv)

	def forward(self, input):
		# input : [batch, complexity,  in_channels,    axis1,    axis2]
		# output: [batch,      	   2, out_channels, newaxis1, newaxis2]

		batch = input.shape[0]

		rotated_weights = []
		
		for n_angle in len(self.n_angles):
			angle = n_angle * self.angle_increment
			rotated_weights.append(rotate_kernel(self.canonical_weight, angle))
		# rotate_weights is list of [out_channels, complexity, in_channels, k, k]
		rotated_weights = torch.stack(rotated_weights, dim=0)
		# rotate_weights is [n_angles * out_channels, complexity, in_channels, k, k]

		if self.mode == 'real':

			out = F.conv2d(input, rotated_weights[:,0], None, self.stride, self.padding, self.dilation)
			Hnew, Wnew = out.shape[-2:]
			raw_activations = out.view(batch, n_angles, self.out_channels, Hnew, Wnew)
			best_activations = torch.max(raw_activations, dim=1)
			magnitudes = torch.relu(best_activations.values)
			# magnitudes is [batch, self.out_channels, Hnew, Wnew]
			arguments = best_activations.indices * self.angle_increment
			# arguments is  [batch, self.out_channels, Hnew, Wnew]
			cos_args = torch.cos(arguments)
			sin_args = torch.sin(arguments)
			# returns [batch, 2, self.out_channels, Hnew, Wnew]
			return torch.stack([cos_args*magnitudes, sin_args*magnitudes], axis=1)

		if self.mode == 'complex':

			out = complex_conv2d(input, rotated_weights, None, self.stride, self.padding, self.dilation)
			Hnew, Wnew = out.shape[-2:]
			raw_activations = out.view(batch, 2, n_angles, self.out_channels, Hnew, Wnew)
			raw_magnitudes = raw_activations[:,0]**2 + raw_activations[:,1]**2
			# raw_magnitudes is [batch, n_angles, self.out_channels, Hnew, Wnew]
			best_activations = torch.max(raw_magnitudes, dim=1)
			magnitudes = raw_magnitudes * (best_activations.values > 1)
			arguments = best_activations.indices * self.angle_increment
			cos_args = torch.cos(arguments)
			sin_args = torch.sin(arguments)
			return torch.stack([cos_args*magnitudes, sin_args*magnitudes], axis=1)

		raise Exception()

if __name__ == '__main__':
	main()