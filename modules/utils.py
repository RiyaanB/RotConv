import itertools
import collections
import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF
from torch.nn.parameter import Parameter
import torch.nn.functional as F

stdinput = input

def ntuple(n):
	def parse(x):
		if type(x) in [list, tuple]:
			return x
		return tuple(itertools.repeat(x, n))
	return parse


def rotate_kernel(kernel, angle):
	k, k = kernel.shape[-2:]
	new_weight = TF.rotate(kernel.view(-1,k,k), (180/torch.pi) * angle, interpolation=InterpolationMode.BILINEAR).view(*kernel.shape)

	if kernel.shape[1] == 2:
		real_kernels = new_weight[:, 0]
		imag_kernels = new_weight[:, 1]
		sintheta, costheta = np.sin(angle), np.cos(angle)
		new_real_kernels = costheta*real_kernels - sintheta*imag_kernels
		new_imag_kernels = sintheta*real_kernels + costheta*imag_kernels
		output = torch.stack((new_real_kernels, new_imag_kernels), axis=1)
	else:
		output = new_weight

	assert output.shape == kernel.shape
	return output

def get_circular_mask(h,w):
	center = int(h/2.), int(w/2.)
	Y, X = np.ogrid[:h, :w]
	dist_from_center = np.sqrt((X-center[0])**2 + (Y-center[1])**2)
	return dist_from_center <= h/2.


def get_rotated_weights(canonical_weight, mask, n_angles):
	angle_increment = (2*torch.pi)/n_angles
	rotated_weights = []
	for n_angle in range(n_angles):
		angle = n_angle * angle_increment
		rotated_weights.append(mask * rotate_kernel(canonical_weight * mask, angle))
	rotated_weights = torch.cat(rotated_weights, dim=0)
	assert rotated_weights.shape[0] == n_angles * canonical_weight.shape[0]
	assert rotated_weights.shape[1:] == canonical_weight.shape[1:]
	return rotated_weights