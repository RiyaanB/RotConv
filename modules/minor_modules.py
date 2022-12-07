import torch
import torch.nn as nn


class ToComplex(nn.Module):

	def forward(self, x):
		return x[:, 0] + x[:, 1]*1j

class ToReal(nn.Module):

	def forward(self, x):
		return torch.real(x)

class ToImag(nn.Module):

	def forward(self, x):
		return torch.imag(x)

class ToNorm(nn.Module):

	def forward(self, x):
		return torch.abs(x)

class Component0Norm(nn.Module):

	def forward(self, x):
		mask = x[:,0] > 0
		x[:,0] *= mask
		x[:,1] *= mask
		return mask

class RealRelu(nn.Module):

	def forward(self, x):
		return x * (torch.real(x) > 0)

# class ComplexReLU(nn.Module):

# 	def forward(self, x):
# 		args = torch.atan2(x[:,1], x[:,0])
# 		squared_norms = x[:,0]**2 + x[:,1]**2
# 		assert True not in torch.isnan(squared_norms)
		
# 		new_norms = torch.relu(torch.sigmoid(squared_norms)-0.5)
# 		output = torch.stack((torch.cos(args)*new_norms, torch.sin(args)*new_norms), dim=1)

# 		assert True not in torch.isnan(args)
# 		assert True not in torch.isnan(output)
		
# 		# x_real = x[:,0]
# 		# x_imag = x[:,1]
# 		# squared_norms = x_real**2 + x_imag**2
# 		# pass_norm = squared_norms > 1
# 		# output = torch.stack((x_real*pass_norm, x_imag*pass_norm),dim=1)
# 		return output

# 	# def forward(self, x):
# 	# 	x_real, x_imag = x[:,0], x[:,1]
# 	# 	x_real *= x_real > 0
# 	# 	x_imag *= x_real > 0
# 	# 	# return torch.stack((x_real, x_imag),dim=1)
# 	# 	args = torch.atan2(x_imag, x_real)
# 	# 	new_norms = torch.sigmoid(x_real**2 + x_imag**2)
# 	# 	output = torch.stack((torch.cos(args)*new_norms, torch.sin(args)*new_norms), dim=1)
# 	# 	assert x.shape == output.shape
# 	# 	return output