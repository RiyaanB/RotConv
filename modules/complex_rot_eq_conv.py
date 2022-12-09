from .utils import *

class ComplexRotEqConv2d(nn.Module):

	modes = {
		'real': 1,
		'complex': 2,
	}

	def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
		padding=0, dilation=1, n_angles=8, mode='complex'):
		super(ComplexRotConv2d, self).__init__()

		self.kernel_size = ntuple(2)(kernel_size)
		self.k = self.kernel_size[0]
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
		self.canonical_weight = Parameter(torch.randn(self.out_channels, self.complexity, self.in_channels, *self.kernel_size))

		self.mask = Parameter(torch.tensor(get_circular_mask(*self.kernel_size)).repeat(out_channels, self.complexity, in_channels, 1, 1).to(torch.float))

		self.reset_parameters()

	def reset_parameters(self):
		n = self.in_channels
		for k in self.kernel_size:
			n *= k
		stdv = 1. / n**0.5
		for i in range(self.complexity):
			self.canonical_weight[i].data.uniform_(-stdv, stdv)
		self.canonical_weight.requires_grad = True

	def forward(self, input):
		# input : [batch, complexity,  in_channels,    axis1,    axis2]
		# output: [batch,      	   2, out_channels, newaxis1, newaxis2]

		assert True not in torch.isnan(input)
		assert len(input.shape) == 5

		batch = input.shape[0]

		rotated_weights = get_rotated_weights(self.canonical_weight, self.mask, self.n_angles)

		if self.mode == 'real':

			out = F.conv2d(input[:,0], rotated_weights[:,0], None, self.stride, self.padding, self.dilation)
			Hnew, Wnew = out.shape[-2:]

			raw_activations = out.view(batch, self.n_angles, self.out_channels, Hnew, Wnew)
			raw_magnitudes = torch.relu(raw_activations)

		elif self.mode == 'complex':

			out = complex_conv2d(input, rotated_weights, None, self.stride, self.padding, self.dilation)
			Hnew, Wnew = out.shape[-2:]

			raw_activations = out.view(batch, 2, self.n_angles, self.out_channels, Hnew, Wnew)
			raw_magnitudes = raw_activations[:,0]**2 + raw_activations[:,1]**2

		else:
			raise Exception()

		best_magnitudes = torch.max(raw_magnitudes, dim=1)

		activations = best_magnitudes.values
		arguments = best_magnitudes.indices * self.angle_increment

		final = torch.stack([torch.cos(arguments)*activations, torch.sin(arguments)*activations], axis=1)

		assert final.shape == (batch, 2, self.out_channels, Hnew, Wnew)
		return final