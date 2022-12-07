from .utils import *

class RotInvariantConv2d(nn.Module):

	def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
		padding=0, dilation=1, n_angles=8):
		super(RotInvariantConv2d, self).__init__()

		self.kernel_size = ntuple(2)(kernel_size)
		self.k = self.kernel_size[0]
		self.stride = ntuple(2)(stride)
		self.padding = ntuple(2)(padding)
		self.dilation = ntuple(2)(dilation)

		self.in_channels = in_channels
		self.out_channels = out_channels

		self.n_angles = n_angles

		# canon weight is [out_channels, complexity, in_channels, k, k]
		self.canonical_weight = Parameter(torch.randn(self.out_channels, self.in_channels, *self.kernel_size))

		self.mask = torch.tensor(get_circular_mask(*self.kernel_size)).repeat(out_channels, in_channels, 1, 1).to('cuda')

		self.reset_parameters()

	def reset_parameters(self):
		n = self.in_channels
		for k in self.kernel_size:
			n *= k
		stdv = 1. / n**0.5
		self.canonical_weight.data.uniform_(-stdv, stdv)
		self.canonical_weight.requires_grad = True

	def forward(self, input):
		# input : [batch,  in_channels,    axis1,    axis2]
		# output: [batch, out_channels, newaxis1, newaxis2]

		assert True not in torch.isnan(input)
		assert len(input.shape) == 4
		assert input.shape[1] == self.in_channels

		batch = input.shape[0]

		rotated_weights = get_rotated_weights(self.canonical_weight, self.mask, self.n_angles)

		out = F.conv2d(input, rotated_weights, None, self.stride, self.padding, self.dilation)
		Hnew, Wnew = out.shape[-2:]

		raw_activations = out.view(batch, self.n_angles, self.out_channels, Hnew, Wnew)
		raw_magnitudes = torch.relu(raw_activations)

		best_magnitudes = torch.max(raw_magnitudes, dim=1)

		final = best_magnitudes.values

		assert final.shape == (batch, self.out_channels, Hnew, Wnew)
		return final