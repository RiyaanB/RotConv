import torch
import torch.nn as nn
from torch.nn import functional as F


torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.set_default_dtype(torch.double)

N, C, H, W = 10, 3, 24, 24
x = torch.randn(N, C, H, W)
weights = []
for _ in range(N):
	weights.append(torch.randn(15, 3, 5, 5))

weights = torch.stack(weights)
new_weights = weights.view(-1, 3, 5, 5)

def shape_equals(shape, tup):
	return shape == tup


def compare(x, y, epsilon=1e-5):
	return torch.allclose(x, y)
	# return torch.mean(torch.square(x-y)) < epsilon


def ComplexConv2d_v1(input, weight, bias=None, stride=1, padding=0, dilation=1):
	# DOES NOT SUPPORT GROUPS YET
	assert input.shape[1] == 2 					# complexity=2
	assert weight.shape[1] == 2 				# complexity=2
	assert input.shape[2] == weight.shape[2]	# in_channels
	assert weight.shape[-1] == weight.shape[-2]	# square kernel (optional)

	batch, complexity, in_channels, H, W = input.shape
	out_channels, complexity, in_channels, k, _ = weight.shape

	x_outs = []
	for b in range(batch):
		x = input[b:b+1]
		kernel_outs = []
		for kernel in weight:
			assert kernel.shape == (2, in_channels, k, k)
			assert x.shape == (1, 2, in_channels, H, W)

			# kernel = p + qi
			# x 	 = a + bi

			a, b = x[:, 0], x[:, 1]
			p, q = kernel[0:1], kernel[1:2]

			assert a.shape == (1, in_channels, H, W)
			assert b.shape == (1, in_channels, H, W)
			assert p.shape == (1, in_channels, k, k)
			assert q.shape == (1, in_channels, k, k)

			ap = F.conv2d(a, p, bias, stride, padding, dilation)
			bp = F.conv2d(b, p, bias, stride, padding, dilation)
			aq = F.conv2d(a, q, bias, stride, padding, dilation)
			bq = F.conv2d(b, q, bias, stride, padding, dilation)

			Hnew, Wnew = ap.shape[-2], ap.shape[-1]
			# ap, bp, aq, bq are [1 (batch), 1 (out_channel), H', W']
			assert ap.shape == (1, 1, Hnew, Wnew)
			assert ap.shape == bp.shape
			assert ap.shape == aq.shape
			assert ap.shape == bq.shape

			real_out = ap - bq
			imag_out = bp + aq

			assert real_out.shape == (1, 1, Hnew, Wnew)
			assert imag_out.shape == (1, 1, Hnew, Wnew)
			# real/imag_out are [1 (batch), 1 (out_channel), H', W']

			kernel_out = torch.stack((real_out, imag_out), dim=1)
			assert kernel_out.shape == (1, 2, 1, Hnew, Wnew)
			# kernel_out is [1 (batch), 2 (complexity), 1 (out_channel), H', W']

			kernel_outs.append(kernel_out)

		kernel_outs = torch.cat(kernel_outs, dim=2)
		assert kernel_outs.shape == (1, 2, out_channels, Hnew, Wnew)
		# kernel_outs is now [1, 2, num_kernels, H', W']
		
		x_outs.append(kernel_outs)

	x_outs = torch.cat(x_outs, dim=0)
	assert x_outs.shape == (batch, 2, out_channels, Hnew, Wnew)
	# x_outs is now [batch, 2, num_kernels, H', W']

	return x_outs


def ComplexConv2d_v2(input, weight, bias=None, stride=1, padding=0, dilation=1):
	# DOES NOT SUPPORT GROUPS YET
	assert input.shape[1] == 2 					# complexity=2
	assert weight.shape[1] == 2 				# complexity=2
	assert input.shape[2] == weight.shape[2]	# in_channels
	assert weight.shape[-1] == weight.shape[-2]	# square kernel (optional)

	batch, complexity, in_channels, H, W = input.shape
	out_channels, complexity, in_channels, k, _ = weight.shape

	x = input
	kernel_outs = []
	for kernel in weight:
		a, b = x[:, 0], x[:, 1]
		p, q = kernel[0:1], kernel[1:2]

		ap = F.conv2d(a, p, bias, stride, padding, dilation)
		bp = F.conv2d(b, p, bias, stride, padding, dilation)
		aq = F.conv2d(a, q, bias, stride, padding, dilation)
		bq = F.conv2d(b, q, bias, stride, padding, dilation)

		Hnew, Wnew = ap.shape[-2], ap.shape[-1]

		real_out = ap - bq
		imag_out = bp + aq

		kernel_out = torch.stack((real_out, imag_out), dim=1)

		kernel_outs.append(kernel_out)

	kernel_outs = torch.cat(kernel_outs, dim=2)
	
	return kernel_outs

hunu = input

def ComplexConv2d_v3(input, weight, bias=None, stride=1, padding=0, dilation=1):
	# DOES NOT SUPPORT GROUPS YET
	assert input.shape[1] == 2 					# complexity=2
	assert weight.shape[1] == 2 				# complexity=2
	assert input.shape[2] == weight.shape[2]	# in_channels
	assert weight.shape[-1] == weight.shape[-2]	# square kernel (optional)

	batch, complexity, in_channels, H, W = input.shape
	out_channels, complexity, in_channels, k, _ = weight.shape

	kernel_outs = []
	for kernel in weight:
		p, q = kernel[0:1], kernel[1:2]

		ab = input.view(batch, 2*in_channels, H, W)
		pp = p.expand(2, in_channels, k, k)
		qq = q.expand(2, in_channels, k, k)
		apbp = F.conv2d(ab, pp, bias, stride, padding, dilation, groups=2)
		aqbq = F.conv2d(ab, qq, bias, stride, padding, dilation, groups=2)

		Hnew, Wnew = apbp.shape[-2], apbp.shape[-1]

		apbp = apbp.view(batch, 2, 1, Hnew, Wnew)
		aqbq = aqbq.view(batch, 2, 1, Hnew, Wnew)
	
		apbp[:,0] -= aqbq[:,1]
		apbp[:,1] += aqbq[:,0]

		kernel_outs.append(apbp)

	kernel_outs = torch.cat(kernel_outs, dim=2)
	
	return kernel_outs


def ComplexConv2d_v4(input, weight, bias=None, stride=1, padding=0, dilation=1):
	# DOES NOT SUPPORT GROUPS YET
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

	return apbp


def ComplexConv2d_v4(input, weight, bias=None, stride=1, padding=0, dilation=1):
	# DOES NOT SUPPORT GROUPS YET
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

	return apbp


if __name__ == '__main__':
	test_input = torch.randn(4, 2, 3, 32, 32)
	test_weights = torch.randn(5, 2, 3, 5, 5)
	output_v1 = ComplexConv2d_v1(test_input.clone(), test_weights.clone())

	output_v2 = ComplexConv2d_v2(test_input.clone(), test_weights.clone())
	if compare(output_v1, output_v2):
		print("V2 is same as V1")

	output_v3 = ComplexConv2d_v3(test_input.clone(), test_weights.clone())
	if compare(output_v1, output_v3):
		print("V3 is same as V1")

	output_v4 = ComplexConv2d_v4(test_input.clone(), test_weights.clone())
	if compare(output_v1, output_v4):
		print("V4 is same as V1")