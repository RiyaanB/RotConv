import torch
import torch.nn as nn
from .modules import *
from torch.nn import functional as F
import time

class ComplexRotNet(nn.Module):

	def __init__(self, c_dim=128, n_angles=8):
		super().__init__()

		self.layers = nn.Sequential(
			ComplexRotConv2d(3, 32, 5, padding=2, mode='real', n_angles=n_angles),
			ComplexReLU(),
			ComplexRotConv2d(32, 64, 5, stride=2, padding=2, n_angles=n_angles),
			ComplexReLU(),
			ComplexRotConv2d(64, 128, 5, padding=2, n_angles=n_angles),
			ComplexReLU(),
			ComplexRotConv2d(128, 128, 5, stride=2, padding=2, n_angles=n_angles),
			ComplexReLU(),
			ComplexRotConv2d(128, 128, 5, padding=2, n_angles=n_angles),
			ComplexReLU(),
			ComplexRotConv2d(128, 256, 5, stride=2, padding=2, n_angles=n_angles),
			ComplexReLU(),
			ToComplex(),
			ToReal(),
			nn.Flatten(),
			nn.Linear(256*4*4, 1024),#.to(torch.cfloat),
			nn.ReLU(),
			nn.Linear(1024, 10),
		)

	def forward(self, x):
		batch, in_channels, H, W = x.shape
		net = x.unsqueeze(1)
		out = self.layers(net)
		return out

class TestNet(nn.Module):

	def __init__(self, c_dim=128, n_angles=8):
		super().__init__()

		self.layers = nn.Sequential(
			ComplexRotConv2d(3, 3, 5, padding=2, mode='real', n_angles=n_angles),
			ComplexReLU(),
			ComplexConv2d(3, 32, 5, stride=2, padding=2),
			ComplexReLU(),
			ComplexConv2d(32, 32, 5, stride=2, padding=2),
			ComplexReLU(),
			ComplexConv2d(32, 64, 5, stride=2, padding=2),
			ComplexReLU(),
			ToComplex(),
			nn.Flatten(),
			nn.Linear(64*4*4, 128).to(torch.cfloat),
			ToReal(),
			nn.ReLU(),
			nn.Linear(128, 10),
			nn.ReLU(),
			nn.Softmax()
		)
		print(sum(p.numel() for p in self.parameters()))

	def forward(self, x):
		batch, in_channels, H, W = x.shape
		net = x.unsqueeze(1)
		out = self.layers(net)
		return out

class RotInvariantNet(nn.Module):

	def __init__(self, c_dim=128, n_angles=8):
		super().__init__()

		self.layers = nn.Sequential( 
			RotInvariantConv2d(3, 32, 7, padding=3, n_angles=n_angles),
			nn.ReLU(),
			RotInvariantConv2d(32, 64, 5, stride=2, padding=2, n_angles=n_angles),
			nn.ReLU(),
			RotInvariantConv2d(64, 64, 5, stride=2, padding=2, n_angles=n_angles),
			nn.ReLU(),
			RotInvariantConv2d(64, 128, 5, stride=2, padding=2, n_angles=n_angles),
			nn.ReLU(),
			nn.Flatten(),
			nn.Linear(128*4*4, 512), 
			nn.ReLU(),
			nn.Linear(512, 10),
		)

	def forward(self, x):
		batch, in_channels, H, W = x.shape
		out = self.layers(x)
		return out


class RotEqNet(nn.Module):

	def __init__(self, c_dim=128, n_angles=8):
		super().__init__()

		self.layers = nn.Sequential(
			RotEquivariantConv2d(3, 3, 9, padding=4, n_angles=n_angles, mode='real'),
			RotEquivariantConv2d(3, 4, 7, stride=2, padding=3, n_angles=n_angles),
			RotEquivariantConv2d(4, 4, 7, stride=2, padding=3, n_angles=n_angles),
			RotEquivariantConv2d(4, 4, 7, stride=2, padding=3, n_angles=n_angles),
			ToComplex(),
			nn.Flatten(),
			nn.Linear(4*4*4, 32).to(torch.cfloat),
			RealRelu(),
			ToReal(),
			nn.Linear(32, 10),
		)

	def forward(self, x):
		batch, in_channels, H, W = x.shape
		x = x.unsqueeze(1)
		out = self.layers(x)
		return out

class ComplexRotEqNet(nn.Module):

	def __init__(self, c_dim=128, n_angles=8):
		super().__init__()

		self.layers = nn.Sequential(
			RotEquivariantConv2d(3, 8, 7, padding=3, n_angles=n_angles, mode='real'),
			RotEquivariantConv2d(8, 16, 5, stride=2, padding=2, n_angles=n_angles),
			RotEquivariantConv2d(16, 32, 5, stride=2, padding=2, n_angles=n_angles),
			RotEquivariantConv2d(32, 64, 5, stride=2, padding=2, n_angles=n_angles),
			ToComplex(),
			nn.Flatten(),
			nn.Linear(64*4*4, 1024).to(torch.cfloat),
			RealRelu(),
			nn.Linear(1024, 512),
			ToReal(),
			nn.ReLU(),
			nn.Linear(1024, 10),
		)

	def forward(self, x):
		batch, in_channels, H, W = x.shape
		x = x.unsqueeze(1)
		out = self.layers(x)
		return out

class MixNet(nn.Module):

	def __init__(self, c_dim=128):
		super().__init__()

		self.layers = nn.Sequential(
			nn.Conv2d(3, 8, 3, padding=1),
			nn.ReLU(),
			nn.Conv2d(8, 16, 3, stride=2, padding=1),
			nn.ReLU(),
			nn.Conv2d(16, 32, 3, padding=1),
			nn.ReLU(),
			nn.Conv2d(32, 64, 3, stride=2, padding=1),
			nn.ReLU(),
			RotEqConv2d(64, 128, 3, padding=1),
			nn.ReLU(),
			RotEqConv2d(128, 256, 3, stride=2, padding=1),
			nn.ReLU(),
			nn.Flatten(),
			nn.Linear(256*4*4, 1024),
			nn.ReLU(),
			nn.Linear(1024, 10),
		)

	def forward(self, x):
		batch, in_channels, H, W = x.shape
		out = self.layers(x)
		return out

class ConvNet(nn.Module):

	def __init__(self, c_dim=128):
		super().__init__()
		self.layers = nn.Sequential( 
			nn.Conv2d(3, 32, 7, padding=3),
			nn.ReLU(),
			nn.Conv2d(32, 64, 5, stride=2, padding=2),
			nn.ReLU(),
			nn.Conv2d(64, 64, 5, stride=2, padding=2),
			nn.ReLU(),
			nn.Conv2d(64, 128, 5, stride=2, padding=2),
			nn.ReLU(),
			nn.Flatten(),
			nn.Linear(128*4*4, 512), 
			nn.ReLU(),
			nn.Linear(512, 10),
		)
		print(sum(p.numel() for p in self.parameters()))

	def forward(self, x):
		return self.layers(x)

"""
self.layers = nn.Sequential(
			ComplexRotConv2d(3, 16, 7, padding=3, mode='real', n_angles=n_angles),
			ComplexReLU(),
			ComplexConv2d(16, 32, 5, stride=2, padding=2),
			ComplexReLU(),
			ComplexConv2d(32, 32, 5, stride=2, padding=2),
			ComplexReLU(),
			ComplexConv2d(32, 64, 5, stride=2, padding=2),
			ComplexReLU(),
			ToComplex(),
			nn.Flatten(),
			nn.Linear(64*4*4, 256).to(torch.cfloat),
			ToReal(),
			nn.ReLU(),
			nn.Linear(256, 10),
		)
"""