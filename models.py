import torch
import torch.nn as nn
from .modules import *
from torch.nn import functional as F
import time


class RotInvariantNet(nn.Module):

	def __init__(self, in_dim=3, out_dim=128, n_angles=8):
		super().__init__()

		self.layers = nn.Sequential( 
			RotInvariantConv2d(in_dim, 32, 7, padding=3, n_angles=n_angles),
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
			nn.Linear(512, out_dim),
		)

	def forward(self, x):
		batch, in_channels, H, W = x.shape
		out = self.layers(x)
		return out


class RotEqNet(nn.Module):

	def __init__(self, in_dim=3, out_dim=128, n_angles=8):
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

	def __init__(self, in_dim=3, out_dim=128, n_angles=8):
		super().__init__()

		self.layers = nn.Sequential(
			RotEquivariantConv2d(in_dim, 8, 7, padding=3, n_angles=n_angles, mode='real'),
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
			nn.Linear(1024, out_dim),
		)

	def forward(self, x):
		batch, in_channels, H, W = x.shape
		x = x.unsqueeze(1)
		out = self.layers(x)
		return out

class ConvNet(nn.Module):

	def __init__(self, in_dim=3, out_dim=128):
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
