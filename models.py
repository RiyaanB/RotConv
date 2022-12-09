import torch
import torch.nn as nn
from .modules import *
from torch.nn import functional as F
import time


class RotInvariantNet(nn.Module):

	def __init__(self, in_dim=3, out_dim=128, n_angles=8):
		super().__init__()

		self.layers = nn.Sequential( 
			RotInvariantConv2d(in_dim, 16, 5, stride=2, padding=1, n_angles=n_angles),
			nn.ReLU(),
			RotInvariantConv2d(16, 64, 5, stride=2, padding=0, n_angles=n_angles),
			nn.ReLU(),
			RotInvariantConv2d(64, 128, 5, stride=2, padding=0, n_angles=n_angles),
			nn.ReLU(),
			nn.Flatten(),
			nn.Linear(128, 64), 
			nn.ReLU(),
			nn.Linear(64, out_dim),
		)

	def forward(self, x):
		batch, in_channels, H, W = x.shape
		out = self.layers(x)
		return out


class RotEqNet(nn.Module):

	def __init__(self, in_dim=3, out_dim=128, n_angles=8):
		super().__init__()

		self.layers = nn.Sequential(
			RotEquivariantConv2d(in_dim , 16, 5, stride=2, padding=1, n_angles=n_angles, mode='real'),
			RotEquivariantConv2d(16, 64, 5, stride=2, padding=0, n_angles=n_angles),
			RotEquivariantConv2d(64, 128, 5, stride=2, padding=0, n_angles=n_angles),
			ToComplex(),
			nn.Flatten(),
			nn.Linear(128, 64).to(torch.cfloat),
			RealRelu(),
			ToReal(),
			nn.Linear(64, out_dim),
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
			RotEquivariantConv2d(in_dim, 16, 5, stride=2, padding=1, n_angles=n_angles, mode='real'),
			RotEquivariantConv2d(16, 64, 5, stride=2, padding=0, n_angles=n_angles),
			RotEquivariantConv2d(64, 128, 5, stride=2, padding=0, n_angles=n_angles),
			ToComplex(),
			nn.Flatten(),
			nn.Linear(128, 32).to(torch.cfloat),
			RealRelu(),
			nn.Linear(32, 32).to(torch.cfloat),
			ToReal(),
			nn.ReLU(),
			nn.Linear(32, out_dim),
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
			nn.Conv2d(in_dim, 16, 5, stride=2, padding=1),
			nn.ReLU(),
			nn.Conv2d(16, 64, 5, stride=2, padding=0),
			nn.ReLU(),
			nn.Conv2d(64, 128, 5, stride=2, padding=0),
			nn.ReLU(),
			nn.Flatten(),
			nn.Linear(128, 64), 
			nn.ReLU(),
			nn.Linear(64, out_dim),
		)
		print(sum(p.numel() for p in self.parameters()))

	def forward(self, x):
		return self.layers(x)
