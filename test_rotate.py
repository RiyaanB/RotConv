import torch
import torchvision.transforms.functional as TF
from torchvision.io import read_image
from torchvision.utils import save_image as write_image

def save_image(img, name):
	write_image(img/255, name)

img = torch.tensor(read_image('lenna.jpg'), dtype=torch.float, requires_grad=True)
rot = TF.rotate(img, 45)
save_image(rot, 'hunumunu.jpg')