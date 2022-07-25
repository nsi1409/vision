import torch
from .. import transforms
import math


class InputTensor(torch.Tensor):
	def __init__(self, *args, **kwargs):
		self.transformation_manifest = []
		torch.Tensor.__init__(*args, **kwargs)

	def append_transform(self, transform):
		self.transformation_manifest.append(transform)


class MetaLabel:
	def __init__(self, index):
		self.index = index

class X(MetaLabel):
	pass

class Y(MetaLabel):
	pass


class LabelTensor(torch.Tensor):
	def __init__(self, *args, **kwargs):
		self.internal_label = []
		torch.Tensor.__init__(*args, **kwargs)

	def meta_label(self, internal_label):
		self.internal_label = internal_label


class RandomRotation(transforms.RandomRotation):
	def __init__(self, *args, **kwargs):
		transforms.RandomRotation.__init__(self, *args, **kwargs)

	def forward(self, img):
		transformations = img.transformation_manifest
		tensor = transforms.RandomRotation.forward(self, img)
		input_tensor = InputTensor(tensor)
		input_tensor.transformation_manifest = transformations
		input_tensor.transformation_manifest.append(['rotation', (math.pi * self.get_params(self.degrees)) / 180])
		return input_tensor


