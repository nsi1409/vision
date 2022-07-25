import torch
from .. import transforms


class InputTensor(torch.Tensor):
	def __init__(self, *args, **kwargs):
		self.transformation_manifest = []
		torch.Tensor.__init__(*args, **kwargs)

	def append_transform(self, transform):
		self.transformation_manifest.append(transform)


class LabelTensor(torch.Tensor):
	def __init__(self, *args, **kwargs):
		torch.Tensor.__init__(*args, **kwargs)


class RandomRotation(transforms.RandomRotation):
	def __init__(self, *args, **kwargs):
		transforms.RandomRotation.__init__(self, *args, **kwargs)

	def forward(self, img):
		print('here')
		transformations = img.transformation_manifest
		tensor = transforms.RandomRotation.forward(self, img)
		input_tensor = InputTensor(tensor)
		input_tensor.transformation_manifest = transformations
		input_tensor.transformation_manifest.append(['rotation', self.get_params(self.degrees)])
		print(input_tensor.transformation_manifest)
		return input_tensor


