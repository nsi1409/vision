import torch
from .. import transforms
import math


class InputTensor(torch.Tensor):
	def __init__(self, *args, **kwargs):
		self.transformation_manifest = []
		torch.Tensor.__init__(*args, **kwargs)

	def append_transform(self, transform):
		self.transformation_manifest.append(transform)


class MetaLabelClass:
	def __init__(self, index):
		self.index = index
		self.x = None
		self.y = None
		self.cl = None

	def extract_from_meta(self, meta):
		self.index = meta.index
		self.x = meta.x
		self.y = meta.y
		self.cl = meta.cl

	def __str__(self):
		return "index: " + str(self.index) + " x: " + str(self.x) + " y: " + str(self.y) + " classifier: " + str(self.cl)

class X(MetaLabelClass):
	def add_to_meta(self, meta, value):
		meta.x = value

class Y(MetaLabelClass):
	def add_to_meta(self, meta, value):
		meta.y = value

class Cl(MetaLabelClass):
	def add_to_meta(self, meta, value):
		meta.cl = value


class LabelTensor(torch.Tensor):
	def __init__(self, *args, **kwargs):
		self.internal_label = {}
		torch.Tensor.__init__(*args, **kwargs)

	def add_to_internal_dictionary(self, current, clone):
		if not current.index in self.internal_label:
			self.internal_label[current.index] = MetaLabelClass(current.index)
			current.add_to_meta(self.internal_label[current.index], clone)
		else:
			current.add_to_meta(self.internal_label[current.index], clone)
			

	def recursive_meta_helper(self, current, clone):
		if not isinstance(current, list):
			self.add_to_internal_dictionary(current, clone)
		else:
			for i in range(len(current)):
				self.recursive_meta_helper(current[i], clone[i])

	def recursive_extractor(self):
		pass

	def metalabel(self, internal_label):
		clone = self.data.clone()
		self.recursive_meta_helper(internal_label, clone)


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


