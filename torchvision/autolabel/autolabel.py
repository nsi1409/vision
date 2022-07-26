import torch
from .. import transforms
import math
from matplotlib import pyplot as plt
from PIL import Image


class ImageTensor(torch.Tensor):
	def __init__(self, *args, **kwargs):
		self.transformation_manifest = []
		torch.Tensor.__init__(*args, **kwargs)

	def append_transform(self, transform):
		self.transformation_manifest.append(transform)

	def plot(self, label=None):
		plt.imshow(self.permute(1, 2, 0))
		if label is not None:
			for index, point in label.internal_label.items():
				plt.scatter(x=[point.x], y=[point.y], c='r', s=7)
				plt.text(point.x+2, point.y-2, str(index), c='r', fontsize=9)
		plt.show()

def frompath(path):
	img = Image.open(path)
	convert = transforms.ToTensor()
	img = convert(img)
	img = ImageTensor(img)
	return img


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
		self.meta_tensor = []
		torch.Tensor.__init__(*args, **kwargs)

	def add_to_internal_dictionary(self, current, clone):
		if not current.index in self.internal_label:
			self.internal_label[current.index] = MetaLabelClass(current.index)
			current.add_to_meta(self.internal_label[current.index], clone.item())
		else:
			current.add_to_meta(self.internal_label[current.index], clone.item())

	def recursive_meta_helper(self, current, clone):
		if not isinstance(current, list):
			self.add_to_internal_dictionary(current, clone)
		else:
			for i in range(len(current)):
				self.recursive_meta_helper(current[i], clone[i])

	def recursive_extractor(self, current):
		if not isinstance(current, list):
			current.extract_from_meta(self.internal_label[current.index])
		else:
			for i in range(len(current)):
				self.recursive_extractor(current[i])

	def metalabel(self, internal_label):
		self.meta_tensor = internal_label
		clone = self.data.clone()
		self.recursive_meta_helper(internal_label, clone)
		self.recursive_extractor(self.meta_tensor)


class RandomRotation(transforms.RandomRotation):
	def __init__(self, *args, **kwargs):
		transforms.RandomRotation.__init__(self, *args, **kwargs)

	def forward(self, img):
		transformations = img.transformation_manifest
		tensor = transforms.RandomRotation.forward(self, img)
		input_tensor = ImageTensor(tensor)
		input_tensor.transformation_manifest = transformations
		input_tensor.transformation_manifest.append(['rotation', (math.pi * self.get_params(self.degrees)) / 180])
		return input_tensor


