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

	def add_self_to_meta(self, meta):
		meta.x = self.x

	def mutate(self, trans):
		if trans[0] == 'rotation':
			self.x = (self.x * math.cos(trans[1])) - (self.y * math.sin(trans[1]))

	def pull(self):
		return self.x

class Y(MetaLabelClass):
	def add_to_meta(self, meta, value):
		meta.y = value

	def add_self_to_meta(self, meta):
		meta.y = self.y

	def mutate(self, trans):
		if trans[0] == 'rotation':
			self.y = (self.x * math.sin(trans[1])) + (self.y * math.cos(trans[1]))

	def pull(self):
		return self.y

class Cl(MetaLabelClass):
	def add_to_meta(self, meta, value):
		meta.cl = value

	def add_self_to_meta(self, meta):
		meta.cl = self.cl

	def mutate(self, trans):
		pass

	def pull(self):
		return self.cl


class LabelTensor(torch.Tensor):
	def __init__(self, *args, **kwargs):
		self.internal_label = {}
		self.meta_tensor = []
		self.init_tensor = []
		self.x_scale = None
		self.y_scale = None
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

	def recursive_mutator(self, current, trans):
		if not isinstance(current, list):
			current.mutate(trans)
			current.add_self_to_meta(self.internal_label[current.index])	
		else:
			for i in range(len(current)):
				self.recursive_mutator(current[i], trans)

	def mutate(self, img, erase=True):
		for i in range(len(img.transformation_manifest)):
			self.recursive_mutator(self.meta_tensor, img.transformation_manifest[i])
			self.recursive_extractor(self.meta_tensor)
		if (erase == True):
			img.transformation_manifest = []

	def scale(self, img):
		dim = list(img.size())
		self.x_scale = dim[2]
		self.y_scale = dim[1]
		for index, point in self.internal_label.items():
			point.x = (point.x / (self.x_scale-1)) - 0.5
			point.y = (point.y / (self.y_scale-1)) - 0.5
		self.recursive_extractor(self.meta_tensor)

	def unscale(self):
		for index, point in self.internal_label.items():
			point.x = (point.x + 0.5) * (self.x_scale-1)
			point.y = (point.y + 0.5) * (self.y_scale-1)
		self.recursive_extractor(self.meta_tensor)

	def recursive_meta_pull(self, current, outp, index):
		if not isinstance(current, list):
			outp[index] = current.pull()
		else:
			if not isinstance(current[0], list):
				last = True
			else:
				last = False
			for i in range(len(current)):
				outp.append([])
				deeper = outp[i]
				if last:
					deeper = outp
				self.recursive_meta_pull(current[i], deeper, i)

	def pull(self):
		self.init_tensor = []
		self.recursive_meta_pull(self.meta_tensor, self.init_tensor, 0)


class RandomRotation(transforms.RandomRotation):
	def __init__(self, *args, **kwargs):
		transforms.RandomRotation.__init__(self, *args, **kwargs)

	def forward(self, img):
		transformations = img.transformation_manifest
		tensor = transforms.RandomRotation.forward(self, img)
		input_tensor = ImageTensor(tensor)
		input_tensor.transformation_manifest = transformations
		input_tensor.transformation_manifest.append(['rotation', -1 * math.radians(self.angle)])
		return input_tensor


