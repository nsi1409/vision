import torch

class InputTensor(torch.Tensor):
	def __init__(self, *args, **kwargs):
		self.transformation_manifest = []
		torch.Tensor.__init__(*args, **kwargs)

class LabelTensor(torch.Tensor):
	def __init__(self, *args, **kwargs):
		torch.Tensor.__init__(*args, **kwargs)


