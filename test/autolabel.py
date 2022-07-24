import torch
from torchvision import transforms
from PIL import Image
from matplotlib import pyplot as plt
import torchvision.autolabel as al

convert_tensor = transforms.ToTensor()
random_rotation = transforms.RandomRotation(degrees=360)

img = Image.open('assets/encode_jpeg/grace_hopper_517x606.jpg')

img = convert_tensor(img)
img = al.LabelTensor(img)
img = random_rotation(img)

plt.imshow(img.permute(1, 2, 0))
plt.show()


