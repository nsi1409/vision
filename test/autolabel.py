import torch
from torchvision import transforms
from PIL import Image
from matplotlib import pyplot as plt
import torchvision.autolabel as al
import pdb

convert_tensor = transforms.ToTensor()
random_rotation = transforms.RandomRotation(degrees=360)

img = Image.open('assets/encode_jpeg/grace_hopper_517x606.jpg')

img = convert_tensor(img)
_ = random_rotation(img)
img = al.InputTensor(img)
#print(f'transformation manifest: {img.transformation_manifest}')
#print(f'input type: {type(img)}')
#pdb.set_trace()
img = random_rotation(img)
#print(f'output type: {type(img)}')

#print(f'transformation manifest: {img.transformation_manifest}')

plt.imshow(img.permute(1, 2, 0))
plt.show()


