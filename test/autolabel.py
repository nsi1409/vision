import torch
from torchvision import transforms
from PIL import Image
from matplotlib import pyplot as plt
import torchvision.autolabel as al
import pdb

convert_tensor = transforms.ToTensor()
#random_rotation = transforms.RandomRotation(degrees=360)
al_rot = al.RandomRotation(degrees=180)

img = Image.open('assets/encode_jpeg/grace_hopper_517x606.jpg')

label = al.LabelTensor([[226, 192], [310, 189]])
label.meta_label([[al.X(0), al.Y(0)], [al.X(1), al.Y(1)]])

img = convert_tensor(img)
img = al.InputTensor(img)
#print(f'transformation manifest: {img.transformation_manifest}')
#print(f'input type: {type(img)}')
#pdb.set_trace()
#img = random_rotation(img)
img = al_rot(img)
img = al_rot(img)
print(f'output type: {type(img)}')

print(f'transformation manifest: {img.transformation_manifest}')

plt.imshow(img.permute(1, 2, 0))
plt.show()


