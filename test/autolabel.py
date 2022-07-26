import torch
from torchvision import transforms
import torchvision.autolabel as al

path = 'assets/encode_jpeg/grace_hopper_517x606.jpg'
al_rot = al.RandomRotation(degrees=180)

label = al.LabelTensor([[226, 192], [310, 189]])
label.metalabel([[al.X(0), al.Y(0)], [al.X(1), al.Y(1)]])

print(type(label.meta_tensor[0][0]))
print(label.meta_tensor[0][0])

img = al.frompath(path)
img = al_rot(img)
img = al_rot(img)

print(f'transformation manifest: {img.transformation_manifest}')

img.plot(label=label)


