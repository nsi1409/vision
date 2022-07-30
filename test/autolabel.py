import torch
from torchvision import transforms
import torchvision.autolabel as al

path = 'assets/encode_jpeg/grace_hopper_517x606.jpg'
al_rot = al.RandomRotation(degrees=180)

initial = [[226, 192], [310, 189]]

label = al.LabelTensor(initial)
label.metalabel([[al.X(0), al.Y(0)], [al.X(1), al.Y(1)]])

print(type(label.meta_tensor[0][0]))
print(label.meta_tensor[0][0])

img = al.frompath(path)
print(f'shape: {img.size()}')
label.scale(img)

#img = al_rot(img)
#img = al_rot(img)

print(f'transformation manifest: {img.transformation_manifest}')

label.mutate(img)
label.unscale()

#print(label.meta_tensor[0][0])
label.pull()
#print(label.meta_tensor[0][0])

print(f'initial: {initial}')
print(f'init_tensor: {label.init_tensor}')

img.plot(label=label)


