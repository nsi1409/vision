import torch
from torchvision import transforms
from PIL import Image
from matplotlib import pyplot as plt

convert_tensor = transforms.ToTensor()
random_rotation = transforms.RandomRotation(degrees=360, angle_tuple=True)

img = Image.open('assets/encode_jpeg/grace_hopper_517x606.jpg')

img = convert_tensor(img)
img, degrees = random_rotation(img)

print(degrees)

plt.imshow(img.permute(1, 2, 0))
plt.show()


