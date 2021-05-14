import torch
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

# A random number is generated at every time step for each pixel in the input image.
# If the random number is less than the normalized pixel value an output spike is generated
# out_2 = torch.mul(torch.le(torch.rand_like(x_input), torch.abs(x_input) * 1.0).float(), torch.sign(x_input))

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0),(0.5))])

trainset = datasets.MNIST('./MNIST_data/', download=True, train=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=False)

x_input, labels = next(iter(trainloader))

x_input = x_input[1, 0, :, :]
out_1 = torch.zeros_like(x_input)
out_2 = torch.zeros_like(x_input)


y = np.zeros(64)
for k in range(64):
    plt.subplot(131)
    plt.imshow(x_input)
    plt.subplot(132)
    out_2 = torch.mul(torch.le(torch.rand_like(x_input), torch.abs(x_input) * 1.0).float(), torch.sign(x_input))
    plt.imshow(out_2)
    plt.subplot(133)
    plt.plot(y[:k], "b-*")
    plt.xlim([0, 64])
    plt.ylim([-1, 1])
    plt.title(k)
    plt.show()

    plt.pause(0.1)
    out_1 = out_2
