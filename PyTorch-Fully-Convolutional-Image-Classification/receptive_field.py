import torch
import torch.nn as nn
from FullyConvolutionalResnet18 import FullyConvolutionalResnet18
from torchvision import models
from torch.hub import load_state_dict_from_url
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd
import matplotlib.pyplot as plt
import cv2
from torch import isfinite


LAYER_INDEX = 0


def sliding_window_receptive_field_size_hook(module, input, output):
    global LAYER_INDEX

    stride_prev, receptive_field_prev = get_prev_layer_data(input)
    stride_current = get_element(module.stride, 0)
    kernel_size = get_element(module.kernel_size, 0)
    receptive_field = receptive_field_prev + (kernel_size - 1) * stride_prev
    stride_total = stride_current * stride_prev
    output[0, LAYER_INDEX % output.size()[1], 0, 0] = receptive_field
    output[0, LAYER_INDEX % output.size()[1], 0, 1] = stride_total
    LAYER_INDEX += 1
    # print(type(module), stride_current, kernel_size, receptive_field)


def get_element(sequence, index):
    try:
        _ = iter(sequence)
    except TypeError:
        element = sequence
    else:
        element = sequence[index]
    return element


def get_prev_layer_data(input):
    input = torch.unique_consecutive(input[0], dim=1)
    receptive_field_max_index = int(torch.argmax(input[:, :, 0, 0]).item())
    receptive_field_prev = int(input[0, receptive_field_max_index, 0, 0].item())
    stride_prev = int(input[0, receptive_field_max_index, 0, 1].item())
    return stride_prev, receptive_field_prev


def no_change_receptive_field_size_hook(module, input, output):
    global LAYER_INDEX
    stride_prev, receptive_field_prev = get_prev_layer_data(input)
    output[0, LAYER_INDEX % output.size()[1], 0, 0] = receptive_field_prev
    output[0, LAYER_INDEX % output.size()[1], 0, 1] = stride_prev
    LAYER_INDEX += 1
    # print(type(module), receptive_field_prev)


def get_receptive_filed_calc_by_module(module):
    mapping = {
        nn.Conv2d: sliding_window_receptive_field_size_hook,
        nn.BatchNorm2d: no_change_receptive_field_size_hook,
        nn.AvgPool2d: sliding_window_receptive_field_size_hook,
        nn.MaxPool2d: sliding_window_receptive_field_size_hook,
        nn.ReLU: no_change_receptive_field_size_hook,
    }
    return mapping.get(type(module))


# Define the architecture by modifying resnet.
# Original code is here
# https://github.com/pytorch/vision/blob/b2e95657cd5f389e3973212ba7ddbdcc751a7878/torchvision/models/resnet.py
class FullyConvolutionalResnet50(models.ResNet):
    def __init__(self, num_classes=1000, pretrained=False, **kwargs):

        # Start with standard resnet18 defined here
        # https://github.com/pytorch/vision/blob/b2e95657cd5f389e3973212ba7ddbdcc751a7878/torchvision/models/resnet.py
        super().__init__(
            block=models.resnet.Bottleneck,
            layers=[3, 4, 6, 3],
            num_classes=num_classes,
            **kwargs
        )
        if pretrained:
            state_dict = load_state_dict_from_url(
                models.resnet.model_urls["resnet50"], progress=True
            )
            self.load_state_dict(state_dict)

        # Replace AdaptiveAvgPool2d with standard AvgPool2d
        # https://github.com/pytorch/vision/blob/b2e95657cd5f389e3973212ba7ddbdcc751a7878/torchvision/models/resnet.py#L153-L154
        self.avgpool = nn.AvgPool2d((7, 7))

        # Add final Convolution Layer.
        self.last_conv = torch.nn.Conv2d(
            in_channels=self.fc.in_features, out_channels=num_classes, kernel_size=1
        )
        self.last_conv.weight.data.copy_(
            self.fc.weight.data.view(*self.fc.weight.data.shape, 1, 1)
        )
        self.last_conv.bias.data.copy_(self.fc.bias.data)

        for module in self.modules():
            if isinstance(module, torch.nn.modules.BatchNorm2d):
                module.eval()

    # Reimplementing forward pass.
    # Replacing the following code
    # https://github.com/pytorch/vision/blob/b2e95657cd5f389e3973212ba7ddbdcc751a7878/torchvision/models/resnet.py#L197-L213
    def _forward_impl(self, x):
        # Standard forward for resnet18
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)

        x = self.last_conv(x)
        return x


def visualize_rf_with_backprop(model):
    model = model.train()
    for module in model.modules():
        try:
            nn.init.constant_(module.weight, 0.05) # inference overflows with ones
            nn.init.zeros_(module.bias)
            nn.init.zeros_(module.running_mean)
            nn.init.ones_(module.running_var)
        except:
            pass

        if isinstance(module, torch.nn.modules.BatchNorm2d):
            module.eval()

    img = torch.ones(size=(1, 3, 1000, 1000), requires_grad=True)

    out = model(img)
    grad = torch.zeros_like(out)
    # grad[0, 0, grad.size()[2] // 2, grad.size()[3] // 2] = 1
    grad[0, 0, 1, 1] = 1
    # img.grad.zero_()
    out.backward(gradient=grad)
    grad_np = img.grad[0, 0].data.numpy()
    grad_np = grad_np / np.amax(grad_np)
    # _, grad_np = cv2.threshold(grad_np, 1e-6, 1., cv2.THRESH_BINARY)
    # grad_np = np.ascontiguousarray(np.stack([grad_np, grad_np, grad_np], axis=2))
    # grad_np = np.transpose(grad_np, (1, 2, 0))
    # _, grad_positive = cv2.threshold(grad_np, 1e-3, 255, cv2.THRESH_BINARY)
    # _, grad_negative = cv2.threshold(grad_np, -1e-3, 255, cv2.THRESH_BINARY_INV)
    # grad_image = grad_negative + grad_positive
    # grad_image = grad_image.astype(np.uint8)
    # grad_image = np.ascontiguousarray(np.stack([grad_image, grad_image, grad_image], axis=0))
    # idx_nonzeros=np.where(grad_np!=0)

    fig, ax = plt.subplots(nrows=1, ncols=2, constrained_layout=True)

    ax[0].set_title('Receptive Field')
    ax[0].imshow(grad_np)

    ax[1].set_title('Non-zero Element Mask')
    ax[1].imshow(grad_np > 0)

    plt.show()


def main():
    # # model = FullyConvolutionalResnet18().eval()
    model = FullyConvolutionalResnet50()
    visualize_rf_with_backprop(model)

    hook_handlers = []
    for module in model.modules():
        hook = get_receptive_filed_calc_by_module(module)
        if hook is not None:
            hook_handlers.append(module.register_forward_hook(hook))
        try:
            nn.init.zeros_(module.weight)
            nn.init.zeros_(module.bias)
            nn.init.zeros_(module.running_mean)
            nn.init.zeros_(module.running_var)
        except:
            pass

    input = torch.zeros(size=(1, 3, 224 * 2, 224 * 2))
    input[0, 0, 0, 0] = 1
    input[0, 0, 0, 1] = 1
    out = model(input)
    with torch.no_grad():
        print(
            "Receptive field size for the pixel in the output feature map:",
            int(torch.max(out, dim=1)[0][0, 0, 0].item()),
        )
    for handler in hook_handlers:
        handler.remove()


if __name__ == "__main__":
    main()
