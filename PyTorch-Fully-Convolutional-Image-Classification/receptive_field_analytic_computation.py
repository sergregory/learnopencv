import cv2
import torch
import torch.nn as nn
from FullyConvolutionalResnet18 import FullyConvolutionalResnet18


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


def get_receptive_filed_calc_by_module(module):
    mapping = {
        nn.Conv2d: sliding_window_receptive_field_size_hook,
        nn.BatchNorm2d: no_change_receptive_field_size_hook,
        nn.AvgPool2d: sliding_window_receptive_field_size_hook,
        nn.MaxPool2d: sliding_window_receptive_field_size_hook,
        nn.ReLU: no_change_receptive_field_size_hook,
    }
    return mapping.get(type(module))


def compute_rf_analytically(image):
    model = FullyConvolutionalResnet18()
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

    image_height, image_width, image_channels = image.shape
    input = torch.zeros(size=(1, image_channels, image_height, image_width))
    input[0, 0, 0, 1] = 1
    input[0, 0, 0, 6] = 1
    out = model(input)
    with torch.no_grad():
        print(
            "Receptive field size for the pixel in the output feature map:",
            int(torch.max(out, dim=1)[0][0, 0, 0].item()),
        )
    for handler in hook_handlers:
        handler.remove()


def main():
    # Read image
    image_path = 'camel.jpg'
    image = cv2.imread(image_path)
    compute_rf_analytically(image)


if __name__ == "__main__":
    main()
