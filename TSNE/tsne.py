import argparse
from tqdm import tqdm
import cv2
import torch
import random
import numpy as np
from sklearn.manifold import TSNE

from animals_dataset import AnimalsDataset, collate_skip_empty, colors_per_class
from resnet import Resnet


def fix_random_seeds():
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_model_outputs(dataset, batch, num_images):
    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    # initialize our implementation of ResNet
    model = Resnet(pretrained=True)
    model.eval()
    model.to(device)

    # read the dataset and initialize the data loader
    dataset = AnimalsDataset(dataset, num_images)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch, collate_fn=collate_skip_empty, shuffle=True)

    # we'll store the features as NumPy array of size num_images x feature_size
    outputs = None

    # we'll also store the image labels and paths to visualize them later
    labels = []
    image_paths = []

    for batch in tqdm(dataloader, desc='Running the model inference'):
        images = batch['image'].to(device)
        labels += batch['label']
        image_paths += batch['image_path']

        with torch.no_grad():
            output = model.forward(images)

        current_outputs = output.cpu().numpy()
        if outputs is not None:
            outputs = np.concatenate((outputs, current_outputs))
        else:
            outputs = current_outputs

    return outputs, labels, image_paths


def plot_legend(colors_per_class):
    width = 300

    offset = 30
    color_size = 30
    color_offset = 5

    row_height = color_size + 2 * color_offset
    num_colors = len(colors_per_class)

    height = row_height * num_colors

    legend = np.zeros((height, width, 3), np.uint8)
    legend.fill(255)

    for i, label in enumerate(sorted(colors_per_class)):
        color = colors_per_class[label]

        tl_x = offset + 1
        tl_y = row_height * i + color_offset

        br_x = tl_x + color_size
        br_y = tl_y + color_size

        legend = cv2.rectangle(
            legend,
            (tl_x, tl_y),
            (br_x, br_y),
            color=color,
            thickness=cv2.FILLED
        )
        legend = cv2.putText(
            legend,
            label,
            (br_x + offset, br_y - 2 * color_offset),
            fontFace=cv2.FONT_HERSHEY_TRIPLEX,
            fontScale=0.5,
            color=(0, 0, 0)
        )

    cv2.imshow('legend', legend)


def scale_to_01_range(x):
    value_range = (np.max(x) - np.min(x))
    starts_from_zero = x - np.min(x)
    return starts_from_zero / value_range


def scale_image(image, max_image_size):
    image_height, image_width, _ = image.shape

    scale = max(1, image_width / max_image_size, image_height / max_image_size)
    image_width = int(image_width / scale)
    image_height = int(image_height / scale)

    image = cv2.resize(image, (image_width, image_height))
    return image


def draw_rectangle_by_class(image, label):
    image_height, image_width, _ = image.shape

    # get the color corresponding to image class
    color = colors_per_class[label]
    image = cv2.rectangle(image, (0, 0), (image_width - 1, image_height - 1), color=color, thickness=5)

    return image


def compute_plot_coordinates(image, x, y, image_centers_area_size, offset):
    image_height, image_width, _ = image.shape

    # compute the image center coordinates on the plot
    center_x = int(image_centers_area_size * x) + offset
    center_y = int(image_centers_area_size * y) + offset

    # knowing the image center, compute the coordinates of the top left and bottom right corner
    tl_x = center_x - int(image_width / 2)
    tl_y = center_y - int(image_height / 2)

    br_x = tl_x + image_width
    br_y = tl_y + image_height

    return tl_x, tl_y, br_x, br_y


def visualize_tsne_plot(tsne, images, labels, plot_size=1000, max_image_size=100):
    # extract x and y coordinates representing the positions of the images on T-SNE plot
    tx = tsne[:, 0]
    ty = tsne[:, 1]

    # scale and move the coordinates so they fit [0; 1] range
    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)

    # we'll put the image centers in the central area of the plot
    # and use offsets to make sure the images fit the plot
    offset = max_image_size // 2
    image_centers_area_size = plot_size - 2 * offset

    tsne_plot = np.zeros((plot_size, plot_size, 3), np.uint8)
    tsne_plot.fill(255)

    # now we'll put a small copy of every image to its corresponding T-SNE coordinate
    for image_path, label, x, y in tqdm(
            zip(images, labels, tx, ty),
            desc='Building the T-SNE plot',
            total=len(images)
    ):
        image = cv2.imread(image_path)

        # scale the image to put it to the plot
        image = scale_image(image, max_image_size)

        # draw a rectangle with a color corresponding to the image class
        image = draw_rectangle_by_class(image, label)

        # compute the coordinates of the image on the scaled plot visualization
        tl_x, tl_y, br_x, br_y = compute_plot_coordinates(image, x, y, image_centers_area_size, offset)

        # put the image to its TSNE coordinates using numpy subarray indices
        tsne_plot[tl_y:br_y, tl_x:br_x, :] = image

    plot_legend(colors_per_class)

    cv2.imshow('T-SNE', tsne_plot)
    cv2.waitKey()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--path', type=str, default='data/raw-img')
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--num_images', type=int, default=500)
    args = parser.parse_args()

    fix_random_seeds()

    model_outputs, labels, image_paths = get_model_outputs(
        dataset=args.path,
        batch=args.batch,
        num_images=args.num_images
    )

    tsne = TSNE(n_components=2).fit_transform(model_outputs)

    visualize_tsne_plot(tsne, image_paths, labels)

if __name__ == '__main__':
    main()
