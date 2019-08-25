import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import string

def segment_maze(maze_height, maze_width):
    im = Image.open("maze.png")
    im = im.resize((maze_width*224, maze_height*224))
    im_arr = np.array(im)
    height = im_arr.shape[0]
    width = im_arr.shape[1]

    frame_h = height//maze_height
    frame_w = width//maze_width
    grid = np.zeros((maze_height, maze_width, frame_h, frame_w, 4), dtype="uint8")
    for i in range(maze_width):
        for j in range(maze_height):
            grid[j, i] = im_arr[j*frame_h:(j+1)*frame_h, i*frame_w:(i+1)*frame_w, :]
    return grid

def read_labels():
    label_dict = {}
    table = str.maketrans({key: None for key in string.punctuation})
    with open("imagenet1000_clsidx_to_labels.txt") as f:
        for line in f:
            (key, val) = line.split(':')
            val = val.translate(table)
            label_dict[int(key)] = val
    return label_dict


def label_images(grid, maze_height, maze_width, imagenet_dict):
    resnet18 = models.wide_resnet101_2(pretrained=True)
    resnet18.eval()

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])

    car_grid = np.zeros([maze_height, maze_width], dtype='uint8')
    labels = []
    for i in range(maze_height):
        labels.append([])
        for j in range(maze_width):
            img = grid[i, j]
            img = Image.fromarray(img[:, :, :3])
            trans = transforms.ToTensor()
            img_tens = transform(img)
            img_tens = img_tens.unsqueeze(0)
            scores = resnet18.forward(img_tens)
            score_arr = scores.detach().numpy()
            labels[i].append(imagenet_dict[score_arr.argmax()])

            # compare triceratops score to sports car score
            dyno_score = np.average(score_arr[0, 38:52]) # looks at all types of lizards / dinosaurs
            car_score = np.average(score_arr[0, 817], score_arr[0,864], score_arr[0,829], score_arr[0,751],
                                   score_arr[0, 436], score_arr[0,479])
            if car_score > dyno_score:
                car_grid[i, j] = 1

    return car_grid



# Set maze parameters here
maze_height = 8
maze_width = 14

# Read image into maze
grid = segment_maze(maze_height, maze_width)
imagenet_dict = read_labels()
labeled_grid = label_images(grid, maze_height, maze_width, imagenet_dict)

# Solvd maze
print("done")