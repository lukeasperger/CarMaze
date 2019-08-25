import numpy as np
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from PIL import ImageDraw
import string

def segment_maze(maze_height, maze_width, filename):
    im = Image.open(filename)
    im_arr = np.array(im.resize((maze_width*224, maze_height*224)))
    height, width, _ = im_arr.shape

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
            img_tens = transform(img)
            img_tens = img_tens.unsqueeze(0)
            scores = resnet18.forward(img_tens)
            score_arr = scores.detach().numpy()
            labels[i].append(imagenet_dict[score_arr.argmax()])

            # compare triceratops score to sports car score
            dyno_score = np.max([score_arr[0, 46], score_arr[0, 51]]) # looks at various lizards / dinosaurs
            car_score = np.max([score_arr[0, 817], score_arr[0,864]]) # looks at various car types
            if car_score > dyno_score:
                car_grid[i, j] = 1

    return car_grid

def try_step(next_coord, labeled_grid, ending_coord, path):
    maze_height, maze_width = labeled_grid.shape
    if next_coord[0] < maze_height and next_coord[1] < maze_width and next_coord[0] >= 0 and next_coord[1] >= 0 and \
            next_coord not in path:
        if labeled_grid[next_coord]:
            path.append(next_coord)
        else:
            return path, 0
    else:
        return path, 0

    # base case, return 1 to signify success
    if (next_coord == ending_coord):
        return path, 1

    left_path, success = try_step((next_coord[0], next_coord[1] - 1), labeled_grid, ending_coord, path.copy())
    if success:
        return left_path, 1

    right_path, success = try_step((next_coord[0], next_coord[1] + 1), labeled_grid, ending_coord, path.copy())
    if success:
        return right_path, 1

    up_path, success = try_step((next_coord[0] - 1, next_coord[1]), labeled_grid, ending_coord, path.copy())
    if success:
        return up_path, 1

    down_path, success = try_step((next_coord[0] + 1, next_coord[1]), labeled_grid, ending_coord, path.copy())
    if success:
        return down_path, 1

    return path, 0

def find_path(labeled_grid, starting_coord, ending_coord):
    path = []
    path, _ = try_step(starting_coord, labeled_grid, ending_coord, path.copy())
    return path

def draw_path(path, filename, maze_height, maze_width):
    im = Image.open(filename)
    frame_h = im.size[1] / maze_height
    frame_w = im.size[0] / maze_width

    img_coords = [tuple([int(a[1]*frame_w + frame_w/2), int(a[0]*frame_h + frame_h/2)]) for a in path]
    draw = ImageDraw.Draw(im)
    draw.line(img_coords, fill='red', width=4)
    im.show()
    im.save("solved_maze.png")

# Set maze parameters here
maze_height = 8
maze_width = 14
starting_coord = (3, 13)
ending_coord = (3, 0)
filename = "maze.png"

# Read image into maze
grid = segment_maze(maze_height, maze_width, filename)
imagenet_dict = read_labels()
labeled_grid = label_images(grid, maze_height, maze_width, imagenet_dict)

# Solve maze
path = find_path(labeled_grid, starting_coord, ending_coord)
draw_path(path, filename, maze_height, maze_width)