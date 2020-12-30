import os
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default=train_path, help='path to the dataset')
parser.add_argument('--output', type=str, default=output_path, help='path to the file list')
args = parser.parse_args()

ext = {'.jpg', '.png'}

# print("root path:", os.listdir(args.path))

images = []
for root, dirs, files in os.walk(args.path):
    print('loading ' + root)
    for file in files:
        if os.path.splitext(file)[1] in ext:
            images.append(os.path.join(root, file))

# images = sorted(images)
np.random.shuffle(images)
np.savetxt(args.output, images, fmt='%s')
