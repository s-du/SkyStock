import numpy as np
import matplotlib.pyplot as plt
import sys
from fastsam import FastSAM, FastSAMPrompt
import resources as res


sys.path.append("..")

# Defining sam model
sam_checkpoint = res.find('other/FastSAM-x.pt')
model_type = "vit_h"
DEVICE = "cpu"
model = FastSAM(sam_checkpoint)

def do_sam(IMAGE_PATH, input_points, input_labels):
    # input_point = np.array([[x, y]])
    # input_label = np.array([1])

    everything_results = model(IMAGE_PATH, device=DEVICE, retina_masks=True, conf=0.4, iou=0.9, )
    prompt_process = FastSAMPrompt(IMAGE_PATH, everything_results, device=DEVICE)

    ann = prompt_process.point_prompt(points=input_points, pointlabel=input_labels)

    return prompt_process, ann

def create_mask_image(ann, output):
    # New array to store RGB values
    mask = np.zeros((400, 300, 3), dtype=np.uint8)

    # Generate a random RGB color
    random_color = np.random.randint(0, 256, 3)

    # Replace False values with black and True values with random color
    mask[ann.squeeze()] = random_color
    mask[~ann.squeeze()] = [0, 0, 0]

    return mask

