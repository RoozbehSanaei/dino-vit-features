#@title Configuration:
#@markdown Choose image paths:
images_paths = ['images/cat.jpg', 'images/ibex.jpg'] #@param
#@markdown Choose loading size:
load_size = 360 #@param
#@markdown Choose layer of descriptor:
layer = 11 #@param
#@markdown Choose facet of descriptor:
facet = 'key' #@param
#@markdown Choose if to use a binned descriptor:
bin=False #@param
#@markdown Choose fg / bg threshold:
thresh=0.065 #@param
#@markdown Choose model type:
model_type='dino_vits8' #@param
#@markdown Choose stride:
stride=4 #@param
#@markdown Choose elbow coefficient for setting number of clusters
elbow=0.975 #@param
#@markdown Choose percentage of votes to make a cluster salient.
votes_percentage=75 #@param
#@markdown Choose interval for sampling descriptors for training
sample_interval=100 #@param
#@markdown Use low resolution saliency maps -- reduces RAM usage.
low_res_saliency_maps=True #@param
#@markdown number of final object parts.
num_parts=4 #@param
#@markdown number of crop augmentations to apply on each input image. relevant for small sets.
num_crop_augmentations=20 #@param
#@markdown If true, use three clustering stages instead of two. relevant for small sets.
three_stages=True #@param
#@markdown elbow method for finding amount of clusters when using three clustering stages.
elbow_second_stage=0.94 #@param


import matplotlib.pyplot as plt
import torch
from part_cosegmentation import find_part_cosegmentation, draw_part_cosegmentation

with torch.no_grad():

    # computing part cosegmentation
    parts_imgs, pil_images = find_part_cosegmentation(images_paths, elbow, load_size, layer, facet, bin, thresh, model_type,
                                                      stride, votes_percentage, sample_interval, low_res_saliency_maps,
                                                      num_parts, num_crop_augmentations, three_stages, elbow_second_stage)

    figs, axes = [], []
    for pil_image in pil_images:
      fig, ax = plt.subplots()
      ax.axis('off')
      ax.imshow(pil_image)
      figs.append(fig)
      axes.append(ax)

    # saving part cosegmentations
    part_figs = draw_part_cosegmentation(num_parts, parts_imgs, pil_images)

plt.show(block=True)