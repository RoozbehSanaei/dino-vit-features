import matplotlib.pyplot as plt
import torch
from cosegmentation import find_cosegmentation, draw_cosegmentation, draw_cosegmentation_binary_masks
from part_cosegmentation import find_part_cosegmentation, draw_part_cosegmentation
from correspondences import find_correspondences, draw_correspondences

images_paths = ['images/cat.jpg', 'images/ibex.jpg']  # Choose loading size
load_size = 360  # Choose loading size
layer = 11  # Choose layer of descriptor
facet = 'key'  # Choose facet of descriptor
bin = False  # Choose if to use a binned descriptor
thresh = 0.065  # Choose fg / bg threshold
model_type = 'dino_vits8'  # Choose model type
stride = 4  # Choose stride
elbow = 0.975  # Choose elbow coefficient for setting number of clusters
votes_percentage = 75  # Choose percentage of votes to make a cluster salient
remove_outliers = False  # Choose whether to remove outlier images
outliers_thresh = 0.7  # Choose threshold to distinguish inliers from outliers
sample_interval = 100  # Choose interval for sampling descriptors for training
low_res_saliency_maps = True  # Use low resolution saliency maps -- reduces RAM usage


num_parts = 4
# Number of crop augmentations to apply on each input image. Relevant for small sets.
num_crop_augmentations = 20
# If true, use three clustering stages instead of two. Relevant for small sets.
three_stages = True
# Elbow method for finding amount of clusters when using three clustering stages.
elbow_second_stage = 0.94


# Configuration:
# Choose image paths:
image_path1 = 'images/cat.jpg'
image_path2 = 'images/ibex.jpg'

# Choose number of points to output:
num_pairs = 10




with torch.no_grad():
    
     # computing cosegmentation
    seg_masks, pil_images = find_cosegmentation(images_paths, elbow, load_size, layer, facet, bin, thresh, model_type,
                                                stride, votes_percentage, sample_interval, remove_outliers,
                                                outliers_thresh, low_res_saliency_maps)

    figs, axes = [], []
    for pil_image in pil_images:
      fig, ax = plt.subplots()
      ax.axis('off')
      ax.imshow(pil_image)
      figs.append(fig)
      axes.append(ax)
    
    # saving cosegmentations
    binary_mask_figs = draw_cosegmentation_binary_masks(seg_masks)
    chessboard_bg_figs = draw_cosegmentation(seg_masks, pil_images)
    
    plt.show(block=True)


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


    points1, points2, image1_pil, image2_pil = find_correspondences(image_path1, image_path2, num_pairs, load_size, layer,
                                                                   facet, bin, thresh, model_type, stride)
    fig_1, ax1 = plt.subplots()
    ax1.axis('off')
    ax1.imshow(image1_pil)
    fig_2, ax2 = plt.subplots()
    ax2.axis('off')
    ax2.imshow(image2_pil)


    fig1, fig2 = draw_correspondences(points1, points2, image1_pil, image2_pil)
    plt.show(block=True)
    plt.close()
