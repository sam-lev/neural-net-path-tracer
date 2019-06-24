from .utils.conditional_augmentors import (
    mean_filter,
    variance_filter,
    median_filter,
    minimum_filter,
    maximum_filter,
    gaussian_blur_filter,
    difference_of_gaussians_filter,
    sobel_filter,
    laplacian_filter,
    neighbor_filter,
)

import os
import numpy as np
from skimage import io
from scipy.misc import imsave
import matplotlib.pyplot as plt
import cv2 # img processing w/ ppm
from PIL import Image
import imageio # for saving augmented img as ppm

def init_paths():
    dataset_path =  os.path.join('path_tracer','raytracingtherestofyourlife','dataset', 'filter_set','train')
    
    gt_path = os.path.join('.',dataset_path,'gt')
    direct_path = os.path.join(dataset_path,'direct')
    depth_path = os.path.join(dataset_path,'depth')
    albedo_path = os.path.join(dataset_path,'albedo')
    normal_path = os.path.join(dataset_path,'normal')
    diff_gauss_path = os.path.join(dataset_path,'diff_gaussian')

def is_image(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg",".pnm", "ppm"])

#gt_imgs = [os.path.join(gt_path, im_name) for im_name in os.listdir(gt_path) if is_image(im_name)]

def augment_groundtruth_images(im_dir = None):

    dataset_path =  os.path.join('path_tracer','raytracingtherestofyourlife','dataset', 'filter_set','train')
    
    gt_path = os.path.join('.',dataset_path,'gt')
    direct_path = os.path.join(dataset_path,'direct')
    depth_path = os.path.join(dataset_path,'depth')
    albedo_path = os.path.join(dataset_path,'albedo')
    normal_path = os.path.join(dataset_path,'normal')
    diff_gauss_path = os.path.join(dataset_path,'diff_gaussian')
    
    if im_dir:
        gt_path = im_dir
        gt_imgs = [os.path.join(gt_path, im_name) for im_name in os.listdir(gt_path) if is_image(im_name)]
    else:
        gt_imgs = [os.path.join(gt_path, im_name) for im_name in os.listdir(gt_path) if is_image(im_name)]
        
    for img in gt_imgs:

        im_name = img.split('/')[-1]
        
        #raw_img =  io.imread(img, as_gray=True)
        im_matrix = cv2.imread(img,cv2.IMREAD_UNCHANGED)
        # for gray ,0) and Image ,'P')
        #print(im_matrix.shape)

        #testt.show()
        blur_sigma1 = 5
        blur_sigma2 = 1
        diff_gauss_image = difference_of_gaussians_filter(im_matrix,blur_sigma1, blur_sigma2)
        blurred_img = gaussian_blur_filter(im_matrix, 2)
        #Image.fromarray(blurred_img).show()
        
        laplacian_image = laplacian_filter(im_matrix)#blurred_img)
        sobel_image = sobel_filter(im_matrix)
        var_image = variance_filter(im_matrix,4)
        nbr_image = neighbor_filter(im_matrix)
        var_image = variance_filter(im_matrix, 4)

        def to_gray(im_arr):
            print(im_arr[...,:3].shape)
            return np.dot(im_arr[...,:3], [0.299, 0.587, 0.114])
        
        #im_temp = Image.fromarray(diff_gauss_img, 'RGB')
        #im_temp.show()
        
        # For imageio
        
        #imageio.plugins.freeimage.download()
        #imageio.imwrite(
        imsave(os.path.join(direct_path, im_name), laplacian_image)
        imsave(os.path.join(depth_path, im_name), sobel_image)
        imsave(os.path.join(normal_path, im_name), var_image)
        #imsave(os.path.join(albedo_path, im_name), nbr_image)
        imsave(os.path.join(albedo_path, im_name), diff_gauss_image)
        
