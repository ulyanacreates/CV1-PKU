'''
This is the main file for the project 2's first method Gibss Sampler
'''
import cv2
import numpy as np
import random
from tqdm import tqdm
import torch
import os
from torch.nn.functional import conv2d, pad
import pdb


def cal_pot(gradient, norm):
    ''' 
    The function to calculate the potential energy based on the selected norm
    Parameters:
        gradient: the gradient of the image, can be nabla_x or nabla_y, numpy array of size:(img_height,img_width, )
        norm: L1 or L2
    Return:
        A term of the potential energy
    '''
    if norm == "L1":
        return abs(gradient)
    elif norm == "L2":
        return gradient**2 
    else:
        raise ValueError("The norm is not supported!")




def gibbs_sampler(img, loc, energy, beta, norm):
    ''' 
    The function to perform the gibbs sampler for a specific pixel location
    Parameters:
        1. img: the image to be processed, numpy array
        2. loc: the location of the pixel, loc[0] is the row number, loc[1] is the column number
        3. energy: a scale, refers to the negative exponent
        4. beta: 1/(annealing temperature)
        5. norm: L1 or L2
    Return:
        img: the updated image
    '''
    
    energy_list = np.zeros((256,1))
    # get the size of the image
    img_height, img_width = img.shape


    # original pixel value
    original_pixel = img[loc[0], loc[1]]

    
    # TODO: calculate the energy
    for pixel_value in range(256):
        img[loc[0], loc[1]] = pixel_value
        nabla_x = conv(img, np.array([[-1, 1]]).astype(np.float64))
        nabla_y = conv(img, np.array([[1], [-1]]).astype(np.float64))
        energy_x = cal_pot(nabla_x[loc[0], loc[1]], norm)
        energy_y = cal_pot(nabla_y[loc[0], loc[1]], norm)
        total_energy = energy_x + energy_y
        energy_list[pixel_value] = total_energy

    # normalize the energy
    energy_list = energy_list - energy_list.min()
    energy_list = energy_list / energy_list.sum()



    # calculate the conditional probability
    probs = np.exp(-energy_list * beta)
    # normalize the probs
    probs = probs / probs.sum()

    try:
        # inverse_cdf and updating the img
        # TODO
        cumulative_probs = np.cumsum(probs)
        random_value = random.random()
        new_pixel_value = np.searchsorted(cumulative_probs, random_value)
        img[loc[0], loc[1]] = new_pixel_value
        print('try')
    except:
        raise ValueError(f'probs = {probs}')
    return img

def conv(image, filter):
    ''' 
    Computes the filter response on an image.
    Parameters:
        image: numpy array of shape (H, W)
        filter: numpy array of shape, can be [[-1,1]] or [[1],[-1]] or [[1,-1]] or [[-1],[1]] ....
    Return:
        filtered_image: numpy array of shape (H, W)
    '''

    # filtered_image = image
    filtered_image = np.zeros_like(image)
    H, W = image.shape
    # TODO
    for x in range(H):
        for y in range(W):
            # Apply convolution based on filter type (nabla_x or nabla_y)
            for i in range(filter.shape[0]):
                # Determine the neighbor position
                dx, dy = 0, 0  # defaults
                if filter.shape == (1, 2):  # nabla_x filter
                    dx, dy = 0, i - 1  # neighbors along x-axis
                elif filter.shape == (2, 1):  # nabla_y filter
                    dx, dy = i - 1, 0  # neighbors along y-axis

                # Apply periodic boundary conditions
                x_neighbor = (x + dx) % H
                y_neighbor = (y + dy) % W

                # Apply filter weight
                filtered_image[x, y] += filter[i, 0] * image[x_neighbor, y_neighbor] if filter.shape == (2, 1) else filter[0, i] * image[x_neighbor, y_neighbor]

    return filtered_image

def main():
    # read the distorted image and mask image
    name = "sce"
    size = "small"

    distorted_path = f"../image/{name}/{size}/imposed_{name}_{size}.bmp"
    mask_path = f"../image/mask_{size}.bmp"
    ori_path = f"../image/{name}/{name}_ori.bmp"


    # read the BGR image
    distort = cv2.imread(distorted_path).astype(np.float64)
    mask = cv2.imread(mask_path).astype(np.float64)
    ori = cv2.imread(ori_path).astype(np.float64)

    # calculate initial energy
    red_channel = distort[:,:,2]
    energy = 0

    #calculate nabla_x
    filtered_img = conv(red_channel, np.array([[-1,1]]).astype(np.float64))
    energy += np.sum(np.abs(filtered_img), axis = (0,1))

    # calculate nabla_y
    filtered_img = conv(red_channel, np.array([[-1],[1]]).astype(np.float64))
    energy += np.sum(np.abs(filtered_img), axis = (0,1))

    norm = "L2"
    beta = 0.1
    img_height, img_width, _ = distort.shape

    sweep = 100
    for s in tqdm(range(sweep)):
        for i in range(img_height):
            for j in range(img_width):
                # only change the channel red
                if mask[i,j,2] == 255:
                    distort[:,:,2] = gibbs_sampler(distort[:,:,2], [i,j], energy, beta, norm)
        # TODO
        beta += 0.01

        save_path = f"./result/{name}/{norm}/{size}"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        cv2.imwrite(f"{save_path}/gibbs_{s}.bmp", distort)



if __name__ == "__main__":
    main()







        

