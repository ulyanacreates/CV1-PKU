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
from PIL import Image
from scipy import signal
import pyximport
pyximport.install()
pyximport.install(setup_args={"include_dirs": [np.get_include()]})
import warnings
from sklearn.metrics import root_mean_squared_error

from gibbs_sampler_loop import run_optimized_loop

# def warning_handler(message, category, filename, lineno, file=None, line=None):
#     print(f"\nWarning occurred: {message}")
#     print(f"Category: {category.__name__}")
#     print(f"Location: {filename}, line {lineno}")

#     pdb.set_trace()

# warnings.showwarning = warning_handler

def mse_calculator(x, o):
    rmse = root_mean_squared_error(x, o, multioutput='uniform_average')
    mse = rmse**2
    return mse


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
    # prep: calculate original values for grads around all original pixels for easier calculation
    nabla_x = conv(img, filter = np.array([[-1,1]]).astype(np.float64)) # forward difference for both 
    nabla_y = conv(img, filter = np.array([[-1],[1]]).astype(np.float64))
    for value in range(256): ## for any value of pixels 0~256
        # assume that the pixel being inspected becomes this intensity
        img[loc[0], loc[1]] = value
        # because the change of this pixel only affects the value of neighbouring 4 pixels, we need to only 
        # calculate for them. the rest of the pixels can be represented by energy - difference between the altered 
        # value for pixels then and now 
        # 1. calculate the changed values for grads around this pixel; x: left and right pixels, y: top and bottom pixels
        # since we're calculating the grads with forward diff -> loc pos pixel change only influences the prev pixel and itself
        left = (loc[0] - 1, loc[1]) if loc[0] - 1 > 0 else (loc[0] + img_width, loc[1])
        right = (loc[0] + 1, loc[1]) if loc[0] + 1 < img_width else (loc[0] - img_width, loc[1])
        top = (loc[0], loc[1] - 1) if loc[1] - 1 > 0 else (loc[0], loc[1] + img_height)
        bottom = (loc[0], loc[1] + 1) if loc[1] + 1 < img_height else (loc[0], loc[1] - img_height)
        # new_left = value - img[left[0], left[1]]
        new_loc_x = img[right[0], right[1]] - value
        # new_bottom = value - img[bottom[0], bottom[1]]
        new_loc_y = img[bottom[0], bottom[1]] - value
        # 2. substitute the original values for energy at this locations and add the new ones to update energy aspect
        energy += np.abs(-nabla_x[loc[0], loc[1]] - nabla_y[loc[0], loc[1]] + (new_loc_x  + new_loc_y))
        energy_list[value] = energy

    # normalize the energy
    energy_list = energy_list - energy_list.min()
    energy_list = energy_list / energy_list.sum()
    # pdb.set_trace()


    # calculate the conditional probability
    probs = np.exp(-energy_list * beta)
    # normalize the probs
    probs = probs / probs.sum()

    try:
        # inverse_cdf and updating the img
        # TODO
        # we sample from the conditional probability the same way we did from 1/p^3 in the last paper 
        # so before the sampling, we just need to have the distro for all possible pixel values for this location (energy list) to sample from it
        cdf = np.cumsum(probs)
        random_value = random.random()
        new_pixel_value = np.searchsorted(cdf, random_value)
        img[loc[0], loc[1]] = new_pixel_value
        # need to adjust the energy according to this value change
        # left = (loc[0] - 1, loc[1]) if loc[0] - 1 > 0 else (loc[0] + img_width, loc[1])
        # right = (loc[0] + 1, loc[1]) if loc[0] + 1 < img_width else (loc[0] - img_width, loc[1])
        # top = (loc[0], loc[1] - 1) if loc[1] - 1 > 0 else (loc[0], loc[1] + img_height)
        # bottom = (loc[0], loc[1] + 1) if loc[1] + 1 < img_height else (loc[0], loc[1] - img_height)
        # # new_left = new_pixel_value - img[left[0], left[1]]
        # new_loc_x = img[right[0], right[1]] - new_pixel_value
        # # new_bottom = new_pixel_value - img[bottom[0], bottom[1]]
        # new_loc_y = img[bottom[0], bottom[1]] - new_pixel_value
        # energy += np.abs(nabla_x[loc[0], loc[1]] - nabla_y[loc[0], loc[1]] +
        #             (new_loc_x + new_loc_y))
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
    # TODO Have checked, should be correct (compared with proj1 result)
    # the whole filter shape check is only for the plotting of horiz grad for testing 
    filtered_image = np.zeros_like(image)
    # if filter.shape == (1, 2):
    #     filtered_image = signal.convolve2d(image, filter, boundary='wrap')
    #     x_grad = Image.fromarray(filtered_image.astype(np.uint8))
    #     x_grad.save('testing/horiz_grad.jpg')
    # elif filter.shape == (2, 1):
    filtered_image = signal.convolve2d(image, filter, boundary='wrap')
        # y_grad = Image.fromarray(filtered_image.astype(np.uint8))
        # y_grad.save('testing/vertical_grad.jpg')
    return filtered_image

def main():
    # read the distorted image and mask image
    name = "sce"
    size = "big"

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
    filtered_img = conv(red_channel, np.array([[-1,1]]).astype(np.float64)) #gradient relative to x
    energy += np.sum(np.abs(filtered_img), axis = (0,1)) # energy relative to x
    # pdb.set_trace()
    # calculate nabla_y
    filtered_img = conv(red_channel, np.array([[-1],[1]]).astype(np.float64)) # gradient relative to y
    energy += np.sum(np.abs(filtered_img), axis = (0,1)) #energy relative to y
    # pdb.set_trace()
    norm = "L2"
    beta = 5
    img_height, img_width, _ = distort.shape

    sweep = 100
    # Optimize with cython: 
    run_optimized_loop(sweep, img_height, img_width, mask, distort, energy, beta, norm, name, size, ori)
    # for s in tqdm(range(sweep)):
    #     for i in range(img_height):
    #         for j in range(img_width):
    #             # only change the channel red
    #             if mask[i,j,2] == 255:
    #                 distort[:,:,2] = gibbs_sampler(distort[:,:,2], [i,j], energy, beta, norm)
    #                 print('calculating red channel values, sweep: ', s, 'i: ', i, 'j: ', j)
    #     # TODO
    #     beta += 0.01

        # save_path = f"./result/{name}/{norm}/{size}"
        # if not os.path.exists(save_path):
        #     os.makedirs(save_path)
        # cv2.imwrite(f"{save_path}/gibbs_{s}.bmp", distort)
    




if __name__ == "__main__":
    # with warnings.catch_warnings():
    main()







        

