'''
This is the main file for the project 2's Second method PDE
'''
import cv2
import numpy as np
import random
from tqdm import tqdm
import torch
from torch.nn.functional import conv2d, pad
import os
from sklearn.metrics import root_mean_squared_error
import matplotlib.pyplot as plt




def pde(img, loc, beta, f):
    ''' 
    The function to perform the pde update for a specific pixel location
    Parameters:
        1. img: the image to be processed, numpy array
        2. loc: the location of the pixel, loc[0] is the row number, loc[1] is the column number
        3. beta: learning rate
    Return:
        img: the updated image
    '''
    
    # TODO
    # do pde only for this specific pixel, the rest of the pixels dont really change 
    original_pixel = img[loc[0], loc[1]]
    laplacian_value = (
        img[loc[0]+1, loc[1]] + img[loc[0]-1, loc[1]] + 
        img[loc[0], loc[1]+1] + img[loc[0], loc[1]-1] - 
        4 * img[loc[0], loc[1]]
    )
    # f.write(f'laplacian value is: {laplacian_value}\n')
    img[loc[0], loc[1]] += beta*laplacian_value
    # f.write(f'previous value at loc: {original_pixel}, new estimated value is: {img[loc[0], loc[1]]}\n')
    img[loc[0], loc[1]] = np.clip(img[loc[0], loc[1]], 0, 255)
    return img


def main():
    # read the distorted image and mask image
    name = "stone"
    size = "small"

    distorted_path = f"../image/{name}/{size}/imposed_{name}_{size}.bmp"
    mask_path = f"../image/mask_{size}.bmp"
    ori_path = f"../image/{name}/{name}_ori.bmp"


    # read the BGR image
    distort = cv2.imread(distorted_path).astype(np.float64)
    mask = cv2.imread(mask_path).astype(np.float64)
    ori = cv2.imread(ori_path).astype(np.float64)



    beta = 0.001
    img_height, img_width, _ = distort.shape

    # f = open("testing/record_method2.txt", "w")
    sweep = 100
    # mse = []
    for s in tqdm(range(sweep)):
        f.write(f'SWEEP: {s}\n')
        for i in range(img_height):
            for j in range(img_width):
                # only change the channel red
                # TODO
                if mask[i,j,2] == 255:
                    distort[:, :, 2] = pde(distort[:, :, 2], [i, j], beta, f)
        # TODO
        beta += 0.0065    
        # rmse = root_mean_squared_error(distort[:, :, 2], ori[:, :, 2], multioutput='raw_values')
        # mse.append(rmse**2)
        if s % 10 == 0:
            save_path = f"./result/{name}/{size}"
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            cv2.imwrite(f"{save_path}/pde_{s}.bmp", distort)
    # open("testing/record_method2.txt", "w").close()

    # plt.figure(figsize=(10, 5))
    # plt.plot(mse, marker='o', linestyle='-', color='b')
    # plt.title("Plot of 100 sweeps")
    # plt.xlabel("Sweep")
    # plt.ylabel("MSE")
    # plt.grid(True)
    # plt.savefig('testing/plt.png', dpi=300)


if __name__ == "__main__":
    main()







        

