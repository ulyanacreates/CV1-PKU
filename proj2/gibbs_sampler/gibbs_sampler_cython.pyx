# gibbs_sampler_cython.pyx
import numpy as np
cimport numpy as cnp
from libc.stdlib cimport malloc, free
import cv2
from cython cimport boundscheck, wraparound
from tqdm import tqdm
import os

from main import gibbs_sampler, mse_calculator

@boundscheck(False)  # Disable bounds checking for speed
@wraparound(False)    # Disable negative indexing
def optimized_loop(int sweep, int img_height, int img_width, 
                   cnp.ndarray[cnp.float64_t, ndim=3] mask, 
                   cnp.ndarray[cnp.float64_t, ndim=3] distort, 
                   float energy, float beta, const char* norm, const char*  name, const char* size, cnp.ndarray[cnp.float64_t, ndim=3] ori):
    
    cdef int i, j, s
    cdef int channel = 2  # Red channel index
    save_path = f"./result/{name}/{norm}/{size}test"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    pbar = tqdm(total=sweep, desc="Processing Loop")

    with open(f"testing/debug{size}.txt", "w") as f:
        for s in range(sweep):
            f.write(f'entering sweep {s}, value of beta: {beta}, energy: {energy}\n')
            for i in range(img_height):
                for j in range(img_width):
                    # Only change the red channel
                    if mask[i, j, channel] == 255:
                        f.write(f'the value at the distorted pixel rn is: {distort[i, j, channel]} at position {i}, {j}\n')
                        distort[:, :, channel] = gibbs_sampler(distort[:, :, channel], [i, j], energy, beta, norm)
                        f.write(f'the value at the new one is: {distort[i, j, channel]} at position {i}, {j}\n')

            beta += 4
            print('saved location is:', f"{save_path}/gibbs_{s}.bmp")
            cv2.imwrite(f"{save_path}/gibbs_{s}.bmp", distort)
            pbar.update(1)
            f.write(f"{mse_calculator(distort[:, :, channel], ori[:, :, channel])}\n")
        pbar.close()
