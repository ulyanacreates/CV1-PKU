'''
This is the code for project 1 question 2
Question 2: Verify the 1/f power law observation in natural images in Set A
'''
import numpy as np
import os
import math
import matplotlib.pyplot as plt
import cv2
import pdb
path = "./image_set/setA/"
colorlist = ['red', 'blue', 'black', 'green']
linetype = ['-', '-', '-', '-']
labellist = ["natural_scene_1.jpg", "natural_scene_2.jpg",
                 "natural_scene_3.jpg", "natural_scene_4.jpg"]

img_list = [cv2.imread(os.path.join(path,labellist[i]), cv2.IMREAD_GRAYSCALE) for i in range(4)]
def fft(img):
    ''' 
    Conduct FFT to the image and move the dc component to the center of the spectrum
    Tips: dc component is the one without frequency. Google it!
    Parameters:
        1. img: the original image
    Return:
        1. fshift: image after fft and dc shift
    '''
    fshift = img # Need to be changed
    # TODO: Add your code here
    transformed_img = np.fft.fft2(img)
    fshift = np.fft.fftshift(transformed_img)
    # magnitude_spectrum = 20 * np.log(np.abs(fshift))
    # plt.imshow(magnitude_spectrum, cmap='gray')
    # plt.title('Magnitude Spectrum')
    # plt.axis('off')
    # plt.savefig('testing/fft.jpg')
    return fshift

def amplitude(fshift):
    '''
    Parameters:
        1. fshift: image after fft and dc shift
    Return:
        1. A: the amplitude of each complex number
    '''

    A = fshift # Need to be changed
    # TODO: Add your code here
    A = np.abs(fshift)
    return A

def xy2r(x, y, centerx, centery):
    ''' 
    change the x,y coordinate to r coordinate
    '''
    rho = math.sqrt((x - centerx)**2 + (y - centery)**2)
    return rho

def cart2porl(A,img):
    ''' 
    Finish question 1, calculate the A(f) 
    Parameters: 
        1. A: the amplitude of each complex number
        2. img: the original image
    Return:
        1. f: the frequency list 
        2. A_f: the amplitude of each frequency
    Tips: 
        1. Use the function xy2r to get the r coordinate!
    '''
    centerx = int(img.shape[0] / 2)
    centery = int(img.shape[1] / 2)
    # build the r coordinate
    basic_f = 1
    max_r = min(centerx,centery)
    # the frequency coordinate
    f = np.arange(0,max_r + 1,basic_f)

    # the following process is to do the sampling for each frequency of f
    A_f = np.zeros_like(f) # Need to be changed
    A_f_cnt = np.zeros_like(f)
    A_f_sum = np.zeros_like(f)

    # TODO: Add your code here
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            rho = int(xy2r(x, y, centerx, centery)) # each frequency bin corresposnds to a polar coordinate value => sum over them & average
            if rho <= max_r:
                A_f_sum[rho] += np.abs(A[x, y])**2  
                A_f_cnt[rho] += 1  
    A_f = np.divide(A_f_sum, A_f_cnt, where=(A_f_cnt != 0))
    # pdb.set_trace()
    return f, A_f


def get_S_f0(A,img):
    ''' 
    Parameters:
        1. A: the amplitude of each complex number
        2. img: the original image
    Return:
        1. S_f0: the S(f0) list
        2. f0: frequency list
    '''
    centerx = int(img.shape[0] / 2)
    centery = int(img.shape[1] / 2)

    # S_f0 = np.zeros(100) # Need to be changed
    # f0 = np.arange(0,100,1) # Need to be changed

    # TODO: Add your code here
    basic_f = 1
    max_r = min(centerx,centery)
    f0 = np.arange(0, int(max_r/2), basic_f) ## since the domain is such that the frequency values lie bw f0 and 2f0, need to rescale the frequency bins to /2
    S_f0 = np.zeros_like(f0)
    for each in f0:
        lower_bound = each
        upper_bound = 2*each
        for f_val in range(max_r):
            if (f_val >= lower_bound) & (f_val <= upper_bound):
                # pdb.set_trace()
                S_f0[each] = np.sum(A[f_val]**2)
    return S_f0, f0
    
def main():
    plt.figure(1)
    # q1

    for i in range(4):
        fshift = fft(img_list[i])
        A = amplitude(fshift)
        f, A_f = cart2porl(A,img_list[i])
        plt.plot(np.log(f[1:190]),np.log(A_f[1:190]), color=colorlist[i], linestyle=linetype[i], label=labellist[i])
    plt.legend()
    plt.title("1/f law")
    plt.savefig("./pro2_result/f1_law.jpg", bbox_inches='tight', pad_inches=0.0)

    # q2
    plt.figure(2)
    for i in range(4):
        fshift = fft(img_list[i])
        A = amplitude(fshift)
        S_f0, f0 = get_S_f0(A,img_list[i])
        plt.plot(f0[10:],S_f0[10:], color=colorlist[i], linestyle=linetype[i], label=labellist[i])
    plt.legend()
    plt.title("S(f0)")
    plt.savefig("./pro2_result/S_f0.jpg", bbox_inches='tight', pad_inches=0.0)
if __name__ == '__main__':
    main()
