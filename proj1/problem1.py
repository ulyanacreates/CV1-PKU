'''
This is the code for project 1 question 1
Question 1: High kurtosis and scale invariance
'''
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from PIL import Image
from scipy.stats import gennorm, fit, norm
from scipy.optimize import curve_fit
import scipy.special
from scipy.special import gamma
from tqdm import tqdm
from math import sqrt
import pdb
from scipy import ndimage

data_repo = "./image_set"
set_repo = ['setA','setB','setC']
img_name_list = []
def read_img_list(set):
    '''
    Read images from the corresponding image set
    '''
    global img_name_list
    img_list = os.listdir(os.path.join(data_repo,set))
    img_list.sort()
    img_name_list.append(img_list)
    img_list = [Image.open(os.path.join(data_repo,set,img)) for img in img_list]
    return img_list

# (a) First convert an image to grey level and re-scale the intensity to [0,31]
def convert_grey(img):
    ''';
    Convert an image to grey
    Parameters:
        1. img: original image
    Return:
        1. img_grey: grey image

    '''
    # img_grey = img # Need to be changed

    # TODO: Add your code here
    pdb.set_trace()
    img_grey = img.convert('L')
    # image_array = np.array(img_grey)
    # unique_intensity_levels = np.unique(image_array)
    # print("Unique intensity levels:", unique_intensity_levels)
    img_grey.save('testing/greyscale.jpg')
    return img_grey

def rescale(img_grey):
    '''
    Rescale the intensity to [0,31]
    Parameters:
        1. img_grey: grey image
    Return:
        1. scale_img_grey: scaled grey image

    '''
    # scale_img_grey = img_grey # Need to be changed
    # TODO: Add your code here
    scale_img_grey = Image.fromarray(np.floor(np.array(img_grey)/8))
    return scale_img_grey


# (b) Convolve the images with a horizontal gradient filter ∇xI
def gradient_filter(img):
    '''
    This function is used to calculate horizontal gradient
    Parameters:
        1. img: img for calculating horizontal gradient 
    Return:
        1. img_dx: an array of horizontal gradient

    >>> img = np.array([[1,2,3],[4,5,6],[7,8,9]])
    >>> gradient_filter(img)
    array([[1, 1],
           [1, 1],
           [1, 1]])
    '''
    # img_dx = np.array([2, 1, 3, 1, 1, 1])    # Need to be changed
    # TODO: Add your code here
    img = np.array(img)
    
    # img_dx = cv2.Sobel(img, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3) ## use filter [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    img_dx = np.zeros_like(img)
    img_dx[:, :-1] = img[:, 1:] - img[:, :-1]
    grad = Image.fromarray(img_dx.astype(np.uint8))
    grad.save('testing/horiz_grad.jpg')
    return img_dx


def plot_Hz(img_dx,log = False, init = np.zeros_like((61, )), down=False):
    '''
    This function is used to plot the histogram of horizontal gradient
    '''
    # clear previous plot
    hz, bins_edge = np.histogram(img_dx, bins=list(range(-31, 31)))
    hz = hz/np.sum(hz)
    epsilon = 1e-5
    if log:
        plt.plot(bins_edge[:-1], np.log(hz+epsilon), 'b-',label="log Histogram")
        if down: 
            hz_init, bins_edge_init = np.histogram(init, bins=list(range(-31, 31)))
            hz_init = hz_init/np.sum(hz_init)
            plt.plot(bins_edge_init[:-1], np.log(hz_init + epsilon), 'y-',label="Initial Histogram")
    else:
        plt.plot(bins_edge[:-1], hz, 'b-',label="Histogram")
        if down: 
            hz_init, bins_edge_init = np.histogram(init, bins=list(range(-31, 31)))
            hz_init = hz_init/np.sum(hz_init)
            plt.plot(bins_edge_init[:-1], hz_init, 'y-',label="Initial Histogram")
    return hz, bins_edge

def compute_mean_variance_kurtosis(img_dx):
    '''
    Compute the mean, variance and kurtosis 
    Parameters:
        1. img_dx: an array of horizontal gradient
    Return:
        1. mean: mean of the horizontal gradient
        2. variance: variance of the horizontal gradient
        3. kurtosis: kurtosis of the horizontal gradient

    '''
    mean = 0
    variance = 0
    kurtosis = 0
    # TODO: Add your code here
    mean = np.mean(img_dx)
    variance = np.var(img_dx) # or ndimage.variance(img_dx), or np.sum((img_dx - mean)**2)/len(img_dx) all give the same results
    kurtosis = scipy.stats.kurtosis(img_dx)
    return mean, variance, kurtosis


def GGD(x, sigma, gammar):
    ''' 
    pdf of GGD
    Parameters:
        1. x: input
        2. sigma: σ
        3. gammar: γ
    Note: The notation of x,σ,γ is the same as the document
    Return:
        1. y: pdf of GGD

    '''
    y = 0
    # TODO: Add your code here
    y = (gammar/(2*sigma*scipy.special.gamma(1/gammar)))*(np.exp(-((np.abs(x/sigma))**gammar)))
    return y


def fit_GGD(hz, bins_edge):
    '''
    Fit the histogram to a Generalized Gaussian Distribution (GGD), and report the fittest sigma and gamma
    Parameters:
        1. hz: histogram of the horizontal gradient
        2. bins_edge: bins_edge of the histogram
    Return:
        None
    '''
    # fit the histogram to a generalized gaussian distribution

    # datax = bins_edge[1:]
    datay = hz


    # TODO: Add your code here
    datax = np.array([0.5 * (bins_edge[i] + bins_edge[i+1]) for i in range(len(bins_edge)-1)])
    popt, pcov = curve_fit(GGD, xdata=datax, ydata=datay)
    xspace = np.linspace(-31, 31, 186)
    fitted_curve = GGD(xspace, *popt)
    # pdb.set_trace()
    plt.bar(bins_edge[:-1], datay, width=np.diff(bins_edge), color='gray', alpha=0.6, label='Histogram')
    plt.plot(xspace, fitted_curve, color='red', linewidth=1.5, label=f'GGD Fit')

    print(f'sigma={popt[0]:.2f}, gamma={popt[1]:.2f}')
    plt.legend()
    plt.xlabel('Horizontal Gradient')
    plt.ylabel('Frequency')
    return


def plot_Gaussian(mean,variance, log=False):
    ''' 
    Plot the Gaussian distribution using the mean and the variance
    Parameters:
        1. mean: mean of the horizontal gradient
        2. variance: variance of the horizontal gradient
    Return:
        None

    '''
    x = np.linspace(-31,31,500)

    y = np.zeros(x.shape) # Need to be changed

    # TODO: Add your code here

    # y: value of pdf of Gassian distribution corresponding to x
    epsilon = 1e-5
    sigma = np.sqrt(variance)
    y = scipy.stats.norm.pdf(x, mean, sigma)
    if log == True:
        plt.plot(x, np.log10(y),'g-', label="Gaussian")
    else:
        plt.plot(x, y,'g-', label="Gaussian")
    # plt.plot(x, np.log(y + epsilon), color='black', label="Gaussian log")
    return 


def downsample(image):
    ''' 
    Downsample our images
    Parameters:
        1. image: original image
    Return:
        1. processed_image: downsampled image
    '''
    processed_image = image # Need to be changed
    # TODO: Add your code here
    processed_image = image.resize(size=(int(image.size[0]/2), int(image.size[1]/2)))
    return processed_image


def main():
    '''
    This is the main function
    '''
    # read img to img list
    # Notice: img_list is a list of image
    test_subj = convert_grey(Image.open('image_set/imposed_sce_small.bmp'))
    gradient_filter(test_subj)
    pdb.set_trace()
    img_list = [read_img_list(set) for set in set_repo]
    # set_repo refers to the three sets we'll handle
    for idx1,set in enumerate(set_repo):
        img_dx_list = []
        img_dx_2_list = []
        img_dx_4_list = []
        for idx2,img in enumerate(img_list[idx1]):
            if set == 'setC':
                img_grey = convert_grey(img)
                img_2_grey = downsample(img_grey)
                img_4_grey = downsample(img_2_grey)
            else:
                img_grey = convert_grey(img)
                img_2_grey = downsample(img_grey)
                img_4_grey = downsample(img_2_grey)

                img_grey = rescale(img_grey)
                img_2_grey = rescale(img_2_grey)
                img_4_grey = rescale(img_4_grey)

            img_dx_list.append(gradient_filter(img_grey).flatten())
            img_dx_2_list.append(gradient_filter(img_2_grey).flatten())
            img_dx_4_list.append(gradient_filter(img_4_grey).flatten())
        img_dx = np.concatenate(img_dx_list)
        img_dx_2 = np.concatenate(img_dx_2_list)
        img_dx_4 = np.concatenate(img_dx_4_list)
        
        

        # plot histogram and log histogram
        print('--'*20)

        plt.clf()
        hz, bins_edge = plot_Hz(img_dx)
        # compute mean, variance and kurtosis
        mean, variance, kurtosis = compute_mean_variance_kurtosis(img_dx)
        print(f"set: {set}")
        print(f"mean: {mean}, variance: {variance}, kurtosis: {kurtosis}")

        # fit the histogram to a generalized gaussian distribution
        fit_GGD(hz, bins_edge)

        # plot the Gaussian distribution using the mean and the variance
        plot_Gaussian(mean,variance)

        plt.savefig(f"./pro1_result/histogram/{set}.png")

        # plot log histogram
        plt.clf()
        hz, bins_edge = plot_Hz(img_dx,log=True)
        # for the super-imposition with log
        plot_Gaussian(mean, variance, log=True)
        # save the histograms
        plt.savefig(f"./pro1_result/log_histogram/gauss-log{set}.png")

        # plot the downsampled images histogram
        plt.clf()
        plot_Hz(img_dx)
        plt.savefig(f"./pro1_result/downsampled_histogram/original_{set}.png")

        plt.clf()
        plot_Hz(img_dx_2)
        plt.savefig(f"./pro1_result/downsampled_histogram/2_{set}.png")

        plt.clf()
        plot_Hz(img_dx_4)
        plt.savefig(f"./pro1_result/downsampled_histogram/4_{set}.png")

        #impose the downsampled images with the initial plot and log plot
        plt.clf()
        plot_Hz(img_dx_2, log=False, init=img_dx, down=True)
        plt.savefig(f"./pro1_result/downsampled_histogram/2_impose{set}.png")
        plt.clf()
        plot_Hz(img_dx_2, log=True, init=img_dx, down=True)
        plt.savefig(f"./pro1_result/downsampled_histogram/2_imposelog{set}.png")

        plt.clf()
        plot_Hz(img_dx_4, log=False, init=img_dx, down=True)
        plt.savefig(f"./pro1_result/downsampled_histogram/4_impose{set}.png")
        plt.clf()
        plot_Hz(img_dx_4, log=True, init=img_dx, down=True)
        plt.savefig(f"./pro1_result/downsampled_histogram/4_imposelog{set}.png")

if __name__ == '__main__':
    main()
