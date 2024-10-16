'''
-----------------------------------------------
This is the code for project 1 question 3
A 2D scale invariant world
'''
import numpy as np
import cv2
import pdb
import matplotlib.pyplot as plt
r_min = 1
def inverse_cdf(x):
    ''' 
    Parameters:
        1. x: the random number sampled from uniform distribution
    Return:
        1. y: the random number sampled from the cubic law power
    '''
    # y = x # Need to be changed
    # TODO: Add your code here
    y = r_min/np.sqrt(1 - x)
    return y
def GenLength(N):
    ''' 
    Function for generating the length of the line
    Parameters:
        1. N: the number of lines
    Return:
        1. random_length: N*1 array, the length of the line, sampled from sample_r
    Tips:
        1. Using inverse transform sampling. Google it!
    '''
    # sample a random number from uniform distribution
    U = np.random.random(N)
    random_length = inverse_cdf(U)
    return random_length

def DrawLine(points,rad,length,pixel,N):
    ''' 
    Function for drawing lines on a image
    Parameters:
        1. points: N*2 array, the coordinate of the start points of the lines, range from 0 to pixel
        2. rad: N*1 array, the orientation of the line, range from 0 to 2\pi
        3. length: N*1 array, the length of the line, sampled from sample_r
        4. pixel: the size of the image
        5. N: the number of lines
    Return:
        1. bg: the image with lines
    '''
    # background
    bg = 255*np.ones((pixel,pixel)).astype('uint8')

    # TODO: Add your code here
    for i in range(N):
        # first get the end points
        x_end = np.clip(int(points[i][0] + length[i] * np.cos(rad[i])), 0, pixel - 1)
        y_end = np.clip(int(points[i][1] + length[i] * np.sin(rad[i])), 0, pixel - 1)
        endpoints = (x_end, y_end)
        # pdb.set_trace()
        cv2.line(bg, (int(points[i][0]), int(points[i][1])), endpoints, 0, 1)

    cv2.imwrite('./pro3_result/'+str(pixel)+'.png', bg)
    return bg

def solve_q1(N = 5000,pixel = 1024):
    ''' 
    Code for solving question 1
    Parameters:
        1. N: the number of lines
        2. pixel: the size of the image
    '''
    # Generating length
    length = GenLength(N)

    # Generating starting points uniformly
    # points = np.array([[0,0] for i in range(N)]) # Need to be changed
    points = []
    # TODO: Add your code here
    # points = np.column_stack((np.random.uniform(0, pixel, N), np.random.uniform(0, pixel, N)))
    points_x = (np.random.uniform(0, pixel, N))
    points_y = (np.random.uniform(0, pixel, N))
    for i in range(N):
        points.append(([points_x[i], points_y[i]]))

    # Generating orientation, range from 0 to 2\pi
    # rad = np.array([0 for i in range(N)]) # Need to be changed
    # TODO: Add your code here
    rad = np.random.uniform(0, 2*np.pi, N)
    image = DrawLine(points,rad,length,pixel,N)
    return image,points,rad,length

def DownSampling(img,points,rad,length,pixel,N,rate):
    ''' 
    Function for down sampling the image
    Parameters:
        1. img: the image with lines
        2. points: N*2 array, the coordinate of the start points of the lines, range from 0 to pixel
        3. rad: N*1 array, the orientation of the line, range from 0 to 2\pi
        4. length: N*1 array, the length of the line
        5. pixel: the size of the image
        6. rate: the rate of down sampling
    Return:
        1. image: the down sampled image
    Tips:
        1. You can use Drawline for drawing lines after downsampling the components
    '''
    image = img # Need to be changed    
    # TODO: Add your code here
    down_points = []
    down_length = np.zeros_like(length)
    down_rad = np.copy(rad)
    to_delete = []
    for i in range(N):
        x_down = int(points[i][0]/rate)
        y_down = int(points[i][1]/rate)
        down_points.append((x_down, y_down))
        down_length[i] = int(length[i]/rate)
        if down_length[i] < 1:
            to_delete.append(i)
    # pdb.set_trace()
    for i in sorted(to_delete, reverse=True):
        down_points.pop(i)
        down_length = np.delete(down_length, i)
        down_rad = np.delete(down_rad, i)
    image = DrawLine(down_points,down_rad,down_length,int(pixel / rate),N - len(to_delete))
    return image

def crop(image1,image2,image3):
    ''' 
    Function for cropping the image
    Parameters:
        1. image1, image2, image3: I1, I2, I3
    '''
    
    # TODO: Add your code here
    crop_size = 128
    patches = []
    for img in [image1, image2, image3]:
        h, w = img.shape
        x1 = np.random.randint(0, w - crop_size)
        y1 = np.random.randint(0, h - crop_size)
        x2 = np.random.randint(0, w - crop_size)
        y2 = np.random.randint(0, h - crop_size)

        patch1 = img[y1:y1 + crop_size, x1:x1 + crop_size]
        patch2 = img[y2:y2 + crop_size, x2:x2 + crop_size]
        patches.append(patch1)
        patches.append(patch2)

    fig, axes = plt.subplots(1, 6, figsize=(15, 5))  
    for i, patch in enumerate(patches):
        axes[i].imshow(patch, cmap='gray')  
        axes[i].axis('off')  
        axes[i].spines['top'].set_color('black')
        axes[i].spines['bottom'].set_color('black')
        axes[i].spines['left'].set_color('black')
        axes[i].spines['right'].set_color('black')
    plt.tight_layout()
    plt.savefig('pro3_result/crop/crop.png')

    # plt.figure()
    # for i in range(2):
    #     plt.subplot(3, 2, i+1)
    #     height, width = image1.shape
    #     pos_x = np.random.randint(width - 128)
    #     pos_y = np.random.randint(height - 128)
    #     plt.imshow(image1[pos_x:pos_x+128, pos_y:pos_y+128], cmap = plt.get_cmap('gray'))
    #     plt.title('Patch of random image')
    #     plt.savefig('testing/image1.png')
    #     plt.xticks([])
    #     plt.yticks([])
    # for i in range(2):
    #     plt.subplot(3, 2, i+1)
    #     height, width = image2.shape
    #     pos_x = np.random.randint(width - 128)
    #     pos_y = np.random.randint(height - 128)
    #     plt.imshow(image2[pos_x:pos_x+128, pos_y:pos_y+128], cmap = plt.get_cmap('gray'))
    #     plt.title('Patch of random image')
    #     plt.xticks([])
    #     plt.yticks([])
    # for i in range(2):
    #     plt.subplot(3, 2, i+1)
    #     height, width = image3.shape
    #     pos_x = np.random.randint(width - 128)
    #     pos_y = np.random.randint(height - 128)
    #     plt.imshow(image3[pos_x:pos_x+128, pos_y:pos_y+128], cmap = plt.get_cmap('gray'))
    #     plt.title('Patch of random image')
    #     plt.savefig('testing/image3.png')
    #     plt.xticks([])
    #     plt.yticks([])
    return


def main():
    N = 10000
    pixel = 1024
    image_1024, points, rad, length = solve_q1(N,pixel)
    # 512 * 512
    image_512 = DownSampling(image_1024,points, rad, length, pixel, N, rate = 2)
    # 256 * 256
    image_256 = DownSampling(image_1024,points, rad, length, pixel, N, rate = 4)
    crop(image_1024,image_512,image_256)
if __name__ == '__main__':
    main()
