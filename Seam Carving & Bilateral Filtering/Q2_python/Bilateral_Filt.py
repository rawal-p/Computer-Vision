import numpy as np
import cv2
import sys
import math


def distance(x, y, i, j):
    return np.sqrt((x-i)**2 + (y-j)**2)


def gaussian_I(x, sigma):
    B = (1.0 / (2 * math.pi * (sigma ** 2))) * math.exp(- (x[0] ** 2) / (2 * sigma ** 2))
    R = (1.0 / (2 * math.pi * (sigma ** 2))) * math.exp(- (x[1] ** 2) / (2 * sigma ** 2))
    G = (1.0 / (2 * math.pi * (sigma ** 2))) * math.exp(- (x[2] ** 2) / (2 * sigma ** 2))
    return [B,R,G]

def gaussian_S(x, sigma):
    return (1.0 / (2 * math.pi * (sigma ** 2))) * math.exp(- (x ** 2) / (2 * sigma ** 2))

def apply_bilateral_filter(source, filtered_image, x, y, diameter, sigma_i, sigma_s):
    hl = (diameter-1)/2
    i_filtered = 0
    Wp = 0
    i = 0
    while i < diameter:
        j = 0
        while j < diameter:
            neighbour_x = int(x - (hl - i))
            neighbour_y = int(y - (hl - j))
            if neighbour_x >= len(source):
                neighbour_x -= len(source)
            if neighbour_y >= len(source[0]):
                neighbour_y -= len(source[0])
            gi = gaussian_I(source[neighbour_x][neighbour_y] - source[x][y], sigma_i)
            gs = gaussian_S(distance(neighbour_x, neighbour_y, x, y), sigma_s)
            w = np.dot(gi,gs)
            i_filtered += source[neighbour_x][neighbour_y] * w
            Wp += w
            j += 1
        i += 1
    i_filtered = i_filtered / Wp
    filtered_image[x][y] = np.ceil(i_filtered)


def bilateral_filter_own(source, sigma_s, sigma_i):
    width = int(2*np.ceil(2*sigma_s))+1
    print(width)
    #width = 5
    filtered_image = np.zeros(source.shape)

    i = 0
    while i < len(source):
        j = 0
        while j < len(source[0]):
            apply_bilateral_filter(source, filtered_image, i, j, width, sigma_s, sigma_i)
            j += 1
        i += 1
    return filtered_image



src = cv2.imread('Yosemite.png', 1)

sigmaS = [2.5, 2.5, 5, 5]
sigmaI = [5, 10, 5, 10]

for i in range(4):
    print(f'run{i}')
    filtered_image_OpenCV = cv2.bilateralFilter(src, -1, sigmaI[i], sigmaS[i])
    cv2.imwrite((f"OpenCV_{i}_sigmaS_{sigmaS[i]}_sigmaI_{sigmaI[i]}.jpg"), filtered_image_OpenCV)
    filtered_image = bilateral_filter_own(src, sigmaS[i], sigmaI[i])
    cv2.imwrite((f"bilateraltask{i}_sigmaS_{sigmaS[i]}_sigmaI_{sigmaI[i]}.jpg"), filtered_image)

