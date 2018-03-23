import numpy as np
import skimage.io as sio
from scipy.io import loadmat
import math
from sklearn import preprocessing
from scipy.misc import imrotate

 
def compute_gradient(im):
    H = len(im)
    W = len(im[0])
    angles = np.zeros((H - 2, W - 2))
    magnitudes = np.zeros((H - 2, W - 2))
    for i in range(0, H - 2):
        for j in range(0, W - 2):
            top_elem = im[i][j + 1]
            bottom_elem = im[i + 2][j + 1]
            left_elem = im[i + 1][j]
            right_elem = im[i + 1][j + 2]
            angle_radians = np.arctan2(top_elem - bottom_elem, left_elem - right_elem)
            angle_deg = np.rad2deg(angle_radians)
            if angle_deg < 0:
                angle_deg += 180
            magnitude = np.sqrt((left_elem - right_elem)**2 + (top_elem - bottom_elem)**2)
            angles[i][j] = angle_deg
            magnitudes[i][j] = magnitude
    return angles, magnitudes

def generate_histogram(angles, magnitudes, nbins = 9):
    histogram = np.zeros(nbins)
    M, N = angles.shape
    bin_range = 180 / nbins
    for i in range(M):
        for j in range(N):
            found_0 = False
            found_1 = False
            angle = angles[i][j]
            magnitude = magnitudes[i][j]
            bin1 = int(angle / bin_range)

            if angle % bin_range == 0:
                bin1 -= 1 #0
            elif angle % bin_range < bin_range / 2:
                bin1 -= 1
            bin2 = bin1  + 1 #1

            if bin1 == -1:
                found_0 = True
                bin2 = 0
                bin1 = nbins - 1 
            if bin2 == nbins:
                found_1 = True
                bin2 = 0
                bin1 = nbins - 1 
                
            center_angle1 = bin1 * bin_range + bin_range / 2 #10
            center_angle2 = bin2 * bin_range + bin_range / 2 #30
            diff1 = np.abs(angle - center_angle2) #10
            diff2 = np.abs(angle - center_angle1) #10

            if found_0 is True:
                diff2 = 180 - diff2
            elif found_1 is True:
                diff1 = 180 - diff1

            histogram[bin1] += magnitude * (diff1 * 1.0 / bin_range)
            histogram[bin2] += magnitude * (diff2 * 1.0 / bin_range)
    return histogram


def compute_hog_features(im, pixels_in_cell, cells_in_block, nbins):
    angles, magnitudes = compute_gradient(im)
    window_size = cells_in_block * pixels_in_cell
    stride = window_size / 2
    h = len(angles)
    w = len(angles[0])
    h -= window_size
    w -= window_size
    hog = np.zeros((h/stride, w/stride, cells_in_block * cells_in_block * nbins))

    for i in range(h/stride):
        for j in range(w/stride):
            block = np.zeros((cells_in_block, cells_in_block, nbins))
            h_min = i * stride
            h_max = h_min + window_size
            w_min = j * stride
            w_max = w_min + window_size
            b_angles = angles[h_min : h_max, w_min : w_max]
            b_magnitudes = magnitudes[h_min : h_max, w_min : w_max]

            for p in range(cells_in_block):
                for q in range(cells_in_block):
                    h_min_cell = p * pixels_in_cell
                    h_max_cell = h_min_cell + pixels_in_cell
                    w_min_cell = q * pixels_in_cell
                    w_max_cell = w_min_cell + pixels_in_cell
                    c_angles = b_angles[h_min_cell : h_max_cell, w_min_cell : w_max_cell]
                    c_magnitudes = b_magnitudes[h_min_cell : h_max_cell, w_min_cell : w_max_cell]
                    c_histogram = generate_histogram(c_angles, c_magnitudes, nbins)
                    block[p, q, :] = c_histogram
            block = np.reshape(block, (cells_in_block * cells_in_block * nbins, ))
            block = block / np.sqrt(np.linalg.norm(block) ** 2 + .01)
            hog[i, j, :] = block
    return hog
