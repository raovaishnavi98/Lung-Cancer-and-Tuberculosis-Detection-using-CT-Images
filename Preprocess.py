#!/usr/bin/env python
# coding: utf-8

# In[40]:


import pandas as pd
import numpy as np
import os
import glob
import sys

from joblib import Parallel, delayed
import SimpleITK as sitk

from PIL import Image

from sklearn.model_selection import train_test_split


# In[41]:


annotations = pd.read_csv('annotations.csv')
candidates = pd.read_csv('candidates_V2.csv')


# In[65]:


class CTScan(object):
    """
	A class that allows you to read .mhd header data, crop images and
	generate and save cropped images
    Args:
    filename: .mhd filename
    coords: a numpy array
	"""
    def __init__(self, filename = None, coords = None, path = None):
        """
        Args
        -----
        filename: .mhd filename
        coords: coordinates to crop around
        ds: data structure that contains CT header data like resolution etc
        path: path to directory with all the raw data
        """
        self.filename = filename
        self.coords = coords
        self.ds = None
        self.image = None
        self.path = path

    def reset_coords(self, coords):
        """
        updates to new coordinates
        """
        self.coords = coords

    def read_mhd_image(self):
        """
        Reads mhd data
        """
        path = glob.glob(self.path + self.filename + '.mhd')
        print(path)
        if len(path) is 0:
            return
        self.ds = sitk.ReadImage(path[0])
        self.image = sitk.GetArrayFromImage(self.ds)

    def get_voxel_coords(self):
        """
        Converts cartesian to voxel coordinates
        """
        origin = self.ds.GetOrigin()
        resolution = self.ds.GetSpacing()
        voxel_coords = [np.absolute(self.coords[j]-origin[j])/resolution[j]             for j in range(len(self.coords))]
        return tuple(voxel_coords)

    def get_image(self):
        """
        Returns axial CT slice
        """
        return self.image

    def get_subimage(self, width):
        """
        Returns cropped image of requested dimensions
        """
        self.read_mhd_image()
        x, y, z = self.get_voxel_coords()
        subImage = self.image[int(z), int(y-width/2):int(y+width/2),         int(x-width/2):int(x+width/2)]
        return subImage

    def normalizePlanes(self, npzarray):
        """
        Copied from SITK tutorial converting Houndsunits to grayscale units
        """
        maxHU = 400.
        minHU = -1000.
        npzarray = (npzarray - minHU) / (maxHU - minHU)
        npzarray[npzarray>1] = 1.
        npzarray[npzarray<0] = 0.
        return npzarray

    def save_image(self, filename, width):
        """
        Saves cropped CT image
        """
        image = self.get_subimage(width)
        image = self.normalizePlanes(image)
        Image.fromarray(image*255).convert('L').save(filename)


def create_data(idx, outDir, X_data,  width = 50):
    '''
    Generates your test, train, validation images
    outDir = a string representing destination
    width (int) specify image size
    '''
    print((X_data.loc[idx])[0])
    scan = CTScan(np.asarray(X_data.loc[idx])[0], np.asarray(X_data.loc[idx])[1:], 'subset0/')
    outfile = outDir  +  str(idx)+ '.jpg'
    path = glob.glob('subset0/' + np.asarray(X_data.loc[idx])[0] + '.mhd')
    print( path)
    if len(path) is 0:
        return
    scan.save_image(outfile, width)
def do_test_train_split(filename):
    """
    Does a test train split if not previously done
    """
    candidates = pd.read_csv(filename)

    positives = candidates[candidates['class']==1].index
    negatives = candidates[candidates['class']==0].index

    ## Under Sample Negative Indexes
    np.random.seed(42)
    negIndexes = np.random.choice(negatives, len(positives)*5, replace = False)

    candidatesDf = candidates.iloc[list(positives)+list(negIndexes)]

    X = candidatesDf.iloc[:,:-1]
    y = candidatesDf.iloc[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y,     test_size = 0.20, random_state = 42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,         test_size = 0.20, random_state = 42)

    X_train.to_pickle('traindata')
    y_train.to_pickle('trainlabels')
    X_test.to_pickle('testdata')
    y_test.to_pickle('testlabels')
    X_val.to_pickle('valdata')
    y_val.to_pickle('vallabels')


def main():
    mode='test'

    inpfile = mode + 'data'
    outDir = mode + '/image_'

    if os.path.isfile(inpfile):
        pass
    else:
        do_test_train_split('candidates_V2.csv')
    X_data = pd.read_pickle(inpfile)
    print(X_data.index[0])
   # [create_data(idx, outDir, X_data) for idx in X_data.index]
    Parallel(n_jobs = 3)(delayed(create_data)(idx, outDir, X_data) for idx in X_data.index)

if __name__ == "__main__":
    main()


# In[ ]:


create_data


# In[26]:


filename = 'subset0/1.3.6.1.4.1.14519.5.2.1.6279.6001.105756658031515062000744821260'
#filename = '/media/sf_New_folder/subset0/1.3.6.1.4.1.14519.5.2.1.6279.6001.105756658031515062000744821260.mhd'
coords = (70.19, -140.93, 877.68)#[877.68, -140.93, 70.19]
scan = CTScan(filename, coords)
scan.read_mhd_image()
x, y, z = scan.get_voxel_coords()
image = scan.get_image()
dx, dy, dz = scan.get_resolution()
x0, y0, z0 = scan.get_origin()


# In[27]:


print(image)


# In[ ]:
