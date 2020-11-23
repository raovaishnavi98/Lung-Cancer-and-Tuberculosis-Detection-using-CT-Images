# Lung-Cancer-and-Tuberculosis-Detection-using-CT-Images

## DataSet obtained from: https://luna16.grand-challenge.org/Data/

## Preprocess.py
Performs various image preprocessing steps such as Greyscale, and obtains the Voxel Coordinates to convert the images to 2D in the required dimensions (250x250)

The .mhd image  and its corresponding voxel coordinates are read.

The image is cropped in and around the voxel coordinates to produce a sub-image with 2 dimensional mappings of the data.

The cropped 2D  image is in hounds unit and is converted to grayscale image.

The output image is  saved for data_processing and augmentation.


## Data_process_and_augment.py
Oversampling is a technique used to re-sample imbalanced class distributions such that the model is not overly biased towards labeling instances as the majority class type.

Due to the biased nature of the dataset, training model has 30% positive and 70% negative CT images with Lung Cancer / Tuberculosis. To make the prediction system unbiased, trained data images having positive result undergo augmentation by rotating the images by 90, 180 and 270 degrees.


## Model.py
The CNN model selected for implementation:
![CNN Model](https://github.com/raovaishnavi98/Lung-Cancer-and-Tuberculosis-Detection-using-CT-Images/blob/main/model.png?raw=true)



## run_model.py
Main file to run the application. It opens up a Django Framework based web interface that allows uploading of CT images and provides output as positive or negative.

