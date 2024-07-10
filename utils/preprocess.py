import os
import cv2
import numpy as np
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import xml.etree.ElementTree as xet
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def parse_xml_to_csv(path_str):
    """
    Parse xml files to csv
    :params:
    """
    # /kaggle/input/car-plate-detection/input/car-plate-detection
    path = glob(path_str)
    labels_dict = dict(filepath=[], img = [],xmin=[],xmax=[],ymin=[],ymax=[])
    for filename in path:

        info = xet.parse(filename)
        root = info.getroot()
        member_img_name = root.find('filename').text
        member_object = root.find('object')
        labels_info = member_object.find('bndbox')
        xmin = int(labels_info.find('xmin').text)
        xmax = int(labels_info.find('xmax').text)
        ymin = int(labels_info.find('ymin').text)
        ymax = int(labels_info.find('ymax').text)

        labels_dict['filepath'].append(filename)
        labels_dict['xmin'].append(xmin)
        labels_dict['xmax'].append(xmax)
        labels_dict['ymin'].append(ymin)
        labels_dict['ymax'].append(ymax)
        labels_dict['img'].append(member_img_name)

    df = pd.DataFrame(labels_dict)
    df = df.sort_values(by=['filepath'])
    df = df.reset_index(drop=True)
    df.to_csv('car-plate-detection-labels.csv',index=False)
    # df.head()

def getFilename(filename):
    filename_image = xet.parse(filename).getroot().find('filename').text
    filepath_image = os.path.join('./datasets/car-plate-detection/images',filename_image)
    return filepath_image


def get_data_labels(df):

    image_path = list(df['filepath'].apply(getFilename))
    #Targeting all our values in array selecting all columns
    labels = df.iloc[:,1:].values
    data = []
    output = []
    for ind in range(len(image_path)):
        image = image_path[ind]
        img_arr = cv2.imread(image)
        h,w,d = img_arr.shape
        # Prepprocesing
        load_image = load_img(image,target_size=(224,224))
        load_image_arr = img_to_array(load_image)
        norm_load_image_arr = load_image_arr/255.0 # Normalization
        # Normalization to labels
        img,xmin,xmax,ymin,ymax = labels[ind]
        nxmin,nxmax = xmin/w,xmax/w
        nymin,nymax = ymin/h,ymax/h
        label_norm = (nxmin,nxmax,nymin,nymax) # Normalized output
        # Append
        data.append(norm_load_image_arr)
        output.append(label_norm)    
    return data, output

def split_train_test(data, output, train_size=0.8, random_state=0):
    # Convert data to array
    X = np.array(data,dtype=np.float32)
    y = np.array(output,dtype=np.float32)

    # Split the data into training and testing set using sklearn.
    x_train,x_test,y_train,y_test = train_test_split(X,y,train_size=0.8,random_state=0)
    x_train.shape,x_test.shape,y_train.shape,y_test.shape
    return X, y