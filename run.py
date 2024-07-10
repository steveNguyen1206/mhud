import os
import pandas as pd
from attack.deid import Blur
from utils.preprocess import parse_xml_to_csv, get_data_labels, split_train_test


# check if csv file exists
if not os.path.exists('./car-plate-detection-labels.csv'):
    parse_xml_to_csv('./datasets/car-plate-detection/annotations/*.xml')

# Load csv file
df = pd.read_csv('./car-plate-detection-labels.csv')
# print(df.head())
data, output = get_data_labels(df)
# print(len(data))

# Split data into train and test
X, y = split_train_test(data, output)