#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow_datasets as tfds

import argparse
import datetime
import numpy as np
import pandas as pd

# Just for plotting data
import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

### Parse arguments
parser = argparse.ArgumentParser(description='Training model')
parser.add_argument('input_data', help="Input normalized data CSV file")
parser.add_argument( '--input-cols', dest='input_cols', type = str, help="Used input cols", default="9,10,11,12,13,5,8,7,0")
parser.add_argument( '--input-csv-sep', dest='input_separator', type = str, help="Input CSV separator", default=',')
parser.add_argument( '--out', dest='out', help="Out model filename", type = str, default="model.tflite")
parser.add_argument( '--verbose', dest='verb', help="Verbose mode", action='store_true')
args = parser.parse_args()

### Read raw data from file
print("Reading data...")
raw_data = pd.read_csv(args.input_data, sep=args.input_separator)
raw_data = raw_data.iloc[:, [int(i) for i in args.input_cols.split(',')]]

print("Done")

### Remove NaN rows
print("Remove NaN rows...")
data = raw_data.dropna().reset_index(drop = True)
print("Done")

### Convert timestamps from STRING
print("Convert timestamps...")
_date_time = data.pop('datetime')
_date_time = _date_time.map(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S") if x != "" or x != np.nan or x != None else None)
data['datetime'] = _date_time
timestamp_s = _date_time.map(pd.Timestamp.timestamp)
#data.insert(0, 'timestamp', timestamp_s)
print("Done")

### Get days
print("Calculate days indices...")
ROW_COUNT = data.shape[0]
first_index = None
num_of_days = 0
days = {}
for index, row in data.iterrows():
    if first_index == None:
        first_index = index
        first_date = row['datetime']

    # Get current date
    date = row['datetime']
    is_same_day = date.date() == first_date.date()
    is_last_etap = index == data.index[-1]
    if is_last_etap:
        is_same_day = False
        index += 1

    if not is_same_day:
        print("Progress: {:.2f}%".format(round(100 * float(index) / float(ROW_COUNT), 2)))
        num_of_days += 1
        days[num_of_days] = [first_index, index - 1]
        first_index = index
        first_date = row['datetime']

NUMBER_OF_DAYS = len(days)
print("Done (Found '{}' day(s))".format(NUMBER_OF_DAYS))
_date_time = data.pop('datetime')
print(data)

### Split data
train_data = data[0:days[int(0.7 * NUMBER_OF_DAYS)][1] + 1]
val_data = data[days[int(0.7 * NUMBER_OF_DAYS) + 1][0]:days[int(0.9 * NUMBER_OF_DAYS)][1] + 1]
test_data = data[days[int(0.9 * NUMBER_OF_DAYS) + 1][0]:]
num_features = data.shape[1]
print(num_features)

class WindowGenerator():
    def __init__(self, input_width=100, label_width=1, shift=1,
                 train_df=train_data, val_df=val_data, test_df=test_data,
                 label_columns=None):
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}
        self.column_indices = {name: i for i, name in enumerate(train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        self.total_window_size = input_width

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack([labels[:, :, self.column_indices[name]] for name in self.label_columns], axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels


w2 = WindowGenerator(input_width=100, label_width=1, shift=1,
                     label_columns=['open_norm'])
w2
# Stack three slices, the length of the total window.
example_window = tf.stack(np.asarray([train_data[:w2.total_window_size * 2]]).astype(np.float32))

example_inputs, example_labels = w2.split_window(example_window)

print('All shapes are: (batch, time, features)')
print(f'Window shape: {example_window.shape}')
print(f'Inputs shape: {example_inputs.shape}')
print(f'Labels shape: {example_labels.shape}')

exit()
