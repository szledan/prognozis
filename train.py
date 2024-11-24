#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import datetime
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_datasets as tfds
#import tensorflowjs as tfjs

# Just for plotting data
import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

### Parse arguments
parser = argparse.ArgumentParser(description='Training model')
parser.add_argument('input_data', help="Input normalized data CSV file")
parser.add_argument( '--input-cols', dest='input_cols', type = str, help="Used input cols", default="0,1,2,3,4,5,6,7,9,8")
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
timestamp_s = _date_time.map(pd.Timestamp.timestamp)
print("Done")

### Get days
print("Calculate days indices...")
ROW_COUNT = data.shape[0]
first_index = None
num_of_days = 0
days = {}
last_day = 1
for index, row in data.iterrows():
  if row['day_count'] != last_day:
    days[last_day] = [first_index, index - 1]
    last_day = data['day_count'][index]
    first_index = index

NUMBER_OF_DAYS = len(days)
print("Done (Found '{}' day(s))".format(NUMBER_OF_DAYS))
### Split data
train_data = data[0:days[int(0.7 * NUMBER_OF_DAYS)][1] + 1]
val_data = data[days[int(0.7 * NUMBER_OF_DAYS) + 1][0]:days[int(0.9 * NUMBER_OF_DAYS)][1] + 1]
test_data = data[days[int(0.9 * NUMBER_OF_DAYS) + 1][0]:]
num_features = data.shape[1]

class WindowGenerator():
  def __init__(self, input_width=100, label_width=1, shift=1, train_df=train_data, val_df=val_data, test_df=test_data, label_columns=None):
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
    self.total_window_size = input_width + shift

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
    inputs = features[:, self.input_slice, :-1]
    labels = features[:, self.labels_slice, :]
    if self.label_columns is not None:
      labels = tf.stack([labels[:, :, self.column_indices[name]] for name in self.label_columns], axis=-1)

    # Slicing doesn't preserve static shape information, so set the shapes
    # manually. This way the `tf.data.Datasets` are easier to inspect.
    inputs.set_shape([None, self.input_width, None])
    labels.set_shape([None, self.label_width, None])

    return inputs, labels

  def make_dataset(self, data):
    data = np.array(data, dtype=np.float32)
    ds = tf.keras.utils.timeseries_dataset_from_array(
        data=data,
        targets=None,
        sequence_length=self.total_window_size,
        sequence_stride=1,
        shuffle=True,
        batch_size=64,)

    ds = ds.map(self.split_window)
    return ds

  @property
  def train(self):
    return self.make_dataset(self.train_df)

  @property
  def val(self):
    return self.make_dataset(self.val_df)

  @property
  def test(self):
    return self.make_dataset(self.test_df)

  @property
  def example(self):
    """Get and cache an example batch of `inputs, labels` for plotting."""
    result = getattr(self, '_example', None)
    if result is None:
      # No example batch was found, so get one from the `.train` dataset
      result = next(iter(self.train))
      # And cache it for next time
      self._example = result
    return result


MAX_EPOCHS = 20

def compile_and_fit(model, window, patience=2):
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min')

  model.compile(loss=tf.keras.losses.MeanSquaredError(),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=[tf.keras.metrics.MeanAbsoluteError()])

  history = model.fit(window.train, epochs=MAX_EPOCHS,
                      validation_data=window.val,
                      callbacks=[early_stopping])
  return history

conv_window = WindowGenerator(input_width=100, label_width=1, shift=1, label_columns=['target'])

multi_step_dense = tf.keras.Sequential([
    # Shape: (time, features) => (time*features)
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=1),
    # Add back the time dimension.
    # Shape: (outputs) => (1, outputs)
    tf.keras.layers.Reshape([1, -1]),
])

#multi_step_dense = tf.keras.Sequential([
#    # Shape: (time, features) => (time*features)
#    tf.keras.layers.Dense(units=64, activation='relu'),
#    tf.keras.layers.Conv1D(filters=32,
#                               kernel_size=(9,),
#                               activation='relu'),
#    tf.keras.layers.Dense(units=32, activation='relu'),
#    tf.keras.layers.Dense(units=1),
#    # Add back the time dimension.
#    # Shape: (outputs) => (1, outputs)
#    tf.keras.layers.Reshape([1, -1]),
#])

history = compile_and_fit(multi_step_dense, conv_window)


#multi_step_dense.predict([2.1857384, 2.2244708, 2.174672, 2.2134044, -0.7800609, 307, 0])

if True:
#  tfjs.converters.save_keras_model(multi_step_dense, ".")
  pass
else:
  #multi_step_dense.save("model.h5")
  pass

exit()
