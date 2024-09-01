#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
parser = argparse.ArgumentParser(description='Preparing and svaing data')
parser.add_argument('input_data', help="Input CSV file")
parser.add_argument( '--input-cols', dest='input_cols', type = str, help="Used input cols", default="0,1,2,3,4,5,7,9,12")
parser.add_argument( '--input-csv-sep', dest='input_separator', type = str, help="Input CSV separator", default=';')
parser.add_argument( '--out', dest='out', help="Output filename", type = str, default="normalized.csv")
parser.add_argument( '--min-window', dest='minwin', help="Minimum size of window", type = int, default=100)
parser.add_argument( '--verbose', dest='verb', help="Verbose mode", action='store_true')
parser.add_argument( '--plot-day', dest='plot_day', help="Plot OPEN values the specific day", type = int, default=0)
args = parser.parse_args()

### Read raw data from file
print("Reading data...")
raw_data = pd.read_csv(args.input_data, sep=args.input_separator)
raw_data = raw_data.iloc[:, [int(i) for i in args.input_cols.split(',')]]
print("Done")

### Convert timestamps from STRING
def date_converting(data, col=0):
    data[data.columns[0]] = data[data.columns[0]].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
    return data

print("Convert timestamps...")
data = date_converting(raw_data)
print("Done")

### Initialize
ROW_COUNT = len(data)
num_of_days = 0
SHOW_DAY = args.plot_day
VERBOSE = args.verb
if VERBOSE:
    print("Number of rows: ", len(data))
    print("Input columns", data.columns)

### Extend dataframe with normalized columns
data["open_norm"] = None
data["high_norm"] = None
data["low_norm"] = None
data["close_norm"] = None
data["market_open_norm"] = None

### Normalize numeric data
first_index = None
days = {}
skipped_rows = []
last_day_mean = None
last_day_std = None
number_of_not_full_day = 0
skipped_days = 0
print("Normalizing...")
for index, row in data.iterrows():
    if first_index == None:
        first_index = index
        first_date = row[data.columns[0]]

    # Get current date
    date = row[data.columns[0]]
    is_same_day = date.date() == first_date.date()
    #is_next_day = date.date() == first_date.date() + datetime.timedelta(days=1)
    #is_in_the_time = date.time() <= datetime.time(22, 0, 0)
    is_last_etap = index == data.index[-1]
    if is_last_etap:
        is_same_day = False
        index += 1

    if not is_same_day:
        num_of_days += 1

        actual_open = data.loc[first_index:index - 1, "open"]
        one_day_mean = actual_open.mean()
        one_day_std = actual_open.std()

        if len(actual_open) < args.minwin:
            if last_day_mean == None:
                print("[WARNING] Skipped {} data rows".format(num_of_days))
                skipped_rows.append([num_of_days, first_index, index - 1])
                first_index = index
                first_date = row[data.columns[0]]
                num_of_days -= 1
                skipped_days += 1
                continue
            else:
                print("[WARNING] Used last day MEANS and STD")
                one_day_mean = last_day_mean
                one_day_std = last_day_std
                number_of_not_full_day += 1

        if SHOW_DAY > 0 and SHOW_DAY != num_of_days:
            last_day_mean = one_day_mean
            last_day_std = one_day_std
            first_index = index
            first_date = row[data.columns[0]]
            continue

        log_txt = "Status: {:.2f}%   Current day: {}".format(round(100 * float(index) / float(ROW_COUNT), 2), num_of_days)
        if VERBOSE:
            log_txt = log_txt + "   index: {}   len, mean, std: {}, {:.2f}, {:.4f}".format(index, len(actual_open), one_day_mean, one_day_std)
        print(log_txt)

        days[num_of_days] = {"indices" : [first_index, index - 1], "mean" : one_day_mean, "std" : one_day_std}
        for n in ["open", "high", "low", "close", "market_open"]:
            data.loc[first_index:index - 1, n + "_norm"] = data.loc[first_index:index - 1, n].apply(lambda x: (x - one_day_mean) / one_day_std)

        if SHOW_DAY == num_of_days:
            title = str(data.loc[first_index]["datetime"]) + " - " + str(data.loc[index -1]["datetime"])
            axs = data.loc[first_index:index - 1, ["open", "open_norm"]].plot.line(stacked=False, layout=(1,2), subplots=True, title=title)
            plt.show()
            exit()

        last_day_mean = one_day_mean
        last_day_std = one_day_std
        first_index = index
        first_date = row[data.columns[0]]

print("Done")

### Print summarized information
num_of_skipped_rows = sum([e[2] - e[1] + 1 for e in skipped_rows])
print("")
print("Normalized rows:", ROW_COUNT - num_of_skipped_rows)
print("Skipped rows (days):", num_of_skipped_rows, "({} day(s))".format(skipped_days))
print("Number of valid days:", num_of_days)
print(" which are not fully:", number_of_not_full_day)
print("")

if VERBOSE:
    print("List of skipped indices (day, fisrt index, last index):", skipped_rows)
    print("")

### Saving normalized data
print("Saving normalized data...")
data.to_csv(args.out, index=False)
print("Done")
