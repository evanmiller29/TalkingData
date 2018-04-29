import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import gc
import stacknet_funcs as funcs
from math import ceil
import numpy as np
from os import chdir

folder = "F:/Nerdy Stuff/Kaggle/Talking data/data/"
sparse_array_path = 'F:/Nerdy Stuff/Kaggle/Talking data/sparse matricies/'

predictors = []

run = "train"

dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint8',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        'click_id'      : 'uint32',
        }

if run == "train":

    file = folder + "train.csv"
    cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed']

    print('loading %s data...' % (run))
    base_df = pd.read_csv(file, parse_dates=['click_time'], low_memory=True,dtype=dtypes, usecols=cols)

else:

    print('loading %s data...' % (run))

    file = folder + "test.csv"
    cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time']

    base_df = pd.read_csv(file, parse_dates=['click_time'], dtype=dtypes, usecols=cols)

rows = base_df.shape[0]
iters = 100
iter_rows = ceil(rows/iters)

X_ttl = np.empty((0, 31))
y_ttl = np.empty((0, ))

start_point = 0

for i in list(range(start_point, iters)):

    print("Cut # %i" % (i))

    if i == 0:

        start = i * iter_rows
        end = (i + 1) * iter_rows

        print("start row = %s and end row = %s" % (start, end))
        df = base_df.iloc[start:end, :].copy()

    else:

        start = i * iter_rows + 1
        end = (i + 1) * iter_rows

        print("start row = %s and end row = %s" % (start, end))
        df = base_df.iloc[start:end, :].copy()

    df['hour'] = pd.to_datetime(df.click_time).dt.hour.astype('int8')
    df['day'] = pd.to_datetime(df.click_time).dt.day.astype('int8')

    df['minute'] = pd.to_datetime(df.click_time).dt.minute.astype('int8')
    predictors.append('minute')
    df['second'] = pd.to_datetime(df.click_time).dt.second.astype('int8')
    predictors.append('second')

    gc.collect()
    predictors, df = funcs.do_next_prev_Click(predictors, df, agg_suffix='nextClick', agg_type='float32')
    predictors, df = funcs.do_next_prev_Click(predictors, df,agg_suffix='prevClick', agg_type='float32')

    print("Calculating unique counts")

    predictors, df = funcs.do_countuniq(predictors, df, ['ip'], 'channel')
    gc.collect()

    predictors, df = funcs.do_countuniq(predictors, df, ['ip', 'device', 'os'], 'app')
    gc.collect()

    predictors, df = funcs.do_countuniq(predictors, df, ['ip', 'day'], 'hour')
    gc.collect()

    predictors, df = funcs.do_countuniq(predictors, df, ['ip'], 'app')
    gc.collect()

    predictors, df = funcs.do_countuniq(predictors, df, ['ip', 'app'], 'os')
    gc.collect()

    predictors, df = funcs.do_countuniq(predictors, df, ['ip'], 'device')
    gc.collect()

    predictors, df = funcs.do_countuniq(predictors, df, ['app'], 'channel')
    gc.collect()

    print("Calculating cumulative counts")

    predictors, df = funcs.do_cumcount(predictors, df, ['ip'], 'os')
    gc.collect()
    predictors, df = funcs.do_cumcount(predictors, df, ['ip', 'device', 'os'], 'app')
    gc.collect()

    print("Calculating counts")

    predictors, df = funcs.do_count(predictors, df, ['ip', 'day', 'hour'])
    gc.collect()
    predictors, df = funcs.do_count(predictors, df, ['ip', 'app'])
    gc.collect()
    predictors, df = funcs.do_count(predictors, df, ['ip', 'app', 'os'])
    gc.collect()

    print("Calculating variances")

    predictors, df = funcs.do_var(predictors, df, ['ip', 'day', 'channel'], 'hour');
    gc.collect()
    predictors, df = funcs.do_var(predictors, df, ['ip', 'app', 'os'], 'hour');
    gc.collect()
    predictors, df = funcs.do_var(predictors, df, ['ip', 'app', 'channel'], 'day');
    gc.collect()

    print("Calculating averages")

    predictors, df = funcs.do_mean(predictors, df, ['ip', 'app', 'channel'], 'hour');
    gc.collect()

    X = df.drop(["is_attributed", "click_time"], axis=1).as_matrix()
    X_ttl = np.empty((0, 31))
    y_ttl = np.empty((0,))

    print(X.shape)

    if run == "train":
        y = df['is_attributed'].values

    X_ttl = np.vstack((X_ttl, X))
    y_ttl = np.concatenate((y_ttl, y))

    if (i + 1) % 10 == 0:

        if run == "train":

            chdir(sparse_array_path) # Changing directory as sparse type doesn't want to work with absolute file paths

            file = "train_" + str(i + 1) + ".csv"
            funcs.from_sparse_to_file(file, X_ttl, deli1=" ", deli2=":", ytarget=y_ttl)

        else:

            chdir(sparse_array_path)  # Changing directory as sparse type doesn't want to work with absolute file paths

            file = "test_" + str(i + 1) + ".csv"
            funcs.from_sparse_to_file(file, X_ttl, deli1=" ", deli2=":", ytarget=None)

            X_ttl = np.empty((0, 31))




