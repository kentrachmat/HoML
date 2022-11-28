import csv
from pathlib import Path

import numpy as np
import pandas as pd
import os
from skimage.io import imsave, imread
from sklearn import decomposition, metrics, model_selection, naive_bayes, pipeline

def number_lines(fname):
    with open(fname) as f:
        return sum(1 for l in f)

def fast_import(arr, fpath):
    with open(fpath) as f:
        for i, row in enumerate(csv.reader(f, delimiter=" ")):
            arr[i] = row

def convert_to_torch(D_PATH, xs, ys, labels_df):
    for key in xs.keys():
      os.makedirs(D_PATH, exist_ok=True)
      data_dir = os.path.join(D_PATH, key)
      os.makedirs(data_dir, exist_ok=True)
      for i in range(len(xs[key])):
        img = xs[key][i].reshape(128, 128, -1)
        if key != 'test' and key != 'valid':
          idx = int(ys[key][i])
          label = labels_df['name'].iloc[idx]
          class_dir = os.path.join(data_dir, label)
        else:
          class_dir = data_dir
        os.makedirs(class_dir, exist_ok=True)
        imsave("{}/{}_{}.png".format(class_dir,key,i), np.uint8(img))

D_PATH = Path("../data/new_data")
DATA_PATH = Path("../data/public_data")
DATA_NAME = "Areal"
DATA_SETS = ["train", "valid", "test"]
def fmain(D_PATH, DATA_PATH, DATA_NAME, DATA_SETS):

    num_fts = number_lines(DATA_PATH / f"{DATA_NAME}_feat.name")

    num = {
        data_set: number_lines(DATA_PATH / f"{DATA_NAME}_{data_set}.data")
        for data_set in DATA_SETS
    }

    xs_raw = {
        data_set: np.empty((num[data_set], num_fts))
        for data_set in DATA_SETS
    }

    for data_set in DATA_SETS:
        fast_import(
            xs_raw[data_set],
            fpath=DATA_PATH / f"{DATA_NAME}_{data_set}.data"
        )

    labels_df = pd.read_csv(
        DATA_PATH / f"{DATA_NAME}_label.name", header=None, names=["name"]
    )

    ys_df = pd.read_csv(
        DATA_PATH / f"{DATA_NAME}_train.solution", header=None, names=["value"]
    )

    ys_raw = ys_df.values.squeeze()

    ys_df["label"] = ys_df.value.map(labels_df.name)

    xs, ys = {}, {}
    (
        xs["train"],
        xs["valid-lab"],
        ys["train"],
        ys["valid-lab"],
    ) = model_selection.train_test_split(
        xs_raw["train"], ys_raw, test_size=0.2, random_state=123
    )

    xs["test"], xs["valid"] = xs_raw["test"], xs_raw["valid"]
    convert_to_torch(D_PATH, xs, ys, labels_df)

if __name__ == '__main__':
    fmain()
