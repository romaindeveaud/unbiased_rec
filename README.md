# Unbiased Recommendations

This repository contains tools and utilities to model the exposure
and ultimately debias the recommendations of a live mobile recommender system.

### 0. Data analysis notebooks
Jupyter notebooks are available in the `notebook` folder. They provide a first level of analysis over the data and the problems we (might) encounter.
Here a list:
* https://github.com/romaindeveaud/unbiased_rec/blob/master/notebooks/Documents.ipynb

Several steps are required to estimate propensities of observation from our recommendation data.

## 1. Training user embeddings

First, click data need to be pre-processed into user interactions that will serve as training data.
```
$ python format_click_data.py
```
This script will output 3 files (`user_ids`, `item_ids`, and `ratings`) per day in the folder specified in `DATASET_PATH`.
If you cloned the repository you should already have the directory structure and wouldn't need to change this path.

Then, onto the actual training of user embeddings:
```
$ python train_user_embeddings.py --help
Usage: train_user_embeddings.py [OPTIONS]

Options:
  -a, --all              Train user embeddings for all days available in data.
  -d, --day TEXT         Train embeddings for a single day. Provide a date to
                         the yyyymmdd format. Example: 20191119

  -k, --num_dim INTEGER  The number of dimensions of the embeddings.
                         [default: 50; required]

  -v, --verbose          [default: False]
  --help                 Show this message and exit.
```
Use the `-d` option if you want to produce embeddings only for a single day.
To train on all data (recommended), use the `-a` option.

As an example, here is the command to train embeddings over all the pre-processed data stored in `DATASET_PATH`, with embeddings of size `100`, with a verbose output:
```
$ python train_user_embeddings.py --all --num_dim=100 -v
```
Training is performed using the [Spotlight](https://github.com/maciejkula/spotlight) library; it will use the GPU if available.

## 2. Grouping similar users

## 3. Fitting a click model

## 4. Estimating propensities
