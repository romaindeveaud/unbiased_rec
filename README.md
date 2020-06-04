# Unbiased Recommendations

This repository contains tools and utilities to model the exposure
and ultimately debias the recommendations of a live mobile recommender system.

### 0. Data analysis notebooks
Jupyter notebooks are available in the `notebook` folder. They provide a first level of analysis over the data and the problems we (might) encounter.
Here a list:
* https://github.com/romaindeveaud/unbiased_rec/blob/master/notebooks/Documents.ipynb

Several steps are required to estimate propensities of observation from our recommendation data.

### 0.1. Setting up a fresh environment
I used `conda` as Python environment manager. The following command sets up a new environment and installs the required packages:
```
$ conda env create -f environment.yml
$ conda activate unbiased_rec
```

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
  -a, --all                 Train user embeddings for all days available in
                            data.

  -d, --day TEXT            Train embeddings for a single day. Provide a date
                            to the yyyymmdd format. Example: 20191119

  -k, --num_dim INTEGER     The number of dimensions of the embeddings.
                            [default: 32; required]

  -e, --num_epochs INTEGER  The number of training epochs.  [default: 10;
                            required]

  -b, --batch_size INTEGER  Mini-batch size for each iteration of SGD.
                            [default: 256; required]

  -v, --verbose             [default: False]
  --help                    Show this message and exit.
```
Use the `-d` option if you want to produce embeddings only for a single day.
To train on all data (recommended), use the `-a` option.

As an example, here is the command to train embeddings over all the pre-processed data stored in `DATASET_PATH`, with embeddings of size `100`, with a verbose output:
```
$ python train_user_embeddings.py --all --num_dim=32 --num_epochs=10 --batch_size=256  -v
```
Training is performed using the [Spotlight](https://github.com/maciejkula/spotlight) library; it will use the GPU if available.

## 2. Grouping similar users
With user embeddings trained and serialised, the next step is to cluster them together in order to use these groups as "queries".
This step is performed by the `clustering.py` script:
```
$ python clustering.py --help
Usage: clustering.py [OPTIONS]

Options:
  -c, --num_clusters INTEGER      The number of user clusters.  [default: 100;
                                  required]

  -m, --model [AgglomerativeClustering|MiniBatchKMeans]
                                  The clustering model.  [default:
                                  AgglomerativeClustering; required]

  -e, --embeddings_file TEXT      Path to the file containing the serialised
                                  user embeddings.  [default: ./datasets/seque
                                  ntial_exposure_explicit_sample_top3k/all_pu_
                                  k50.npy; required]

  -q, --output_query_users BOOLEAN
                                  Whether the script should save clusters of
                                  users to be further used as 'queries'.
                                  [default: True]

  --help                          Show this message and exit.
```

As an example, here is the command for clustering pre-trained embeddings into 200 clusters, and further serialising these clusters:
```
$ python clustering.py --num_clusters=200 --model=AgglomerativeClustering --embeddings_file=datasets/sequential_exposure_explicit_sample_top3k/all_pu_k50.npy --output_query_users
```

## 3. Fitting a click model

## 4. Estimating propensities
