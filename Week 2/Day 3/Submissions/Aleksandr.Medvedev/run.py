import os
import argparse
import numpy as np
import pandas as pd
import tarfile
import urllib.request
import zipfile
from glob import glob
from dask_ml.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from dask_ml.model_selection import train_test_split
from dask_ml.linear_model import LinearRegression
from dask.distributed import Client
import dask.dataframe as dd
import dask.array as da

data_dir = 'data'

def load_flights(data_dir, nyc_url):
    flights_raw = os.path.join(data_dir, 'nycflights.tar.gz')
    flightdir = os.path.join(data_dir, 'nycflights')

    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    if not os.path.exists(flights_raw):
        print("- Downloading NYC Flights dataset... ", end='', flush=True)
        urllib.request.urlretrieve(nyc_url, flights_raw)
        print("done", flush=True)

    if not os.path.exists(flightdir):
        print("- Extracting flight data... ", end='', flush=True)
        tar_path = os.path.join('data', 'nycflights.tar.gz')
        with tarfile.open(tar_path, mode='r:gz') as flights:
            flights.extractall('data/')
        print("done", flush=True)



def encode_data(df):
    categorical = ['Year', 'Month', 'DayOfWeek', 'UniqueCarrier', 'Origin', 'Dest']
    #for cat in categorical:
    #    print(f'{cat}: {np.unique(df[cat])}')
    encoder = OneHotEncoder(sparse=False)
    transformed = encoder.fit_transform(df.loc[:, categorical])
    quantitative = ['CRSDepTimeHours', 'Distance', 'TaxiIn']
    return da.concatenate([transformed.to_dask_array(), 
                           df.loc[:, quantitative].to_dask_array()], 
                          axis=1, allow_unknown_chunksizes=True)
    
    
def filter_data(df):
    df.loc[:, 'CRSDepTimeHours'] = df.CRSDepTime // 100
    cols = ['Year', 'Month', 'DayOfWeek', 'CRSDepTimeHours', 
            'Distance', 'TaxiIn', 'UniqueCarrier', 'Origin', 'Dest', 'DepDelay']
    df = df.fillna(0)
    return df.loc[df.Cancelled, cols]
    

def df_to_array(df, y_col):
    
    y = df[y_col].values
    del df[y_col]
    data = encode_data(df)
    return data, y

    
def get_full_data(data_dir):
    cols = ['Year', 'Month', 'DayOfWeek', 'CRSDepTime', 'Cancelled',
            'Distance', 'TaxiIn', 'UniqueCarrier', 'Origin', 'Dest', 'DepDelay']
    
    categorical = ['Year', 'Month', 'DayOfWeek', 'UniqueCarrier', 'Origin', 'Dest']
    
    data = dd.read_csv(os.path.join(data_dir, 'nycflights', '*.csv'), usecols=cols, dtype={'Cancelled': 'bool'})
    data = data.categorize(columns=categorical)
    X, y = df_to_array(data, y_col='DepDelay')
    
    return X, y


def prepare_dataset(X, y):
    scaler = StandardScaler()

    X.compute_chunk_sizes()
    X_train, X_test, y_train, y_test = train_test_split(X.rechunk({1: X.shape[1]}), y, test_size=0.25)
    del X
    del y

    X_test, X_valid, y_test, y_valid = train_test_split(X_test, y_test, test_size=0.5)
    
    y_train = scaler.fit_transform(y_train.compute().reshape(-1, 1))
    y_test = scaler.transform(y_test.compute().reshape(-1, 1))
    y_valid = scaler.transform(y_valid.compute().reshape(-1, 1))
    
    return X_train, X_test, X_valid, y_train, y_test, y_valid


def train(model, X_train, X_test, X_valid, y_train, y_test, y_valid):
    model.fit(X_train, y_train)
    test_score = model.score(X_test, y_test)
    valid_score = model.score(X_valid, y_valid)
    return test_score, valid_score


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Example of predicting departure delay from NYC flights')
    parser.add_argument('--data_dir', default='data', dest='data_dir', type=str)
    parser.add_argument('--nyc_url', default='https://storage.googleapis.com/dask-tutorial-data/nycflights.tar.gz', dest='nyc_url', type=str)
    
    args = parser.parse_args()
    
    client = Client(n_workers=2)
    
    load_flights(args.data_dir, args.nyc_url)
    
    X, y = get_full_data(data_dir)
    
    X_train, X_test, X_valid, y_train, y_test, y_valid = prepare_dataset(X, y)
    
    lr = LinearRegression()
    test_score, valid_score = train(lr, X_train, X_test, X_valid, y_train, y_test, y_valid)
    
    print(test_score, valid_score)