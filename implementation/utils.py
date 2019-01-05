import numpy as np
import pandas as pd


def autocorr_function(arr1d, n_lags=100):
    arr1d = np.squeeze(arr1d)
    arr1d_ser = pd.Series(arr1d)
    lags = np.arange(start=1, stop=n_lags+1)
    autocorrs = np.array([arr1d_ser.autocorr(lag=lag) for lag in lags])
    return lags, autocorrs