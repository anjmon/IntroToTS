import pandas as pd
import math
import numpy as np
def autocorr(df:pd.DataFrame, cols:list[str], max_lag:int) -> pd.DataFrame:


    indices = ["Lag "+str(lag) for lag in range(1, max_lag+1)]
    autocorr_df = pd.DataFrame(columns=cols, index=indices)

    for column in cols:
        ser = df[column]
        col_autocorrs = []
        for lag in range(1, max_lag+1):
            col_autocorrs.append( ser.autocorr(lag=lag))
        autocorr_df[column] = col_autocorrs

    return autocorr_df


def is_stationary(df:pd.DataFrame, cols:list[str], max_lag) -> dict[str, bool]:
    autocorr_df = autocorr(df, columns_to_check, max_lag=max_lag)
    statn_dict ={}
    sample_size = df.shape[0]
    for col in cols:

        l_bound, u_bound = (-1.96/math.sqrt(sample_size), 1.96/math.sqrt(sample_size))
        max_outlier_samples = sample_size*0.05
        ser = autocorr_df[col]

        num_outliers =ser[(ser < l_bound) | (ser > u_bound)].count()


        is_statn = num_outliers <= max_outlier_samples
        statn_dict[col] =  is_statn
    return statn_dict
def plot_autocorr(df, cols:list[str]) -> None:
    # for column in cols:
    #     pd.plotting.autocorrelation_plot(df[column])
    pass



# Test
np.random.seed(10)
data = {
    'col1': np.random.normal(0, 0.1, 100),   # Stationary-like series (normal distribution)
    'col2': np.random.normal(0, 1, 100).cumsum(),  # Non-stationary series (random walk)
    'col3': np.random.normal(0, 5, 100),  # Another stationary series with larger variance
}

df = pd.DataFrame(data)


columns_to_check = ['col1', 'col2', 'col3']

stationarity_results = is_stationary(df, columns_to_check, max_lag=40)
print(stationarity_results)
