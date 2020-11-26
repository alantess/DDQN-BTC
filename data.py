import pandas as pd
import matplotlib.pyplot as plt

def retrieve_data():
    train_data = 'data/Nov_btc.csv'
    test_data  = 'data/btc_test_data.csv'
    df = pd.read_csv(test_data)
    df = df.drop(columns=['date', 'weighted','volume'])
# Columns are set at close, high, low and open.
    df = df.dropna()
    data = df.values
    return data


