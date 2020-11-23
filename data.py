import pandas as pd
import matplotlib.pyplot as plt



def retrieve_data():
    btc_data = '/mnt/c/Downloads/Nov_btc.csv'
    df = pd.read_csv(btc_data)
    df = df.drop(columns=['date', 'weighted','volume'])
# Columns are set at close, high, low and open.
    df = df.dropna()
    data = df.values
    return data

# Test Data = data[:250]
# Train Data = data[250:]



