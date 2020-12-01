import pandas as pd
import matplotlib.pyplot as plt

def retrieve_data():
    train_data = 'data/Nov_btc.csv'
    df = pd.read_csv(train_data)
    df = df.drop(columns=['date', 'weighted','volume'])
# Columns are set at close, high, low and open.
    df = df.dropna()
    df.to_csv("data/cpp_data.csv",index=False)


retrieve_data()


