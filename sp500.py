import csv
import numpy as np
from scipy.linalg import toeplitz, hankel

class sp500:
    def __init__(self, window_size):
        self.r = self.load_data(window_size)

    def load_data(self, window_size):
        r = []
        with open('sp500c.csv', mode='r') as file:
            csvFile = csv.DictReader(file)
            for row in csvFile:
                print(row['RETURNS']) # Access data by column name
                r.append(float(row['RETURNS']))
        num = len(r) - window_size

        #Add noise
        r = r + np.random.normal(0.0, 0.05, size=(len(r),))
        
        H = hankel(r[:num+1],r[num:])
        H = np.cumsum(H, axis=1)
        return H

