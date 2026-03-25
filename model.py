import numpy as np
import pandas as pd

class SimpleLinearRegression:
    def __init__(self):
        pass

    def compute_w1(self):
        num = np.sum((self.X - self.Xmean) * (self.y - self.ymean))
        den = np.sum((self.X - self.Xmean) ** 2)
        self.w1 = num / den

    def compute_w0(self):
        w0 = self.ymean - (self.w1*self.Xmean)
        self.w0 = w0
    
    def coefficients(self):
        print(f"W1: {self.w1}\nw0: {self.w0}")
        return self.w1, self.w0
    
    def fit(self, X, y):
        self.X = X
        self.y = y
        self.Xmean = np.mean(X)
        self.ymean = np.mean(y)
        self.compute_w1()
        self.compute_w0()
    
    def predict(self, X):
        predictions = []
        for x in X:
            y = (self.w1*x) + self.w0
            predictions.append(y)
        return predictions
    
    def r_squared(self, y_actual, y_pred):
        ss_res = []
        ss_tot = []
        for i in range(len(y_actual)):
            y1 = (y_actual[i] - y_pred[i])**2
            y2 = (y_actual[i] - self.ymean)**2
            ss_res.append(y1)
            ss_tot.append(y2)
        
        r_sq = 1 - (np.sum(ss_res) / np.sum(ss_tot))
        return r_sq