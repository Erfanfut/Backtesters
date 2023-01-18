
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
plt.style.use("seaborn")

class MLBacktester():
    ''' 
    Class for the vectorized backtesting of Machine Learning-based trading strategies (Classification).
    '''

    def __init__(self):


        self.model = LogisticRegression(C = 1e6, max_iter = 100000, multi_class = "ovr")
        self.results = None
        self.get_data()
        self.symbol = 'AMAZON'
                             
    def get_data(self):
        ''' 
        Imports the data from AMZN.csv (source can be changed).
        '''
        raw = pd.read_csv("AMZN.csv", parse_dates = ["Date"], index_col = "Date")
        raw = raw[['Adj Close']].rename(columns={'Adj Close':'price'})
        raw['returns'] = raw['price'] / raw['price'].shift(1) - 1
        raw.dropna(inplace=True)
        self.data = raw
    
    def prepare_features(self, split_ratio, lags):
        ''' 
        Prepares the feature columns for training set and test set.
        Parameters
        ----------
        split_ratio: float (between 0 and 1.0 excl.)
            Splitting the dataset into training set (split_ratio) and test set (1 - split_ratio).
        lags: int
            number of lags serving as model features.
        '''
        split = int(len(self.data) * split_ratio)
        data = self.data.copy()
        data['direction'] = np.sign(data['returns'])
        train_data = data.iloc[:split]
        test_data = data.iloc[split:]

        self.feature_columns = []
        for lag in range(1,lags + 1):
            col = "lag{}".format(lag)
            data[col] = data["returns"].shift(lag)
            train_data[col] = train_data["returns"].shift(lag)
            test_data[col] = test_data["returns"].shift(lag)
            self.feature_columns.append(col)
        data.dropna(inplace=True)
        train_data.dropna(inplace=True)
        test_data.dropna(inplace=True)
        self.train_data = train_data
        self.test_data = test_data
        self.data = data
        
    def train_test_strategy(self, split_ratio, lags):
        ''' 
        Backtests the ML-based strategy.
        
        Parameters
        ----------
        split_ratio: float (between 0 and 1.0 excl.)
            Splitting the dataset into training set (split_ratio) and test set (1 - split_ratio).
        lags: int
            number of lags serving as model features.
        '''   
        self.prepare_features(split_ratio=split_ratio,lags=lags)               
        # fit the model on the training set
        model = self.model.fit(self.train_data[self.feature_columns], self.train_data.direction)
                  
        # make predictions
        train_predict = model.predict(self.train_data[self.feature_columns])
        test_predict = model.predict(self.test_data[self.feature_columns])
        self.train_data["prediction"] = train_predict
        self.test_data["prediction"] = test_predict
       
        # calculate Strategy Returns
        self.train_data["strategy"] = self.train_data["prediction"] * self.train_data["returns"]
        self.test_data["strategy"] = self.test_data["prediction"] * self.test_data["returns"]
        
        # calculate cumulative returns for strategy & buy and hold
        self.train_data["creturns"] = self.train_data["returns"].cumsum().apply(np.exp)
        self.train_data["cstrategy"] = self.train_data['strategy'].cumsum().apply(np.exp)
        self.test_data["creturns"] = self.test_data["returns"].cumsum().apply(np.exp)
        self.test_data["cstrategy"] = self.test_data['strategy'].cumsum().apply(np.exp)
        
        train_perf = self.train_data["cstrategy"].iloc[-1] # absolute performance of the strategy in train data
        train_outperf = train_perf - self.train_data["creturns"].iloc[-1] # out-/underperformance of strategy in train data
        test_perf = self.test_data["cstrategy"].iloc[-1] # absolute performance of the strategy in test data
        test_outperf = test_perf - self.test_data["creturns"].iloc[-1] # out-/underperformance of strategy in test data
        self.results = (round(train_perf, 6), round(train_outperf, 6) , round(test_perf, 6), round(test_outperf, 6))
        return self.results
        
    def plot_results(self):
        ''' 
        Plots the performance of the trading strategy and compares to "buy and hold".
        '''
        if self.results is None:
            print("Run test_strategy() first.")
        else:
            title = "Logistic Regression: {}".format(self.symbol)
            fig,ax = plt.subplots(1,2,figsize=(20,8))
            ax[0].plot(self.train_data[["creturns", "cstrategy"]])
            ax[0].set_title(title + " (train data)")
            ax[1].plot(self.test_data[["creturns", "cstrategy"]])
            ax[1].set_title(title + " (test data)")
