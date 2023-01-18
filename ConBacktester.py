
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn")


class ConBacktester():
    ''' 
    Class for the vectorized backtesting of simple contrarian trading strategies.
    '''    
    
    def __init__(self):

        self.results = None
        self.get_data()
        self.symbol = "AMAZON"
    
        
    def get_data(self):
        ''' 
        Imports the data from AMZN.csv (source can be changed).
        '''
        raw = pd.read_csv("AMZN.csv", parse_dates = ["Date"], index_col = "Date")
        raw = raw[['Adj Close']].rename(columns={'Adj Close':'price'})
        raw['returns'] = raw['price'] / raw['price'].shift(1) - 1
        raw.dropna(inplace=True)
        self.data = raw
        
    def test_strategy(self, window = 1):
        ''' 
        Backtests the simple contrarian trading strategy.
        
        Parameters
        ----------
        window: int
            time window (number of bars) to be considered for the strategy.
        '''
        self.window = window
        data = self.data.copy().dropna()
        data["position"] = -np.sign(data["returns"].rolling(self.window).mean())
        data["strategy"] = data["position"].shift(1) * data["returns"]
        data.dropna(inplace=True)
              
        data["creturns"] = data["returns"].cumsum().apply(np.exp)
        data["cstrategy"] = data["strategy"].cumsum().apply(np.exp)
        self.results = data
        
        perf = data["cstrategy"].iloc[-1] # absolute performance of the strategy
        outperf = perf - data["creturns"].iloc[-1] # out-/underperformance of strategy
        
        return round(perf, 6), round(outperf, 6)
    
    def plot_results(self):
        ''' 
        Plots the performance of the trading strategy and compares to "buy and hold".
        '''
        if self.results is None:
            print("Run test_strategy() first.")
        else:
            title = "Symbol = {} | Window = {}".format(self.symbol,self.window)
            self.results[["creturns", "cstrategy"]].plot(title=title, figsize=(12, 8))
            
    def optimize_parameter(self, window_range):
        ''' 
        Finds the optimal strategy (global maximum) given the window parameter range.

        Parameters
        ----------
        window_range: tuple
            tuples of the form (start, end, step size)
        '''
        
        windows = range(*window_range)
            
        results = []
        for window in windows:
            results.append(self.test_strategy(window)[0])
        
        best_perf = np.max(results) # best performance
        opt = windows[np.argmax(results)] # optimal parameter
        
        # run/set the optimal strategy
        self.test_strategy(opt)
        
        # create a df with many results
        many_results =  pd.DataFrame(data = {"window": windows, "performance": results})
        self.results_overview = many_results
        
        return opt, best_perf
                               
        