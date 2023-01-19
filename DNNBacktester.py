
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
plt.style.use("seaborn")

class DNNBacktester():
    ''' 
    Class for the vectorized backtesting of DNN trading strategies (Classification).
    '''

    def __init__(self):


        self.get_data()
        self.results = None
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
        # Preparing features
        data = self.data.copy()
        self.feature_columns = []
        for lag in range(1,lags + 1):
            col = "lag{}".format(lag)
            data[col] = data["returns"].shift(lag)
            self.feature_columns.append(col)
        data.dropna(inplace=True)
        split = int(len(self.data) * split_ratio)
        train_data = data.iloc[:split]
        test_data = data.iloc[split:]
        # Preparing Labels
        returns_classes = len(np.sign(data.returns).value_counts())
        direction = pd.DataFrame(np.array(np.sign(data.returns)).astype('int'),columns=['col'])
        direction = pd.get_dummies(direction.col).to_numpy()  
        train_direction = direction[:split]
        test_direction = direction[split:]  
        
        mu,std = train_data[self.feature_columns].mean(),train_data[self.feature_columns].std()
        train_data[self.feature_columns] = (train_data[self.feature_columns] - mu) / std
        
        mu,std = test_data[self.feature_columns].mean(),test_data[self.feature_columns].std()
        test_data[self.feature_columns] = (test_data[self.feature_columns] - mu) / std
              

        self.train_data = train_data
        self.test_data = test_data
        self.data = data
        self.input_shape = train_data[self.feature_columns].shape
        self.train_direction = train_direction
        self.test_direction = test_direction
        self.return_classes = returns_classes
        
    def DNN_train_test_strategy(self, split_ratio, lags):
        ''' 
        Backtests the DNN strategy.
        
        Parameters
        ----------
        split_ratio: float (between 0 and 1.0 excl.)
            Splitting the dataset into training set (split_ratio) and test set (1 - split_ratio).
        lags: int
            number of lags serving as model features.
        '''   
        self.prepare_features(split_ratio=split_ratio,lags=lags)               
        # fit the model on the training set
        tf.keras.backend.clear_session()
        model = tf.keras.models.Sequential(
            [   
                tf.keras.layers.Dense(512,activation='linear',input_shape=[self.input_shape[1],]), 
                tf.keras.layers.Dropout(0.4),
                tf.keras.layers.Dense(128, activation="tanh"),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(self.return_classes, activation='softmax'),
            ])
        model.compile(
            loss='categorical_crossentropy',
            optimizer=tf.keras.optimizers.Adam(learning_rate = 1e-5),
            metrics=['accuracy'])
        model_history = model.fit(x=self.train_data[self.feature_columns],y=self.train_direction,
                                  validation_data=(self.test_data[self.feature_columns],self.test_direction),
                                  epochs=200,batch_size=64,verbose=2)           
                
        #predictions
        train_predict = model.predict(self.train_data[self.feature_columns])
        test_predict = model.predict(self.test_data[self.feature_columns])
        lst_train=[]
        lst_train_prob = []
        lst_test=[]
        lst_test_prob = []

        for i in range(len(train_predict)):
            lst_train.append(np.argmax(train_predict[i]))
            lst_train_prob.append(np.max(train_predict[i]))

        for i in range(len(test_predict)):
            lst_test.append(np.argmax(test_predict[i]))
            lst_test_prob.append(np.max(test_predict[i]))

        if self.return_classes == 3:
            train_predict = np.array(lst_train) -1
            test_predict = np.array(lst_test) -1
            
        else: 
            train_predict = np.array(lst_train)*2 -1
            test_predict = np.array(lst_test)*2 -1
        
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
        self.model = model
        self.model_history = model_history
        return self.results
 
    def plot_loss_accuracy(self):
        '''
        plots loss and accuracy for training and test data
        '''
        acc_train = self.model_history.history['accuracy']
        loss_train = self.model_history.history['loss']
        acc_test = self.model_history.history['val_accuracy']
        loss_test = self.model_history.history['val_loss']
        fig,ax = plt.subplots(2,2,figsize=(15,6))
        ax[0,0].plot(acc_train)
        ax[0,1].plot(acc_test)
        ax[1,0].plot(loss_train)
        ax[1,1].plot(loss_test)
        ax[0,0].set_title("acc_train")
        ax[0,1].set_title("acc_test")
        ax[1,0].set_title("loss_train")
        ax[1,1].set_title("loss_test")
                  
    def plot_results(self):
        ''' 
        Plots the performance of the trading strategy and compares to "buy and hold".
        '''
        if self.results is None:
            print("Run test_strategy() first.")
        else:
            fig,ax = plt.subplots(2,figsize=(20,12))
            ax[0].plot(self.train_data.creturns,label='creturns')
            ax[0].plot(self.train_data.cstrategy,label='cstrategy')
            ax[1].plot(self.test_data.creturns,label='creturns')
            ax[1].plot(self.test_data.cstrategy,label='cstrategy')
            ax[0].legend()
            ax[1].legend()
            ax[0].set_title("Training")
            ax[1].set_title("Validation")
