import os
import numpy as np
import pandas as pd

# viualization
#import matplotlib.pyplot
import hvplot.pandas

# forecasting
from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation
from fbprophet.diagnostics import performance_metrics

## anomaly detection
from utils import pcp


class ProcessTimeSeriesData:
    ''' 
        this class load the data from disk or database and 
        prepare train and test for forecasting.
    '''
    def __init__(self, disk=False):
        self.disk = disk
        self.df = None
        self.train = None
        self.test = None
        self.frequency = None
        
    def loadDataFromDisk(self,filename,filetype):     
        self.filename = filename
        self.filetype = filetype

        try:
            with open(self.filename,'r') as f:
                if self.filetype == 'json':
                    df = pd.read_json(f)
                elif self.filetype == 'pickle':
                    df = pd.read_pickle(f)
                else:
                    df = pd.read_csv(f)
                self.df = df
                return self.df           

        except FileNotFoundError:
            raise "FileNotFoundError"

    def train_test(self,
                start_time=None,
                time_col=None,
                log_data = False, 
                scale_data=-1, 
                difference=False,
                train_period='21 days', 
                forecast_period='7 days',
                frequency='1 H', 
                how=['sum','mean'], 
                plot=True,):
        ''' df: dataframe of data series of interest for forecasting
            output: train dataframe/Series, and test dataframe/Series

            Parameters
            ----------
            train_period: string with pd.Timedelta compatible style, e.g., '5 days',
                    '3 hours', '10 seconds'.
            forecast_period: string with pd.Timedelta compatible style, e.g., '5 days',
                    '3 hours', '10 seconds'..
            frequency: 
            how: sum,mean,std, etc
        '''
        self.log_data= log_data
        self.difference= difference
        self.scale_data = scale_data
        self.start_time = start_time
        self.frequency = frequency
        self.train_period = pd.Timedelta(train_period)
        self.forecast_period = pd.Timedelta(forecast_period)

        df_cut = self.df[:]
        df_cut[time_col] = df_cut[time_col].astype('datetime64')
        df_cut = df_cut.sort_values(time_col).set_index(time_col)
        self.df_cut = df_cut[df_cut.index>=start_time]

        cols = list(self.df_cut.columns)
        self.resampled = self.df_cut.resample(self.frequency).agg(dict(list(zip(cols,how))))

        if self.log_data:
            self.resampled = np.log10(self.resampled)
        if self.scale_data>0:
            self.resampled = self.resampled/scale_data
        if self.difference:
            self.resampled = self.resampled.diff(1)[1:]

        idx1 = self.df_cut.index[0]+self.train_period
        idx2 = idx1+self.forecast_period

        self.train = self.resampled[self.resampled.index<idx1]
        self.test = self.resampled[(self.resampled.index>=idx1)
                                   &(self.resampled.index<idx2)]
                                   
        if plot:
            for col in cols:
                display(self.train.loc[:,col].hvplot(color='k',ylabel='') *\
                self.test.loc[:,col].hvplot(color='b'))
       
        return self.train,self.test 



class Forecasting:
    ''' 
        this class utilizes fbprophet to forecast for a given time-interval. 
        Some deafults have changed. For example, with most use cases n_changepoints of 25
        could be very large if time-steps are not seconds. 

    '''

    def prophet_forecast(self,
                 in_data, # existing time-series data
                 out_data, # to forecast
                 period, 
                 regressor=False,
                 frequency=None,
                 log=False,
                 plot_components=True,
                 n_changepoints=5, #Prophet default is 25
                 yearly_seasonality=False,
                 weekly_seasonality=True, #Prophet default is auto
                 daily_seasonality=True,  #Prophet default is auto
                 seasonality_prior_scale=.1,
                 changepoint_prior_scale=.01,  # Prophet default is 0.05
                 mcmc_samples=40,  # Prophet default is 0
                 interval_width=0.90,  # Prophet default is 0.8
                 uncertainty_samples=100,):    

        self.period = period
        self.log = log 
        self.train = in_data.reset_index()
        cols = self.train.columns
        print(cols)
        print(cols[2])
        self.train.rename(columns={cols[0]:'ds',cols[1]:'y'}, inplace=True)

        if frequency:
            self.frequency = frequency
        else:
            self.frequency = self.train['ds'][1]-self.train['ds'][0]

        
        self.model = Prophet(n_changepoints=n_changepoints, #Prophet default is 2
                 yearly_seasonality=yearly_seasonality,
                 weekly_seasonality=weekly_seasonality, #Prophet default is auto
                 daily_seasonality=daily_seasonality,  #Prophet default is auto
                 seasonality_prior_scale=seasonality_prior_scale,
                 changepoint_prior_scale=changepoint_prior_scale,  # Prophet default is 0.05
                 mcmc_samples=mcmc_samples,  # Prophet default is 0
                 interval_width=interval_width,  # Prophet default is 0.8
                 uncertainty_samples=uncertainty_samples)
        
        if regressor: ## if regressor, then its future values should be part of future dataframe
            self.model.add_regressor(cols[2])
        
        self.model.fit(self.train)
        
        self.future = self.model.make_future_dataframe(periods=self.period,
                                                        freq=self.frequency)
        self.forecast = self.model.predict(self.future)
        
        self.test = out_data.reset_index()
        self.test.rename(columns={cols[0]:'ds',cols[1]:'y'}, inplace=True)
        self.test['yhat'] = self.forecast['yhat'][-len(self.test):].values

        if plot_components:
            pd.plotting.register_matplotlib_converters()
            self.model.plot_components(self.forecast);

        return self.model,self.forecast

    #### add grid search or random search for prophet hyperparametes. 

    def plot_forecast(self,ylabel='bytes',title=None):       
        
        # if self.log:
        #     self.train['y'] = 10**(self.train['y'])
        #     self.test['y'] = 10**(self.test['y'])
        #     self.forecast['yhat'] = 10**(self.forecast['yhat'])
        #     self.forecast['yhat_lower'] = 10**(self.forecast['yhat_lower'])
        #     self.forecast['yhat_upper'] = 10**(self.forecast['yhat_upper'])

        return self.train.hvplot('ds','y',color='k',alpha=.7,width=750,height=300,
               ylabel=ylabel,xlabel='date',size=7,
               title=title,grid=True,rot=30) *\
               self.test.hvplot('ds','y',color='b',kind='line',line_width=1) *\
               self.forecast.hvplot('ds','yhat',color='brown',kind='line',line_width=2) *\
               self.forecast.hvplot.area('ds','yhat_lower','yhat_upper',color='blue',alpha=.2)

    ##TODO:  
    # def performance(self,
    #                 horizon,
    #                 initial,
    #                 period,):

    #     self.horizon = horizon
    #     self.initial = initial
 
    #     self.df_cv = cross_validation(self.model,horizon=self.horizon,
    #                                     initial=self.initial,period=period)       
    #     return performance_metrics(self.df_cv,
    #         metrics=['mse', 'rmse', 'mae', 'mape','coverage'],rolling_window=1)

    def point_metric(self):
        if self.log:
            self.mape = (abs(10**self.test['y']-10**self.test['yhat'])/(10**self.test['y']))*100
            self.mse = (10**self.test['y']-10**self.test['yhat'])**2
            self.mae = abs(10**self.test['y']-10**self.test['yhat']) 
        else: 
            self.mape = (abs(self.test['y']-self.test['yhat'])/self.test['y'])*100
            self.mse = (self.test['y']-self.test['yhat'])**2
            self.mae = abs(self.test['y']-self.test['yhat'])

        return self.mse,np.sqrt(self.mse),self.mae,self.mape

    def metric(self):
        if self.log:
            self.mape = np.mean(abs(10**self.test['y']-10**self.test['yhat'])/(10**self.test['y'])*100)
            self.mse = np.mean((10**self.test['y']-10**self.test['yhat'])**2)
            self.mae = np.mean(abs(10**self.test['y']-10**self.test['yhat']))
 
        else: 
            self.mape = np.mean((abs(self.test['y']-self.test['yhat'])/self.test['y'])*100)
            self.mse = np.mean((self.test['y']-self.test['yhat'])**2)
            self.mae = np.mean(abs(self.test['y']-self.test['yhat']))

        return self.mse,np.sqrt(self.mse),self.mae,self.mape



#### Class Anomaly Detection
    # RPCA
    # Todo: Matrix Profile
    # Todo: iForest

class AnomalyDetection:
    ''' 
        this class utilizes RPCA using PCP approach to factorize time series into 
        low-rank and sparse (anomalies).
    '''

    def __init__(self):
        self.data_matrix = None
        self.rank = None
        self.lam = None
        self.low_rank = None
        self.sparse = None
        self.noise = None

    def rpca_pcp(self,
                 data, 
                 row_size=24,
                 svd_method='exact', 
                 maxiter=250, 
                 verbose=True,):
    
        col_size = len(data)//row_size
        array = np.array(data[:col_size*row_size])
        M = np.reshape(array,[row_size,col_size])
        self.data_matrix = M

        L, S, Y, (u, s, v),rank, lam = pcp(
                                           M, 
                                           maxiter=maxiter, 
                                           verbose=verbose, 
                                           svd_method=svd_method,)
        self.low_rank = L
        self.sparse = S
        self.noise = Y
        self.rank = rank
        self.lam = lam
        
        return L.flatten(),S.flatten(),Y.flatten()




