import os
import pandas as pd
import numpy as np
# import xgboost as xgb
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.metrics import root_mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm
import scipy.stats as stats
from itertools import product

data_folder = "sim_data_05_10"
output_folder = "Results15"

start_simulation_date = '2018-01-01' 
filename = 'BTCUSD.csv'
long_thresh = 0.005    #0.001 for 24hr
short_thresh = -0.01  #-0.001 for 24hr, 0.01 for 5day
slippage = 0.001#about 100 or more for 5 day, 200 or more for 20 day #500 for 24hr
# XGB_data_len = 100
pred_len = 5
data_resolution = 24 # 24 for one tick every 24hrs, 1 for 1 per hour
stop_loss_thresh = 0.05    #0.025 for 24hr, 0.05 for 5 day, 0.10 for 20 day
start_offset = 1951  #28613 #1935
sma_len = 2

#things to test:
# data length
data_length = [250]
# start_offset (1-5)
# start = [1, 2, 3, 4, 5]
upper_thresh = [0.005, 0.01, 0.02, 0.05]
lower_thresh = [-0.01, -0.025, -0.05, -0.07]
# variable
variables = ['log_close_lag1']
# log_lag_close
# lag_close
# sma_3

test_grid = {'data_length': data_length, 'variable': variables, 'upper_thresh': upper_thresh, 'lower_thresh': lower_thresh}

#things to record:
# cum_return
# BA
# trade_return


# param_grid = {
#     'max_depth': [3, 5, 7],
#     'learning_rate': [0.01, 0.1, 0.3],
#     'subsample': [0.5, 0.85],
#     'lambda': [1, 5],
#     'colsample_bytree': [1, 0.5, 0.2],
#     'n_estimators': [100, 200, 300]
# }

file_path = os.path.join(data_folder, filename)
symbol = filename.split('.')[0]  

df = pd.read_csv(
    file_path,
    parse_dates={'datetime': ['date', 'time']},
    infer_datetime_format=True
)
df = df.iloc[10:]
df = df.iloc[::data_resolution].reset_index()
df = df.drop('index', axis=1)

df = df.dropna()
# df_shifted = df.shift(-20)
df['datetime'] = pd.to_datetime(df['datetime'])
df.sort_values('datetime', inplace=True)
df = df[df['datetime'] >= pd.Timestamp(start_simulation_date)]

df['close_lag1'] = df['close'].shift(1)
df['vol_lag1'] = df['vol'].shift(1)
df['log_close_lag1'] = np.log(df['close_lag1']) #make sma of log _close
df['log_close'] = np.log(df['close'])
df['sma_lag1'] = df['close_lag1'].rolling(window=sma_len).mean()


iter_list = list(product(test_grid['data_length'], test_grid['variable'], test_grid['upper_thresh'], test_grid['lower_thresh']))

for iter in iter_list:
    output_stats_file = str(iter[0])+ "_" + str(iter[1])+ "_" + str(iter[2])+ "_" + str(iter[3]) + ".csv"
    output_stats_file_name = os.path.join(output_folder, output_stats_file)
    if not os.path.isfile(output_stats_file_name):
        stats = []
        cum_ret = 0
        all_ret = []
        all_ret = []
        pred_BA_c = 0

        AR_BA_c = 0
        AR_thresh_BA_c = 0
        AR_pred_BA_thresh_c = 1
        for offset in np.arange(start_offset, 1, -pred_len):
            for s in range(pred_len): 
                offset = offset+s
                
                AR_train_test_df = df.iloc[-(offset+iter[0]):-offset]
                AR_train_data = AR_train_test_df.iloc[0:-pred_len]
                AR_test_data = AR_train_test_df.iloc[-pred_len:]
                print("start predictions: " + str(AR_test_data.iloc[0]['datetime']))

                var_col_name = iter[1]
                if var_col_name[:3] == 'log':
                    log_lin = 'log'
                    target_col_name = 'log_close'
                else:
                    log_lin = 'lin'
                    target_col_name = 'close'

            # AR Auto model creation        
                act_enter = AR_test_data.iloc[0]['close_lag1']
                act_exit = AR_test_data.iloc[-1]['close']

                scaler = StandardScaler()
                X_train = scaler.fit_transform(AR_train_data[var_col_name].values.reshape(-1,1))
                y_train = scaler.transform(AR_train_data[target_col_name].values.reshape(-1,1))

                X_test = scaler.transform(AR_test_data[var_col_name].values.reshape(-1,1))
                y_test = scaler.transform(AR_test_data[target_col_name].values.reshape(-1,1))


            # AR model creation
                ARIMA_model = pm.auto_arima(y = y_train, X = X_train,
                                start_p=0, start_q=0, start_d = 0,
                        max_p=8, max_q=5, max_d = 1,
                        start_P=1, start_Q=1,
                        max_P=5, max_Q=5, 
                        seasonal=True,
                        stepwise=True, suppress_warnings=True, 
                        D=5, max_D=10,
                        # n_jobs=-1,
                        error_action='ignore')

                try:
                    ARIMA_fit = ARIMA_model.fit(y_train)
                    ARIMA_forecast = ARIMA_fit.predict(n_periods = pred_len)
                    ARIMA_forecast = scaler.inverse_transform(ARIMA_forecast.reshape(-1,1))
                    if log_lin == "log":
                        predictions = np.exp(ARIMA_forecast)
                    elif log_lin == "lin":
                        predictions = ARIMA_forecast
                    # enter_price = scaler.inverse_transform(X_test.iloc[0])
                    
                    predicted_close = predictions[-1]
                except:
                    # ARIMA_forecast = X_test.iloc[:1]
                    # ARIMA_forecast = scaler.inverse_transform(ARIMA_forecast)
                    predictions =  ARIMA_forecast
                    pred_pct_del = 0
                    
                    # AR pred_pct_dels
                    # if log_lin == 'log':
                    #     predicted_close = np.exp(prediction[0][0])
                    #     predictions = list(np.exp(predictions))
                    #     pred_pct_del = (np.exp(prediction[0][0])-act_enter)/act_enter
                    # elif log_lin == 'lin':
                    #     predicted_close = prediction[0][0]
                    #     pred_pct_del = (prediction[0][0]-act_enter)/act_enter
                pred_pct_del = (predictions[-1]- act_enter)/act_enter
                act_pct_del = (act_exit -act_enter)/act_enter   

                    #straight AR BA, not going to be great
                if np.abs(pred_pct_del)> 0.000001 and (np.sign(pred_pct_del) == np.sign(act_pct_del)):
                    AR_BA_c = AR_BA_c+1

                trade_ret = 0
                votes = []
                side = 'cash'
                if (pred_pct_del > (iter[2])):
                    side = 'long'
                    AR_pred_BA_thresh_c = AR_pred_BA_thresh_c + 1
                    trade_ret = act_pct_del
                elif (pred_pct_del < (iter[3])):
                    side = 'short'
                    AR_pred_BA_thresh_c = AR_pred_BA_thresh_c +1
                    trade_ret = -1*act_pct_del
                
                if trade_ret>0:
                    AR_thresh_BA_c = AR_thresh_BA_c + 1

                pred_BA_c = pred_BA_c + 1

                trade_ret = trade_ret/pred_len

                all_ret.append(trade_ret)

                AR_BA = AR_BA_c/pred_BA_c
                AR_thresh_BA = AR_thresh_BA_c/AR_pred_BA_thresh_c

                all_ret_1 = [1+ret for ret in all_ret]
                cum_ret_list = np.cumprod(all_ret_1)
                cum_ret_list = cum_ret_list-1
                cum_ret = cum_ret_list[-1]

                if cum_ret<-0.95:
                    break

                d = {
                    'symbol': symbol,
                    'date_entered': AR_train_data['datetime'].iloc[-1],
                    'date_exited': AR_test_data['datetime'].iloc[pred_len-1],
                    'price_entered': act_enter,
                    'price_exited': act_exit,
                    'signal': side,
                    'trade_return': trade_ret,
                    'BA': AR_BA,
                    'BA_thresh': AR_thresh_BA,
                    'cum_return': cum_ret,
                    'predicted_close': predicted_close
                }

                for i in range(len(predictions)):
                    pred_lag = 'pred_lag ' + str(i)
                    d[pred_lag] = predictions[i][0] 
                for i in range(len(predictions)):
                    actual_lag = 'actual_lag ' + str(i)
                    d[actual_lag] = AR_test_data.iloc[i]['close']
                    

                stats.append(d)
            
                print(iter)
                print(" Cum Return: " + '{0:.2f}'.format(cum_ret*100) + "%. \nThis side: " + side)
                print("AR_thresh_BA: " + '{0:.3f}'.format(AR_thresh_BA))
                
                print(" AR BA: " + '{0:.2f}'.format(AR_BA))
                print("iter: " + str(iter))

                stats_df = pd.DataFrame(stats)

                stats_df.to_csv(output_stats_file_name, index=False)
                print(f"Statistics written to {output_stats_file_name}")