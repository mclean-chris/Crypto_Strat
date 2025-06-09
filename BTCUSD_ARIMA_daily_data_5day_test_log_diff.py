import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.metrics import root_mean_squared_error
import pmdarima as pm
import pandas_market_calendars as mcal
from scipy.stats import uniform

# Create a calendar
nyse = mcal.get_calendar('NYSE')
# from statsmodels.tsa.arima.model import ARIMA
# import pmdarima as pm
import scipy.stats as stats
# from skforecast.recursive import ForecasterRecursive

data_folder = "crypto_data"
output_folder = "Results22"
output_stats_file = f"{output_folder}/BTCUSDARIMA_5DAY_l1s1sl05_bf75.csv"
start_simulation_date = '2019-01-01' 
filename = 'BTCUSD.csv'
# long_thresh = 0.0025    #0.0375 for 1 day
# short_thresh = -0.01  #-0.05 for 1 day
AR_long_thresh = 0.01
AR_short_thresh = -0.01
slippage = 0.001
XGB_data_len = 500
pred_len = 5
data_resolution = 1 # 24 for one tick every 24hrs, 1 for 1 per hour
stop_loss_thresh = 0.05   #0.025 for 24hr, 0.05 for 5 day, 0.10 for 20 day
start_offset = 1080  #28613 #1303 1080 780 350
sma_size = 20
ema_len = 3 #was 5 for "great" results
data_len_diff = 0
# param_grid = {
#     'max_depth': [3, 5, 7], # 3 5 7 10
#     'learning_rate': [0.01, 0.1, 0.3], #0.01 0.1 0.3
#     'subsample': [0.5, 0.85],
#     # 'lambda': [1, 5],
#     # 'alpha': [1, 5],
#     # 'colsample_bytree': [0.8, 0.2], #[0.8, 0.5, 0.2]
#     # 'gamma': [0.1, 0.5, 0.8],
#     # "eval_metric": ["logloss"],
#     'n_estimators': [100, 200, 300, 1000] #1000 max 100 200 300 1000
# }
scaler = StandardScaler() 
file_path = os.path.join(data_folder, filename)
symbol = filename.split('.')[0]  

df = pd.read_csv(
    file_path    # parse_dates={'datetime': ['datetime']},
    # infer_datetime_format=True
)
# df = df.iloc[10:]
# df.drop(columns=['Unnamed: 7', 'Unnamed: 8'], inplace=True)
df = df.iloc[::data_resolution].reset_index()
df = df.drop('index', axis=1)

# df = df.dropna()
df['datetime'] = pd.to_datetime(df['date'])
df = df[['datetime', 'close', 'low', 'high', 'open']]
df.sort_values('datetime', inplace=True)
# df = df[df['datetime'] >= pd.Timestamp(start_simulation_date)]

df['close_lag1'] = df['close'].shift(1)
df['ema_lag1'] = df['close_lag1'].ewm(span=ema_len, adjust=False).mean()
df['sma_lag1'] = df['close_lag1'].rolling(sma_size).mean()

# Apply logarithmic transform to stabilize variance
df['log_close_lag1'] = np.log(df['close_lag1'])
df['log_close'] = np.log(df['close'])


df['diff_log_close'] = df['log_close'].diff()
df['diff_log_close_lag1'] = df['log_close_lag1'].diff()

df['diff_close_lag1'] = df['close_lag1'].diff().dropna()
df['diff_close'] = df['close'].diff().dropna()
df['d1_momentum_lag1'] = df['ema_lag1'].rolling(2).mean().dropna()
# df['d2_momentum_lag1'] = df['diff_close_lag1'].rolling(2).mean().dropna()

df.dropna(subset=['diff_log_close', 'diff_log_close_lag1'], inplace=True)

cum_rets = []
BAs = []
stats = []
all_ret = []
prev_side = 'cash'

# BA and BA counters
trade_c = 0
trade_BA_c = 0
BA_c = 0

ema_BA = 0
ema_BA_c = 0
ema_BA_thresh_c = 0
pred_BA_c = 0
XGB_BA_c = 0
XGB_BA = 0
XGB_thresh_BA_c = 0
XGB_pred_BA_thresh_c = 1
sma_BA_thresh_c = 0
sma_pred_BA_thresh_c = 1
AR_pred_BA_thresh_c = 1
AR_thresh_BA_c = 0
AR_BA_c = 0
AR_BA = 0
posit = 1
errors = []
bf = 0
# make a list of prediction for each model,
# find the mean difference between the predicted and actual over the last 200 previous
# BE CAREFUL because you have to look back [-(200 + pred_len) : -pred_len] as you can not know the next 
# difference in prediction v actual until it has been measured, pred_len time steps from the time of prediction

for offset in np.arange(start_offset, 1, -1):

    XGB_train_test_df = df.iloc[-(XGB_data_len+offset+data_len_diff):-(offset)]
    XGB_train_test_df = XGB_train_test_df.dropna()
    XGB_train_data = XGB_train_test_df.iloc[0:-pred_len]
    XGB_test_data = XGB_train_test_df.iloc[-pred_len:]
    print("start predictions: " + str(XGB_test_data.iloc[0]['datetime']))


#XGB scaling
    #StandardScaler

    X_train = scaler.fit_transform(XGB_train_data['diff_log_close_lag1'].values.reshape(-1,1))
    y_train = scaler.transform(XGB_train_data['diff_log_close'].values.reshape(-1,1))

    X_test = scaler.transform(XGB_test_data['diff_log_close_lag1'].values.reshape(-1,1))
    y_test = scaler.transform(XGB_test_data['diff_log_close'].values.reshape(-1,1))



    # hold_dates = pd.bdate_range(start=XGB_test_data['datetime'].iloc[0], end=XGB_test_data['datetime'].iloc[-1], holidays=mcal.holidays)
    cal_dates = pd.date_range(start=XGB_test_data['datetime'].iloc[0], end=XGB_test_data['datetime'].iloc[-1])

    market_dates = set(XGB_test_data['datetime'].iloc[:])
    cal_dates = set(pd.date_range(start=XGB_test_data['datetime'].iloc[0], end=XGB_test_data['datetime'].iloc[-1]))

    market_closed = cal_dates-market_dates
    # if len(market_closed)>0
    # market_reopen_date = #the date in the market dates that is the next one greater than the greatest market closed date
    #going to check if the stop loss condition is met on any of the hold dates, or on the XGB_test_data['datetime'] == market_reopen_date
    mcs = []
    if len(market_closed) > 0:
        mcs = list(market_closed)
        mcs.sort()
        end_mcs = mcs[-1]
        market_dates = list(market_dates)
        market_dates.sort()
        market_reopen_date = [md for md in market_dates if md > end_mcs]
        market_reopen_date = market_reopen_date[0]

    act_enter = XGB_test_data.iloc[0]['close_lag1']
    act_exit = XGB_test_data.iloc[-1]['close']

    if len(errors)>20:
        
        bf = errors[-(50+pred_len):-pred_len]
        bf = bf[::-1] #used error period of 20 for ETHUSD
        bf = pd.Series(bf)
        bf = bf.ewm(span = len(bf)).mean().iloc[-1]
        # bf = np.mean(bf)
        bf = np.clip(bf, -0.05, 0.05)



    ARIMA_model = pm.auto_arima(y = y_train, X = X_train,
                              start_p=2, start_q=0, start_d = 0,
                        max_p=8, max_q=5, max_d = 1,
                        # start_P=1, start_Q=1,
                        # max_P=5, max_Q=5, 
                        # seasonal=True,
                        stepwise=True, suppress_warnings=True, 
                        # D=5, max_D=10,
                        # n_jobs=-1,
                        error_action='ignore')

    try:
        ARIMA_fit = ARIMA_model.fit(y_train)
        ARIMA_forecast = ARIMA_fit.predict(n_periods = pred_len)
        ARIMA_forecast = scaler.inverse_transform(ARIMA_forecast.reshape(-1,1))
        cumlogdiff = ((ARIMA_forecast + 1).cumprod()-1)[-1]
        AR_pred_log_close = XGB_test_data['log_close_lag1'].iloc[0] + cumlogdiff
        AR_pred_close = np.exp(AR_pred_log_close)
        init_pred_pct = (AR_pred_close - act_enter)/act_enter

        AR_pred_pct_del =   init_pred_pct + bf
        #some bias in the model -0.04
        # print("AR pred pct del: " + str(AR_pred_pct_del))
    except:
        AR_pred_pct_del = 0

    act_pct_del = (act_exit -act_enter)/act_enter
    print("AR pred pct del: " + str(AR_pred_pct_del))
    print("actual pct del: " + str(act_pct_del))
    print("bias: "+ str(bf))


# XGB model creation
#     tscv = TimeSeriesSplit(n_splits=3)
#     xgb_model = xgb.XGBRegressor(random_state=42) #
#     grid_search = GridSearchCV(
#         estimator=xgb_model,
#         param_grid=param_grid,
#         cv=tscv,
#         n_jobs=-1,
#         verbose=0,
#         # random_state=42
#     )
#     grid_search.fit(X_train, y_train)

#     best_model = grid_search.best_estimator_

# #forecast loop for XGB
#     for i in range(pred_len):
#         if i == 0:
#             prediction = best_model.predict(X_test[0]) # making a one-step prediction
#             forecast_X_train = np.append(X_train, X_test[0])
#             forecast_y_train = np.append(y_train, prediction)
            
#         else:
#             grid_search.fit(forecast_X_train.reshape(-1,1), forecast_y_train.reshape(-1,1))
#             best_model = grid_search.best_estimator_
#             prediction = best_model.predict(prior_prediction)
#             forecast_X_train = np.append(forecast_X_train, prior_prediction[0])
#             forecast_y_train = np.append(forecast_y_train, prediction[0])
#         prior_prediction = prediction
#     prediction = scaler.inverse_transform(prediction.reshape(-1,1))

#     # prediction = np.exp(prediction[0][0])
#     pred_pct_del = prediction[0][0]
#     # XGB pred_pct_dels
# # 

#     #this is the difference of the logs
#     # so to undo, add this difference to the last log, then exp() then this is the pred_close

#     pred_log_close = XGB_test_data['log_close_lag1'].iloc[0] + prediction[0][0]
#     pred_close = np.exp(pred_log_close)


#     pred_pct_del = (pred_close - act_enter)/act_enter
#     # pred_pct_del = (prediction-act_enter)/act_enter #when using non-diff data

    #straight XGB BA, not going to be great
    # if np.sign(pred_pct_del) == np.sign(act_pct_del):
    #     XGB_BA_c = XGB_BA_c+1

    if np.sign(AR_pred_pct_del) == np.sign(act_pct_del):
        AR_BA_c = AR_BA_c+1
    #XGB BA with threshold
        # if ((pred_pct_del > long_thresh) and (np.sign(pred_pct_del)) == np.sign(act_pct_del))\
        # or ((pred_pct_del < short_thresh) and (np.sign(pred_pct_del)) == np.sign(act_pct_del)):
        #     XGB_thresh_BA_c = XGB_thresh_BA_c + 1

    if ((AR_pred_pct_del > AR_long_thresh) and (np.sign(AR_pred_pct_del)) == np.sign(act_pct_del))\
        or ((AR_pred_pct_del < AR_short_thresh) and (np.sign(AR_pred_pct_del)) == np.sign(act_pct_del)):
            AR_thresh_BA_c = AR_thresh_BA_c + 1
        #XGB BA with threshold
    
    pred_BA_c = pred_BA_c + 1
    # XGB_thresh_BA = XGB_thresh_BA_c/pred_BA_c

    votes = []
    side = 'cash'
    #XGB vote 
    # if (pred_pct_del > (long_thresh)):
    #     # pred_pct_del > (XGB_test_data.iloc[0]['ema1_lag1']/act_enter + long_thresh)):
    #     xgb_vote = 'long'
    #     XGB_pred_BA_thresh_c = XGB_pred_BA_thresh_c + 1
    # elif (pred_pct_del < (short_thresh)):
    #     #   pred_pct_del < (XGB_test_data.iloc[0]['ema1_lag1']/act_enter + short_thresh)):
    #     xgb_vote = 'short'
    #     XGB_pred_BA_thresh_c = XGB_pred_BA_thresh_c + 1
    # else:
    #     xgb_vote = 'none'
    # print("xgb_vote: " + xgb_vote)
    # votes.append(xgb_vote)

    if (AR_pred_pct_del > (AR_long_thresh)):
        # pred_pct_del > (XGB_test_data.iloc[0]['ema1_lag1']/act_enter + long_thresh)):
        ar_vote = 'long'
        AR_pred_BA_thresh_c = AR_pred_BA_thresh_c + 1
    elif (AR_pred_pct_del < (AR_short_thresh)):
        #   pred_pct_del < (XGB_test_data.iloc[0]['ema1_lag1']/act_enter + short_thresh)):
        ar_vote = 'short'
        AR_pred_BA_thresh_c = AR_pred_BA_thresh_c + 1
    else:
        ar_vote = 'none'
    print("ar_vote: " + ar_vote)
    votes.append(ar_vote)


    
    # if (XGB_train_data['ema_lag1'].iloc[-1] > XGB_train_data['sma_lag1'].iloc[-1]):
        
    #     sma_vote = 'long'
    #     sma_pred_BA_thresh_c = sma_pred_BA_thresh_c + 1
    #     if act_pct_del > 0:
    #         sma_BA_thresh_c = sma_BA_thresh_c + 1
    # elif (XGB_train_data['ema_lag1'].iloc[-1] < XGB_train_data['sma_lag1'].iloc[-1]):
        
    #     sma_vote = 'short'
    #     sma_pred_BA_thresh_c = sma_pred_BA_thresh_c + 1
    #     if act_pct_del<0:
    #         sma_BA_thresh_c = sma_BA_thresh_c+ 1
    # else:
    #     sma_vote = 'none'
    # print("sma_vote: " + sma_vote)
    # votes.append(sma_vote)


    # if (XGB_train_data['d1_momentum_lag1'].iloc[-1] > 0):
        
    #     sma_vote = 'long'
    #     sma_pred_BA_thresh_c = sma_pred_BA_thresh_c + 1
    #     if act_pct_del > 0:
    #         sma_BA_thresh_c = sma_BA_thresh_c + 1
    # elif(XGB_train_data['d1_momentum_lag1'].iloc[-1] < 0):
        
    #     sma_vote = 'short'
    #     sma_pred_BA_thresh_c = sma_pred_BA_thresh_c + 1
    #     if act_pct_del<0:
    #         sma_BA_thresh_c = sma_BA_thresh_c+ 1
    # else:
    #     sma_vote = 'none'
    # print("sma_vote: " + sma_vote)
    # votes.append(sma_vote)

    long_votes = [vote for vote in votes if vote == 'long']
    short_votes = [vote for vote in votes if vote == 'short']

    vote = 'none'

    # tally votes
    if len(long_votes) > len(short_votes):
        vote = 'long'
    elif len(short_votes) > len(long_votes):
        vote = 'short'
    else:
        vote = 'tied'

    stop = 0
    if vote == 'long' or vote == 'short':
        for index, row in XGB_test_data.iterrows():
        # for row in XGB_test_data:
            if stop == 0:
                if vote == 'long':
                    if row['open'] < act_enter*(1-stop_loss_thresh):
                        stop = (row['open'] - act_enter)/act_enter - slippage
                    elif row['low'] < act_enter*(1-stop_loss_thresh):
                        stop = -1*stop_loss_thresh - slippage
                elif vote == 'short':
                    if row['open'] > act_enter*(1+stop_loss_thresh):
                        stop = (act_enter - row['open'])/act_enter - slippage
                    elif row['low'] > act_enter*(1+stop_loss_thresh):
                        stop = -1*stop_loss_thresh-slippage
            else:
                break

    if stop != 0:
        print("stop: " + '{0:.3f}'.format(stop))

    
    # weights = [XGB_BA]
    posit = 0
    #make trade
    side = 'cash'
    if (vote == 'long') or (vote == 'short'):
        trade_c = trade_c + 1
        trade_ret = 0
        if vote == 'long':
            side = 'long'
            posit = 1
            if stop != 0:
                trade_ret = stop*posit
            elif stop == 0:
                trade_ret = act_pct_del*posit
        if vote == 'short':
            side = 'short'
            posit = 1
            if stop != 0:
                trade_ret = stop*posit
            elif stop == 0:
                trade_ret = -1*act_pct_del*posit
        if side == prev_side:
            trade_ret = trade_ret + slippage
    else:
        trade_ret = 0

    if trade_ret>0:
        trade_BA_c = trade_BA_c+1

    trade_ret = trade_ret/pred_len

    all_ret.append(trade_ret)

    if trade_c>0:
        trade_BA = trade_BA_c/trade_c
    else: 
        trade_BA = 0

    stats.append({
        'symbol': symbol,
        'date_entered': XGB_train_data['datetime'].iloc[-1],
        'date_exited': XGB_test_data['datetime'].iloc[pred_len-1],
        'price_entered': act_enter,
        'predicted_close': AR_pred_close,
        'price_exited': act_exit,
        'signal': side,
        'trade_return': trade_ret
    })

    all_ret_1 = [1+ret for ret in all_ret]
    cum_ret_list = np.cumprod(all_ret_1)
    cum_ret_list = cum_ret_list-1
    cum_ret = cum_ret_list[-1]
    
    errors.append(act_pct_del-AR_pred_pct_del)
    # XGB_BA = XGB_BA_c/pred_BA_c
    # XGB_thresh_BA = XGB_thresh_BA_c/XGB_pred_BA_thresh_c
    # sma_BA = sma_BA_thresh_c/pred_BA_c
    AR_BA = AR_BA_c/pred_BA_c
    AR_thresh_BA = AR_thresh_BA_c/AR_pred_BA_thresh_c


    print(" Cum Return: " + '{0:.2f}'.format(cum_ret*100) + "%. \nThis side: " + side)
    print("offset: " + str(offset))
    print("XGB Model BA: " +'{0:.3f}'.format(XGB_BA))
    # print("XGB_thresh_BA: " + '{0:.3f}'.format(XGB_thresh_BA))
    print("AR Model BA: " +'{0:.3f}'.format(AR_BA))
    print("AR_thresh_BA: " + '{0:.3f}'.format(AR_thresh_BA))
    print("trade BA: " + '{0:.3f}'.format(trade_BA))
    # print("sma BA: " + '{0:.3f}'.format(sma_BA))
    print("posit: " + '{0:.3f}'.format(posit))

    prev_side = side

    stats_df = pd.DataFrame(stats)

    stats_df.to_csv(output_stats_file, index=False)
print(f"Statistics written to {output_stats_file}")
