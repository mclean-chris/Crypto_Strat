import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.metrics import root_mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
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
output_folder = "Results21"
output_stats_file = f"{output_folder}/ETHUSD_XGB_diifclose_moving_data_len_l05s05_sl05.csv"
start_simulation_date = '2019-01-01' 
filename = 'ETHUSD.csv'
long_thresh = 0.005    #0.0375 for 1 day
short_thresh = -0.005  #-0.05 for 1 day
AR_long_thresh = 0.001
AR_short_thresh = -0.001
slippage = 0.001
XGB_data_len = 500
pred_len = 5
data_resolution = 1 # 24 for one tick every 24hrs, 1 for 1 per hour
stop_loss_thresh = 0.05   #0.025 for 24hr, 0.05 for 5 day, 0.10 for 20 da
start_offset = 1080  #857
sma_size = 200   #was 20 for great results
ema_len = 50 #was 5 for "great" results
data_len_diff = 0


param_grid = {
    'max_depth': [3, 5, 7], # 3 5 7 10
    'learning_rate': [0.01, 0.1, 0.3], #0.01 0.1 0.3
    'subsample': [0.7, 0.9],
    # 'lambda': [1, 5],
    # 'alpha': [1, 5],
    'colsample_bytree': [0.8, 0.2], #[0.8, 0.5, 0.2]
    # 'gamma': [0.1, 0.5, 0.8],
    # "eval_metric": ["logloss"],
    # 'reg_lambda': [0.5, 5],
    'tree_method' : ['hist'],
    'n_estimators': [100, 300, 500, 1000] #1000 max 100 200 300 1000
}
scaler = MinMaxScaler()
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

# make a list of prediction for each model,
# find the mean difference between the predicted and actual over the last 200 previous
# BE CAREFUL because you have to look back [-(200 + pred_len) : -pred_len] as you can not know the next 
# difference in prediction v actual until it has been measured, pred_len time steps from the time of prediction
errors = [] #to be used for bias factor
bf = 0

for offset in np.arange(start_offset, 1, -1):

    XGB_train_test_df = df.iloc[-(XGB_data_len+offset+data_len_diff):-(offset)]
    if len(XGB_train_test_df)< XGB_data_len:
        print("Not enough test data, shift to later date")
    XGB_train_test_df = XGB_train_test_df.dropna()
    XGB_train_data = XGB_train_test_df.iloc[0:-pred_len]
    XGB_test_data = XGB_train_test_df.iloc[-pred_len:]
    print("start predictions: " + str(XGB_test_data.iloc[0]['datetime']))


#XGB scaling
    #StandardScaler

    X_train = scaler.fit_transform(XGB_train_data['diff_close_lag1'].values.reshape(-1,1))
    y_train = scaler.transform(XGB_train_data['diff_close'].values.reshape(-1,1))

    X_test = scaler.transform(XGB_test_data['diff_close_lag1'].values.reshape(-1,1))
    y_test = scaler.transform(XGB_test_data['diff_close'].values.reshape(-1,1))

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

    # track the error from the last 20 days, from now
    # importantly, do not look forward, so you have to have -pred_len errors
    if len(errors)>10:
        
        bf = errors[-(20+pred_len):-pred_len] #used error period of 20 for ETHUSD
        bf = np.mean(bf)


# # XGB model creation
    tscv = TimeSeriesSplit(n_splits=3)
    xgb_model = xgb.XGBRegressor(random_state=42) #
    grid_search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_grid,
        cv=tscv,
        n_jobs=-1,
        verbose=0,
        # random_state=42
    )
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

# forecast loop for XGB
    for i in range(pred_len):
        if i == 0:
            prediction = best_model.predict(X_test[0]) # making a one-step prediction
            forecast_X_train = np.append(X_train, X_test[0])
            forecast_y_train = np.append(y_train, prediction)
            
        else:
            grid_search.fit(forecast_X_train.reshape(-1,1), forecast_y_train.reshape(-1,1))
            best_model = grid_search.best_estimator_
            prediction = best_model.predict(prior_prediction)
            forecast_X_train = np.append(forecast_X_train, prior_prediction[0])
            forecast_y_train = np.append(forecast_y_train, prediction[0])
        prior_prediction = prediction
    
    preds = forecast_y_train[-pred_len:]  #- best_model.intercept_
    preds = scaler.inverse_transform(preds.reshape(-1,1))
    cum_preds = preds.cumsum()
    cum_pred = cum_preds[-1]
    
    pred_val = cum_pred + XGB_test_data['close_lag1'].iloc[0]

    # pred_close = np.exp(pred_val)

    init_pred_pct_del = (pred_val - act_enter)/act_enter
    pred_pct_del = init_pred_pct_del + bf
#     # pred_pct_del = (prediction-act_enter)/act_enter #when using non-diff data
    act_pct_del = (act_exit -act_enter)/act_enter
    print("XGB pred pct del: " + str(pred_pct_del))
    print("actual pct del: " + str(act_pct_del))
    print("bias: " + str(bf))
#     #straight XGB BA, not going to be great
    if np.sign(pred_pct_del) == np.sign(act_pct_del):
        XGB_BA_c = XGB_BA_c+1

    # if np.sign(AR_pred_pct_del) == np.sign(act_pct_del):
    #     AR_BA_c = AR_BA_c+1
    #XGB BA with threshold
    if ((pred_pct_del > long_thresh) and (np.sign(pred_pct_del)) == np.sign(act_pct_del))\
        or ((pred_pct_del < short_thresh) and (np.sign(pred_pct_del)) == np.sign(act_pct_del)):
            XGB_thresh_BA_c = XGB_thresh_BA_c + 1

    # if ((AR_pred_pct_del > AR_long_thresh) and (np.sign(AR_pred_pct_del)) == np.sign(act_pct_del))\
    #     or ((AR_pred_pct_del < AR_short_thresh) and (np.sign(AR_pred_pct_del)) == np.sign(act_pct_del)):
    #         AR_thresh_BA_c = AR_thresh_BA_c + 1

    
    pred_BA_c = pred_BA_c + 1
    XGB_thresh_BA = XGB_thresh_BA_c/pred_BA_c

    votes = []
#     side = 'cash'
#     #XGB vote 
    if (pred_pct_del > (long_thresh)):
        # pred_pct_del > (XGB_test_data.iloc[0]['ema1_lag1']/act_enter + long_thresh)):
        xgb_vote = 'long'
        XGB_pred_BA_thresh_c = XGB_pred_BA_thresh_c + 1
    elif (pred_pct_del < (short_thresh)):
        #   pred_pct_del < (XGB_test_data.iloc[0]['ema1_lag1']/act_enter + short_thresh)):
        xgb_vote = 'short'
        XGB_pred_BA_thresh_c = XGB_pred_BA_thresh_c + 1
    else:
        xgb_vote = 'none'
    print("xgb_vote: " + xgb_vote)
    votes.append(xgb_vote)

    # if (AR_pred_pct_del > (AR_long_thresh)):
    #     # pred_pct_del > (XGB_test_data.iloc[0]['ema1_lag1']/act_enter + long_thresh)):
    #     ar_vote = 'long'
    #     AR_pred_BA_thresh_c = AR_pred_BA_thresh_c + 1
    # elif (AR_pred_pct_del < (AR_short_thresh)):
    #     #   pred_pct_del < (XGB_test_data.iloc[0]['ema1_lag1']/act_enter + short_thresh)):
    #     ar_vote = 'short'
    #     AR_pred_BA_thresh_c = AR_pred_BA_thresh_c + 1
    # else:
    #     ar_vote = 'none'
    # print("ar_vote: " + ar_vote)
    # votes.append(ar_vote)
    
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

    X_we_stop_loss = []
    sl_flag = 'none'
    #which one happened first?

    i = 0
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


    posit = 0
    #make trade #stop is always a loss
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

    errors.append(act_pct_del-init_pred_pct_del)

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
        'price_exited': act_exit,
        'signal': side,
        'trade_return': trade_ret
    })

    all_ret_1 = [1+ret for ret in all_ret]
    cum_ret_list = np.cumprod(all_ret_1)
    cum_ret_list = cum_ret_list-1
    cum_ret = cum_ret_list[-1]
    
    XGB_BA = XGB_BA_c/pred_BA_c
    XGB_thresh_BA = XGB_thresh_BA_c/XGB_pred_BA_thresh_c
    # sma_BA = sma_BA_thresh_c/pred_BA_c
    AR_BA = AR_BA_c/pred_BA_c
    AR_thresh_BA = AR_thresh_BA_c/AR_pred_BA_thresh_c


    print(" Cum Return: " + '{0:.2f}'.format(cum_ret*100) + "%. \nThis side: " + side)
    print("offset: " + str(offset))
    print("XGB Model BA: " +'{0:.3f}'.format(XGB_BA))
    print("XGB_thresh_BA: " + '{0:.3f}'.format(XGB_thresh_BA))
    # print("AR Model BA: " +'{0:.3f}'.format(AR_BA))
    # print("AR_thresh_BA: " + '{0:.3f}'.format(AR_thresh_BA))
    print("trade BA: " + '{0:.3f}'.format(trade_BA))
    # print("sma BA: " + '{0:.3f}'.format(sma_BA))
    print("posit: " + '{0:.3f}'.format(posit))

    prev_side = side

    stats_df = pd.DataFrame(stats)

    stats_df.to_csv(output_stats_file, index=False)
print(f"Statistics written to {output_stats_file}")
