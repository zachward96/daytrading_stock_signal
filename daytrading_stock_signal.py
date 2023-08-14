# ******************************* IMPORTS *******************************

import streamlit as st
import numpy as np
import pandas as pd
import math
import scipy as stats
from pandas_datareader import data as pdr
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
import time

import yfinance as yf
yf.pdr_override()

import warnings
warnings.filterwarnings('ignore')


st.title('Day Trading Stock Signal')

# ********************************* FUNCTIONS ******************************

def get_data(ticker, start_date, end_date):
    data = pdr.get_data_yahoo(ticker, start_date, end_date)
    data = data.drop(columns = ['Adj Close'])
    return data

def add_returns_volatility(df):
    #adding monthly returns using 20 day periods as months and the percent change over that period
    df['1_month_returns'] = df['Close'].pct_change(20)
    df['2_month_returns'] = df['Close'].pct_change(40)
    df['3_month_returns'] = df['Close'].pct_change(60)

    #adding volatility using standard deviation
    df['1_month_vol'] = (np.log(df['Close']).diff().rolling(20).std())
    df['2_month_vol'] = (np.log(df['Close']).diff().rolling(40).std())
    df['3_month_vol'] = (np.log(df['Close']).diff().rolling(60).std())

    return df

#this will add emas and apo to our database.
#Most of the ideas in this function come from Learn Algorithmic Trading by Sabestien Donadio
def add_emas(df):
    close = df['Close']
    apo_values = []
    ema_fast_values = []
    ema_slow_values = []
    price_history = []
    ema_fast = 0
    ema_slow = 0

    for close_price in close:
        price_history.append(close_price)
        if len(price_history) > 20:
            del(price_history[0])

        #This idea is from Learn Algorithmic Trading by Sabestien Donadio. It will be used for our volatility measure
        sma = stats.mean(price_history)
        variance = 0
        for hist_price in price_history:
            variance = variance + ((hist_price - sma) ** 2)

        #this idea for a volatility factor comes from Learn Algorithmic Trading by Sabestien Donadio
        stdev = math.sqrt(variance / len(price_history))
        stdev_factor = stdev/15
        if stdev_factor == 0:
            stdev_factor = 1


        if (ema_fast == 0): # first observation
            ema_fast = close_price
            ema_slow = close_price
        else:
            #calculating ema with a smoothing factor and the stdev_factor which is a way to account for volatility
            ema_fast = (close_price - ema_fast) * (2/(10+1)) *stdev_factor + ema_fast
            ema_slow = (close_price - ema_slow) * (2/(40+1)) *stdev_factor + ema_slow

        ema_fast_values.append(ema_fast)
        ema_slow_values.append(ema_slow)

        #calculating apo as the difference between fast and slow emas
        apo = ema_fast - ema_slow
        apo_values.append(apo)

    df = df.assign(fast_ema = pd.Series(ema_fast_values, index=df.index))
    df = df.assign(slow_ema = pd.Series(ema_slow_values, index=df.index))
    df = df.assign(APO = pd.Series(apo_values, index=df.index))

    return df


#this function adds labels to the data
def add_labels(df):
    day_change = df['Close'] - df['Open']
    percent_day_change = (df['Close'] - df['Open']) / df['Open']
    labels = []

    for change in percent_day_change:
        #stock is a buy if it closes higher than it opens
        if change > 0.01:
            labels.append(1)
        #a sell if the stock closes lower than it opens
        elif change < -0.01:
            labels.append(-1)
        #a hold if the stock doesn't move by at least 1%. This will hopefully minimize losses from missed predictions
        else:
            labels.append(0)

    df['day_change'] = day_change
    df['percent_day_change'] = percent_day_change
    df['signal'] = labels

    return df

#this function combines all of our previous functions to output a single dataframe with all of our engineered features
def clean_data(user_input_symbol, user_input_start_date, user_input_end_date):
    data = get_data(user_input_symbol, user_input_start_date, user_input_end_date)
    data = add_returns_volatility(data)
    data = add_emas(data)
    data = add_labels(data)
    data = data.fillna(0)

    return data

# ****************************** TRAIN/TEST SPLITTING AND SCALING ************************

@st.cache_data
def load_data(user_input_symbol, user_input_start_date, user_input_end_date):
	data = clean_data(user_input_symbol, user_input_start_date, user_input_end_date)
	return data

# ****************************** STREAMLIT FOUNDATION ************************************

tab1, tab2, tab3, tab4 = st.tabs(["Welcome!", "How Do I Use This?", "Prediction System", "Under The Hood"])

#app_mode = st.sidebar.selectbox(label = 'Navigation', options = ['Resume', 'Instructions', 
#'Data Viewing', 'Prediction'])

with tab1:
	st.header('Welcome!')
	st.markdown('Welcome to my day trading stock signal system.')
	st.markdown('My name is Zach Ward and, first and foremost, I am not a registered financial advisor in any way.') 
	st.markdown('Any decisions that you make regarding your money are your own and you are responsible for them.') 
	st.markdown("Don't be dumb.")
	st.markdown('')
	st.markdown('')
	st.markdown('The idea behind this system is to leverage some of the statistical tools used by major trading firms for the individual investor')
	st.markdown("If you are interested in what exactly this system is doing head on over to the 'Under the Hood' tab")
	st.markdown('I owe specific thanks to Kaggle user Devpark0506 and Sebestian Donadio as their work was instrumental in the creation of this system')
	st.markdown('')
	st.markdown('')
	st.markdown('If you have any questions or suggestions feel free to contact me at zach.ward@me.com. Otherwise dive in')

with tab2:
	st.header("How to use this app")
	st.markdown('To begin enter the desired stock ticker symbol in the box on the predictions page')
	st.text("")
	st.markdown('If an invalid ticker symbol is entered then you will receive an error')
	st.text("")
	st.markdown("If it is still in the middle of the trading day, enter today's date in the today's date box. If the trading day has already closed then enter tomorrow's date")
	st.text("")
	st.markdown("You will receive predictions for the date entered. If the trading day has closed an you enter today's date then you will receive outdated predictions. This can be used to test the system against what actually happened.")

with tab3:
	user_input_symbol = st.text_input('Enter Ticker Symbol Here')
	user_input_end_date = st.text_input("Today's Date yyyy-mm-dd")
	user_input_start_date = st.text_input("10 years prior to today's date yyyy-mm-dd")
	neural_net = st.checkbox("Neural Net")
	random_forest = st.checkbox("Random Forest")
	if neural_net:
		if st.button('Generate Neural Net Predictions (This could take up to 5 minutes)'):
			data = load_data(user_input_symbol, user_input_start_date, user_input_end_date)

			open_labels = data['Open'].shift(-1).fillna(data['Open'].iloc[-1])
			open_features = data.drop(['Open'], axis = 1)

			close_labels = data['Close'].shift(-1).fillna(data['Close'].iloc[-1])
			close_features = data.drop(['Close'], axis = 1)

			x_train_open, x_test_open, y_train_open, y_test_open = train_test_split(open_features, open_labels, test_size = 0.2)
			x_train_close, x_test_close, y_train_close, y_test_close = train_test_split(close_features,close_labels, test_size = 0.2)

			x_train_open_nn, x_valid_open, y_train_open_nn, y_valid_open = train_test_split(x_train_open, y_train_open)
			x_train_close_nn, x_valid_close, y_train_close_nn, y_valid_close = train_test_split(x_train_close, y_train_close)

			scaler = StandardScaler()

			x_train_open_nn = scaler.fit_transform(x_train_open_nn)
			x_valid_open = scaler.fit_transform(x_valid_open)

			x_train_close_nn = scaler.fit_transform(x_train_close_nn)
			x_valid_close = scaler.fit_transform(x_valid_close)

			scaled_open_features = scaler.fit_transform(open_features)
			scaled_close_features = scaler.fit_transform(close_features)

			scaled_x_test_open = scaler.fit_transform(x_test_open)
			scaled_x_test_close = scaler.fit_transform(x_test_close)

			open_model_deep_normalized = keras.models.Sequential([
    				keras.layers.Dense(1509, activation = 'relu'),
    				keras.layers.Dense(755, activation = 'relu'),
   				keras.layers.Dense(378, activation = 'relu'),
    				keras.layers.Dense(188, activation = 'relu'),
    				keras.layers.Dense(95, activation = 'relu'),
    				keras.layers.Dense(45, activation = 'relu'),
    				keras.layers.Dense(24, activation = 'relu'),
    				keras.layers.BatchNormalization(),
    				keras.layers.Dense(1)
			])

			early_stopping = keras.callbacks.EarlyStopping(patience = 10, restore_best_weights = True)
			optimizer = keras.optimizers.legacy.Adam(learning_rate = 0.01)

			open_model_deep_normalized.compile(loss = 'huber', 
						     optimizer = optimizer, 
						     metrics=['mean_squared_error'])

			open_model_deep_norm = open_model_deep_normalized.fit(x_train_open_nn, y_train_open_nn, epochs=200, 
									  validation_data = (x_valid_open, y_valid_open), 
									  callbacks = [early_stopping])

			deep_norm_error = open_model_deep_normalized.evaluate(scaled_x_test_open, y_test_open)
			rmse_open_nn = np.sqrt(deep_norm_error)
			rmse_open_nn = rmse_open_nn[1]
			tomorrows_open_pred_nn = open_model_deep_normalized.predict(scaled_open_features)
			open_pred_nn = tomorrows_open_pred_nn[-1]

			input = keras.layers.Input(shape = x_train_close_nn.shape[1:])
			hidden1 = keras.layers.Dense(1509, activation = 'relu')(input)
			hidden2 = keras.layers.Dense(755, activation = 'relu')(hidden1)
			hidden3 = keras.layers.Dense(378, activation = 'relu')(hidden2)
			hidden4 = keras.layers.Dense(188, activation = 'relu')(hidden3)
			hidden5 = keras.layers.Dense(95, activation = 'relu')(hidden4)
			hidden6 = keras.layers.Dense(45, activation = 'relu')(hidden5)
			hidden7 = keras.layers.Dense(24, activation = 'relu')(hidden6)
			concat = keras.layers.concatenate([input, hidden7])
			output = keras.layers.Dense(1)(concat)
			close_model_deep_wide = keras.models.Model(inputs=[input], outputs = [output])

			optimizer = keras.optimizers.legacy.Adam(learning_rate = 0.01)

			close_model_deep_wide.compile(loss = 'huber',
						      optimizer = optimizer, 
						      metrics=['mean_squared_error'])

			close_model_deep_wide_1 = close_model_deep_wide.fit(x_train_close_nn, y_train_close_nn, epochs=200, 
									    validation_data = (x_valid_open, y_valid_open), 
									    callbacks = [early_stopping])

			close_error_deep_wide = close_model_deep_wide.evaluate(scaled_x_test_close, y_test_close)
			rmse_close_nn = np.sqrt(close_error_deep_wide)
			rmse_close_nn = rmse_close_nn[1]
			tomorrows_close_pred_nn = close_model_deep_wide.predict(scaled_close_features)
			close_pred_nn = tomorrows_close_pred_nn[-1]

			if close_pred_nn > open_pred_nn:
  				st.write('BUY, the stock is predicted to increase by', close_pred_nn - open_pred_nn, 'tomorrow, NN')
				st.write('WARNING: this prediction could be off by as much as +/-', round(rmse_open_nn + rmse_close_nn, 3))
			else:
  				st.write('SELL, the stock is predicted to decrease by', close_pred_nn - open_pred_nn, 'tomorrow, NN')
				st.write('WARNING: this prediction could be off by as much as +/-', round(rmse_open_nn + rmse_close_nn, 3))

	if random_forest:
		if st.button('Generate Random Forest Predictions'):
			data = load_data(user_input_symbol, user_input_start_date, user_input_end_date)

			open_labels = data['Open'].shift(-1).fillna(data['Open'].iloc[-1])
			open_features = data.drop(['Open'], axis = 1)

			close_labels = data['Close'].shift(-1).fillna(data['Close'].iloc[-1])
			close_features = data.drop(['Close'], axis = 1)

			x_train_open, x_test_open, y_train_open, y_test_open = train_test_split(open_features, open_labels, test_size = 0.2)
			x_train_close, x_test_close, y_train_close, y_test_close = train_test_split(close_features,close_labels, test_size = 0.2)

			rnd_for_open = RandomForestRegressor()
			rnd_for_open.fit(x_train_open, y_train_open)
			pred_rf_open = rnd_for_open.predict(x_test_open)

			RMSE_open = np.sqrt(mean_squared_error(y_test_open, pred_rf_open))
			tomorrows_open_pred_rf = rnd_for_open.predict(open_features)
			open_pred_rf = tomorrows_open_pred_rf[-1]

			rnd_for_close = RandomForestRegressor()
			rnd_for_close.fit(x_train_close, y_train_close)
			pred_rf_close = rnd_for_close.predict(x_test_close)

			RMSE_close = np.sqrt(mean_squared_error(y_test_close, pred_rf_close))
			tomorrows_close_pred_rf = rnd_for_close.predict(close_features)
			close_pred_rf = tomorrows_close_pred_rf[-1]

			if close_pred_rf > open_pred_rf:
  				st.write('BUY: the stock is predicted to increase by', round(close_pred_rf - open_pred_rf, 3), 'tomorrow')
  				st.write('WARNING: this prediction could be off by as much as +/-', round(RMSE_open + RMSE_close, 3))
			else:
  				st.write('SELL: the stock is predicted to decrease by', round(close_pred_rf - open_pred_rf, 3), 'tomorrow')
  				st.write('WARNING: this prediction could be off by as much as +/-', round(RMSE_open + RMSE_close, 3))


with tab4:
	st.header('How does this system work?')
	st.subheader('Basic Structure')
	st.markdown('This system leverages statistical tools like emas, rolling averages, volatility calculations, and apos and then feeds these features into two different machine learning algorithms.')
	st.markdown('These statistical tools are frequently used by major trading firms, but not by the average investor.')
	st.markdown('The main reason for this is that they take a lot of time to calculate unless you are using a computer system (who wants to calculate rolling averages in 20, 40, and 60 day periods by hand right?)')
	st.markdown("You might be wondering though, why I don't show you these features.")
	st.markdown('The answer is that these features are not overly helpful in and of themselves. The 40 day volatility of a stock is not going to be the only decision point on wether or not to buy it.')
	st.markdown('Rather, these features excel as factors in the decision making process. Then we use machine learning algorithms (in this case Neural Networks and Random Forests) to make that decision')
	st.subheader('Neural Networks')
	st.markdown("I won't go to far in depth on how neural networks operate here, but think of them as your personal pocket savant who can memorize the whole dataset and extract what you want out of it.")
	st.markdown('In theory this would make neural networks incredibly effective at stock prediction. However, in this particular system, we see a higher error rate than in the random forest system')
	st.subheader('Random Forest')
	st.markdown('The best way to conceptualize a random forest is visually. Think of it as a big system of decision trees (if feature A is yes you go right, if no you go left)')
	st.markdown('This ultimately will create a numeric prediction for the stock')
	st.subheader('How do we actually get the prediction?')
	st.markdown('The idea of using machine learning for stock prediction is not a new one.')
	st.markdown('This system is mainly focused at trying to solve the issue of time horizon.')
	st.markdown('Most stock prediction systems currently available to the average individual inform you if a stock will go up or down, but not how long it will take for it to do so.')
	st.markdown('This means that you could turn a profit in a day, or it could take six months.')
	st.markdown('In an effort to eliminate this issue, this system produces two different predictions. One for the Open price of the next daay, and one for the close price.')
	st.markdown('The positive side of this is that it very clearly defines our time horizon at one trading day.')
	st.markdown('Unfortunately it carries with it the downside of effectively doubling our error rate since there are two predictions instead of one.')
	st.markdown('However, I believe that the benefit of a defined time horizon outweights the downside of increased error rate, especially when the system is utilized by an aware investor.')

#data_load_state = st.text('Data is loading')
#	st.dataframe(data)
#	data_load_state.text('Loading complete')

