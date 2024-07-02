import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import nltk

nltk.download('vader_lexicon')

fileDB = 'C:/Users/Nont/Desktop/23T2/FINS3645/Assignment/'
df = pd.read_json(fileDB + 'commodity_News.json')
df.columns = ['Date', 'Source', 'Headline']

# Instantiate the sentiment intensity analyzer
vader = SentimentIntensityAnalyzer()

# Iterate through the headlines and get the polarity scores using vader
scores = df['Headline'].apply(vader.polarity_scores).tolist()
# Convert the 'scores' list of dicts into a DataFrame
scores_df = pd.DataFrame(scores)

# Join the DataFrames of the news and the list of dicts
df = df.join(scores_df, rsuffix='_right')
df.to_json(fileDB + 'vader_scores.json')

# Group by date and ticker columns from scored_news and calculate the mean
date_format = '%A, %B %d, %Y %I:%M:%S %p (GMT)'
df.index = pd.to_datetime(df['Date'], format=date_format)
df = df.drop(columns=['Date'])
mean_scores = df.groupby(['Date']).mean()

# Fit ARIMA model (change the order depending on your data)
model = ARIMA(mean_scores['compound'], order=(7, 1, 0))
model_fit = model.fit()
forecast = model_fit.forecast(steps=7)

last_date = mean_scores.index[-1]
forecast_index = pd.date_range(start=last_date + pd.DateOffset(days=1), periods=7)
forecast_series = pd.Series(forecast.values, index=forecast_index)
print('Forecast: \n', forecast_series)

# Boxplot of time-series of sentiment score for each date
# mean_scores.plot(kind = 'box')
# plt.grid()
# plt.title('Box plot of sentiment score overtime')
# plt.show()
#
# # get the lastest month sentiment score as adjustment
# lastest_month_score = mean_scores.tail(30).mean()
# lastest_month_score.fillna(0,inplace=True)
# lastest_month_score.plot(kind='bar')
# plt.title('Sentiment Score')
# plt.grid()
# plt.show()
#
# # adjusted weights - based on relative strength of sentiment score
# average_score = lastest_month_score.mean()
# lastest_month_score = (lastest_month_score -average_score)/len(lastest_month_score)
# lastest_month_score.plot(kind='bar')
# plt.title('Adjustment on Portfolio Weights -- Sentiment Score')
# plt.grid()
# print(lastest_month_score)
# plt.show()
