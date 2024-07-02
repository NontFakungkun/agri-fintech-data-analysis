import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')


def Station1_loadDataExcel(file):
    """
    Process data load post from Excel extract and transfer phases
    :return: df cleansed dataset
    """
    df = pd.read_excel(file)
    return df


def Station1_loadDataJSON(file):
    """
    Process data load post from JSON extract and transfer phases
    :return: df cleansed dataset
    """
    df = pd.read_json(file)
    return df


def Station2_features_Price(crop, df):
    """
    Receive cleaned data from Station #1 process all relevant features
    :param df: input clean data streams
    :return: rets: return dictionary containing relevant price features

    Corn and Wheat prices
    Price distribution - relative price
    Seasonal Price, mean, median, mode
    Seasonal Change, mean, median, mode
    Price correlation of price at 2 different years
    """

    df['Date'] = pd.to_datetime(df['Date'])
    jan_feb_mar_data = df[(df['Date'].dt.month == 1) | (df['Date'].dt.month == 2) | (df['Date'].dt.month == 3)]
    apr_may_jun_data = df[(df['Date'].dt.month == 4) | (df['Date'].dt.month == 5) | (df['Date'].dt.month == 6)]
    jul_aug_sep_data = df[(df['Date'].dt.month == 7) | (df['Date'].dt.month == 8) | (df['Date'].dt.month == 9)]
    oct_nov_dec_data = df[(df['Date'].dt.month == 10) | (df['Date'].dt.month == 11) | (df['Date'].dt.month == 12)]

    fig, axs = plt.subplots(2, 3, figsize=(15, 10), sharey=True)  # Adjust figure size if needed

    df['price_int'] = (df['Settlement Price'] * 100).astype(int)
    axs[0, 0].hist(df['price_int'], bins=50)
    axs[0, 0].set_xlabel('Values')
    axs[0, 0].set_ylabel('Frequency')
    axs[0, 0].set_title(f'Distribution of {crop} settlement price for all season')
    df['price_int'] = (apr_may_jun_data['Settlement Price'] * 100).astype(int)
    axs[0, 1].hist(df['price_int'], bins=40)
    axs[0, 1].set_xlabel('Values')
    axs[0, 1].set_ylabel('Frequency')
    axs[0, 1].set_title(f'Distribution of {crop} settlement price for spring')
    df['price_int'] = (jul_aug_sep_data['Settlement Price'] * 100).astype(int)
    axs[0, 2].hist(df['price_int'], bins=40)
    axs[0, 2].set_xlabel('Values')
    axs[0, 2].set_ylabel('Frequency')
    axs[0, 2].set_title(f'Distribution of {crop} settlement price for summer')
    df['price_int'] = (oct_nov_dec_data['Settlement Price'] * 100).astype(int)
    axs[1, 0].hist(df['price_int'], bins=40)
    axs[1, 0].set_xlabel('Values')
    axs[1, 0].set_ylabel('Frequency')
    axs[1, 0].set_title(f'Distribution of {crop} settlement price for fall')
    df['price_int'] = (jan_feb_mar_data['Settlement Price'] * 100).astype(int)
    axs[1, 1].hist(df['price_int'], bins=40)
    axs[1, 1].set_xlabel('Values')
    axs[1, 1].set_ylabel('Frequency')
    axs[1, 1].set_title(f'Distribution of {crop} settlement price for winter')
    plt.tight_layout()  # Adjust subplot spacing for better visualization
    plt.show()

    total_dict = {
        'mean_price': round(df['Settlement Price'].mean(), 4),
        'median_price': df['Settlement Price'].median(),
        'mode_price': df['Settlement Price'].mode().values[0]
    }

    winter_data_dict = {
        'mean_price': round(jan_feb_mar_data['Settlement Price'].mean(), 4),
        'median_price': jan_feb_mar_data['Settlement Price'].median(),
        'mode_price': jan_feb_mar_data['Settlement Price'].mode().values[0]
    }

    spring_data_dict = {
        'mean_price': round(apr_may_jun_data['Settlement Price'].mean(), 4),
        'median_price': apr_may_jun_data['Settlement Price'].median(),
        'mode_price': apr_may_jun_data['Settlement Price'].mode().values[0]
    }

    summer_data_dict = {
        'mean_price': round(jul_aug_sep_data['Settlement Price'].mean(), 4),
        'median_price': jul_aug_sep_data['Settlement Price'].median(),
        'mode_price': jul_aug_sep_data['Settlement Price'].mode().values[0]
    }

    fall_data_dict = {
        'mean_price': round(oct_nov_dec_data['Settlement Price'].mean(), 4),
        'median_price': oct_nov_dec_data['Settlement Price'].median(),
        'mode_price': oct_nov_dec_data['Settlement Price'].mode().values[0]
    }

    rets_price = {
        'price_total': total_dict,
        'price_spring': spring_data_dict,
        'price_summer': summer_data_dict,
        'price_fall': fall_data_dict,
        'price_winter': winter_data_dict,
    }

    total_dict = {
        'mean_price_change': round(df['% Change'].mean(), 2),
        'median_price_change': round(df['% Change'].median(), 2),
        'mode_price_change': round(df['% Change'].mode().values[0], 2),
    }

    winter_data_dict = {
        'mean_price_change': round(jan_feb_mar_data['% Change'].mean(), 2),
        'median_price_change': round(jan_feb_mar_data['% Change'].median(), 2),
        'mode_price_change': round(jan_feb_mar_data['% Change'].mode().values[0], 2),
    }

    spring_data_dict = {
        'mean_price_change': round(apr_may_jun_data['% Change'].mean(), 2),
        'median_price_change': round(apr_may_jun_data['% Change'].median(), 2),
        'mode_price_change': round(apr_may_jun_data['% Change'].mode().values[0], 2),
    }

    summer_data_dict = {
        'mean_price_change': round(jul_aug_sep_data['% Change'].mean(), 2),
        'median_price_change': round(jul_aug_sep_data['% Change'].median(), 2),
        'mode_price_change': round(jul_aug_sep_data['% Change'].mode().values[0], 2),
    }

    fall_data_dict = {
        'mean_price_change': round(oct_nov_dec_data['% Change'].mean(), 2),
        'median_price_change': round(oct_nov_dec_data['% Change'].median(), 2),
        'mode_price_change': round(oct_nov_dec_data['% Change'].mode().values[0], 2),
    }

    rets_change = {
        'price_change_total': total_dict,
        'price_change_spring': spring_data_dict,
        'price_change_summer': summer_data_dict,
        'price_change_fall': fall_data_dict,
        'price_change_winter': winter_data_dict,
    }

    return rets_price, rets_change


def Station2_features_Corr(df1, df2):
    price_corr = df1['Settlement Price'].corr(df2['Settlement Price'])
    change_corr = df1['% Change'].corr(df2['% Change'])
    vol_corr = df1['CVol'].corr(df2['CVol'])
    rets = {
        'price_correlation': price_corr,
        'price_change_correlation': change_corr,
        'volume_correlation': vol_corr,
    }
    return rets


def Station2_features_Volume(df):
    """
    Receive cleaned data from Station #1 process all relevant features
    :param df: input clean data streams
    :return: rets: return dictionary containing relevant volume features

    Corn and wheat volume - Seasonal volume, suitable for crops
    the season will be roughly estimate by months
    Winter: Jan Feb March
    Spring: April May June
    Summer: July August September
    Fall: October November December
    """
    df['Date'] = pd.to_datetime(df['Date'])
    jan_feb_mar_data = df[(df['Date'].dt.month == 1) | (df['Date'].dt.month == 2) | (df['Date'].dt.month == 3)]
    apr_may_jun_data = df[(df['Date'].dt.month == 4) | (df['Date'].dt.month == 5) | (df['Date'].dt.month == 6)]
    jul_aug_sep_data = df[(df['Date'].dt.month == 7) | (df['Date'].dt.month == 8) | (df['Date'].dt.month == 9)]
    oct_nov_dec_data = df[(df['Date'].dt.month == 10) | (df['Date'].dt.month == 11) | (df['Date'].dt.month == 12)]

    total_dict = {
        'mean_volume': round(df['CVol'].mean(), 2),
        'median_volume': df['CVol'].median(),
        'mode_volume': df['CVol'].mode().values[0]
    }

    winter_data_dict = {
        'mean_volume': round(jan_feb_mar_data['CVol'].mean(), 2),
        'median_volume': jan_feb_mar_data['CVol'].median(),
        'mode_volume': jan_feb_mar_data['CVol'].mode().values[0]
    }

    spring_data_dict = {
        'mean_volume': round(apr_may_jun_data['CVol'].mean(), 2),
        'median_volume': apr_may_jun_data['CVol'].median(),
        'mode_volume': apr_may_jun_data['CVol'].mode().values[0]
    }

    summer_data_dict = {
        'mean_volume': round(jul_aug_sep_data['CVol'].mean(), 2),
        'median_volume': jul_aug_sep_data['CVol'].median(),
        'mode_volume': jul_aug_sep_data['CVol'].mode().values[0]
    }

    fall_data_dict = {
        'mean_volume': round(oct_nov_dec_data['CVol'].mean(), 2),
        'median_volume': oct_nov_dec_data['CVol'].median(),
        'mode_volume': oct_nov_dec_data['CVol'].mode().values[0]
    }

    rets = {
        'total': total_dict,
        'spring': spring_data_dict,
        'summer': summer_data_dict,
        'fall': fall_data_dict,
        'winter': winter_data_dict,
    }

    return rets


def Station2_relevant_news(news_df, text):
    df = pd.DataFrame()
    selected_rows = news_df[news_df['Headline'].str.contains(text, case=False)]
    df = df.append(selected_rows, ignore_index=True)
    return df


def Station3_ARIMAForecast(df, p, d, q, st, display_range):
    model = ARIMA(df, order=(p, d, q))
    forecast_arima = model.fit().forecast(steps=st).round(4)
    last_date = df.index[-1]
    forecast_index = pd.date_range(start=last_date + pd.DateOffset(), periods=st)
    forecast_series = pd.Series(forecast_arima.values, index=forecast_index)
    forecast_arima = pd.concat([df[-display_range:], forecast_series])
    return forecast_arima


def Station3_VADER(df):
    # Instantiate the sentiment intensity analyzer
    vader = SentimentIntensityAnalyzer()

    # Iterate through the headlines and get the polarity scores using vader
    scores = df['Headline'].apply(vader.polarity_scores).tolist()
    # Convert the 'scores' list of dicts into a DataFrame
    scores_df = pd.DataFrame(scores)

    # Join the DataFrames of the news and the list of dicts
    df = df.join(scores_df, rsuffix='_right')

    # Group by date and ticker columns from scored_news and calculate the mean
    date_format = '%A, %B %d, %Y %I:%M:%S %p (GMT)'
    df.index = pd.to_datetime(df['Date'], format=date_format)
    df = df.drop(columns=['Date'])
    mean_scores = df.groupby(['Date']).mean()

    return mean_scores


def Station4_ReturnForecast(current_price, fc1_price, fc2_price, fc3_price, fc4_price, topic, fc_period_list):
    fc1_ret = (fc1_price - current_price) / current_price * 100
    fc2_ret = (fc2_price - current_price) / current_price * 100
    fc3_ret = (fc3_price - current_price) / current_price * 100
    fc4_ret = (fc4_price - current_price) / current_price * 100
    print(f"{topic} {fc_period_list[0]} return: {fc1_ret.round(2)}%")
    print(f"{topic} {fc_period_list[1]} return: {fc2_ret.round(2)}%")
    print(f"{topic} {fc_period_list[2]} return: {fc3_ret.round(2)}%")
    print(f"{topic} {fc_period_list[3]} return: {fc4_ret.round(2)}%")


def Station4_MeanSentiment(compound_data, topic, average_period_list, display_period_list):
    mean_value1 = compound_data[-average_period_list[0]:].mean()
    result1 = 'Neutral'
    if mean_value1 > 0:
        result1 = 'Positive'
    elif mean_value1 < 0:
        result1 = 'Negative'
    mean_value2 = compound_data[-average_period_list[1]:].mean()
    result2 = 'Neutral'
    if mean_value2 > 0:
        result2 = 'Positive'
    elif mean_value2 < 0:
        result2 = 'Negative'
    mean_value3 = compound_data[-average_period_list[2]:].mean()
    result3 = 'Neutral'
    if mean_value3 > 0:
        result3 = 'Positive'
    elif mean_value3 < 0:
        result3 = 'Negative'
    mean_value4 = compound_data[-average_period_list[3]:].mean()
    result4 = 'Neutral'
    if mean_value4 > 0:
        result4 = 'Positive'
    elif mean_value4 < 0:
        result4 = 'Negative'
    mean_value5 = compound_data[:].mean()
    result5 = 'Neutral'
    if mean_value5 > 0:
        result5 = 'Positive'
    elif mean_value5 < 0:
        result5 = 'Negative'

    print(f"{topic} for the {display_period_list[0]} is {result1} ({mean_value1.round(4)})")
    print(f"{topic} for the {display_period_list[1]} is {result2} ({mean_value2.round(4)})")
    print(f"{topic} for the {display_period_list[2]} is {result3} ({mean_value3.round(4)})")
    print(f"{topic} for the {display_period_list[3]} is {result4} ({mean_value4.round(4)})")
    print(f"{topic} for the {display_period_list[4]} is {result4} ({mean_value5.round(4)})")


def Station4_ARIMALinePlot(fc1, fc2, fc3, fc4, display_list, forecast_topic, forecast_period_list):
    fig, axs = plt.subplots(2, 2, figsize=(15, 10), sharey=True)
    axs[0, 0].plot(fc1.index[:display_list[0]], fc1.values[:display_list[0]],
                   color='midnightblue')
    axs[0, 0].plot(fc1.index[display_list[0]:], fc1.values[display_list[0]:], color='red')
    axs[0, 0].set_xlabel('Date')
    axs[0, 0].set_ylabel('Price')
    axs[0, 0].set_title(f'{forecast_topic} Forecast for {forecast_period_list[0]}')
    axs[0, 1].plot(fc2.index[:display_list[1]], fc2.values[:display_list[1]],
                   color='midnightblue')
    axs[0, 1].plot(fc2.index[display_list[1]:], fc2.values[display_list[1]:],
                   color='red')
    axs[0, 1].set_xlabel('Date')
    axs[0, 1].set_ylabel('Price')
    axs[0, 1].set_title(f'{forecast_topic} Forecast for {forecast_period_list[1]}')
    axs[1, 0].plot(fc3.index[:display_list[2]], fc3.values[:display_list[2]],
                   color='midnightblue')
    axs[1, 0].plot(fc3.index[display_list[2]:], fc3.values[display_list[2]:],
                   color='red')
    axs[1, 0].set_xlabel('Date')
    axs[1, 0].set_ylabel('Price')
    axs[1, 0].set_title(f'{forecast_topic} Forecast for {forecast_period_list[2]}')
    axs[1, 1].plot(fc4.index[:display_list[3]], fc4.values[:display_list[3]],
                   color='midnightblue')
    axs[1, 1].plot(fc4.index[display_list[3]:], fc4.values[display_list[3]:],
                   color='red')
    axs[1, 1].set_xlabel('Date')
    axs[1, 1].set_ylabel('Price')
    axs[1, 1].set_title(f'{forecast_topic} Forecast for {forecast_period_list[3]}')
    plt.tight_layout()
    plt.show()


def Station4_ARIMABarPlot(fc1, fc2, fc3, fc4, display_list, forecast_topic, forecast_period_list):
    fig, axs = plt.subplots(2, 2, figsize=(15, 10), sharey=True)
    axs[0, 0].bar(fc1.index[:display_list[0]], fc1.values[:display_list[0]],
                  color='midnightblue')
    axs[0, 0].bar(fc1.index[display_list[0]:], fc1.values[display_list[0]:], color='red')
    axs[0, 0].set_xlabel('Date')
    axs[0, 0].set_ylabel('Price')
    axs[0, 0].set_title(f'{forecast_topic} Forecast for {forecast_period_list[0]}')
    axs[0, 1].bar(fc2.index[:display_list[1]], fc2.values[:display_list[1]],
                  color='midnightblue')
    axs[0, 1].bar(fc2.index[display_list[1]:], fc2.values[display_list[1]:],
                  color='red')
    axs[0, 1].set_xlabel('Date')
    axs[0, 1].set_ylabel('Price')
    axs[0, 1].set_title(f'{forecast_topic} Forecast for {forecast_period_list[1]}')
    axs[1, 0].bar(fc3.index[:display_list[2]], fc3.values[:display_list[2]],
                  color='midnightblue')
    axs[1, 0].bar(fc3.index[display_list[2]:], fc3.values[display_list[2]:],
                  color='red')
    axs[1, 0].set_xlabel('Date')
    axs[1, 0].set_ylabel('Price')
    axs[1, 0].set_title(f'{forecast_topic} Forecast for {forecast_period_list[2]}')
    axs[1, 1].bar(fc4.index[:display_list[3]], fc4.values[:display_list[3]],
                  color='midnightblue')
    axs[1, 1].bar(fc4.index[display_list[3]:], fc4.values[display_list[3]:],
                  color='red')
    axs[1, 1].set_xlabel('Date')
    axs[1, 1].set_ylabel('Price')
    axs[1, 1].set_title(f'{forecast_topic} Forecast for {forecast_period_list[3]}')
    plt.tight_layout()
    plt.show()


def main():
    # Station #1
    weather_df = Station1_loadDataExcel('C:/Users/Nont/Desktop/23T2/FINS3645/Assignment/Weather.xlsx')
    corn_price_df = Station1_loadDataExcel(
        'C:/Users/Nont/Desktop/23T2/FINS3645/Assignment/Processed_Corn_PriceHistory.xlsx')
    wheat_price_df = Station1_loadDataExcel(
        'C:/Users/Nont/Desktop/23T2/FINS3645/Assignment/Processed_Wheat_PriceHistory.xlsx')
    client_cash_df = Station1_loadDataExcel('C:/Users/Nont/Desktop/23T2/FINS3645/Assignment/Client_Cash_Accounts.xlsx')
    news_df = Station1_loadDataJSON('C:/Users/Nont/Desktop/23T2/FINS3645/Assignment/commodity_news.json')
    news_df.columns = ['Date', 'Source', 'Headline']

    # Station #2
    corn_price_feat, corn_change_feat = Station2_features_Price('corn', corn_price_df)
    wheat_price_feat, wheat_change_feat = Station2_features_Price('wheat', wheat_price_df)
    corn_volume_feat = Station2_features_Volume(corn_price_df)
    wheat_volume_feat = Station2_features_Volume(wheat_price_df)
    correlation = Station2_features_Corr(corn_price_df, wheat_price_df)

    # Separate Corn and Wheat news
    corn_news_df = Station2_relevant_news(news_df, 'corn')
    wheat_news_df = Station2_relevant_news(news_df, 'wheat')

    print(corn_price_feat)
    print(wheat_price_feat)
    print(corn_change_feat)
    print(wheat_change_feat)
    print(correlation)
    print(corn_volume_feat)
    print(wheat_volume_feat)

    # Station #3
    # Set indexes into dates
    corn_price_df.index = pd.to_datetime(corn_price_df['Date'])
    corn_price_df = corn_price_df.drop(columns=['Date'])
    wheat_price_df.index = pd.to_datetime(wheat_price_df['Date'])
    wheat_price_df = wheat_price_df.drop(columns=['Date'])

    # ARIMA Price prediction
    corn_price_arima_forecast_7days = Station3_ARIMAForecast(corn_price_df['Settlement Price'], 7, 1, 30, 7, 30)
    corn_price_arima_forecast_14days = Station3_ARIMAForecast(corn_price_df['Settlement Price'], 14, 1, 30, 14, 60)
    corn_price_arima_forecast_1month = Station3_ARIMAForecast(corn_price_df['Settlement Price'], 30, 1, 30, 30, 90)
    corn_price_arima_forecast_3months = Station3_ARIMAForecast(corn_price_df['Settlement Price'], 90, 1, 30, 90, 180)
    wheat_price_arima_forecast_7days = Station3_ARIMAForecast(wheat_price_df['Settlement Price'], 7, 1, 30, 7, 30)
    wheat_price_arima_forecast_14days = Station3_ARIMAForecast(wheat_price_df['Settlement Price'], 14, 1, 30, 14, 60)
    wheat_price_arima_forecast_1month = Station3_ARIMAForecast(wheat_price_df['Settlement Price'], 30, 1, 30, 30, 90)
    wheat_price_arima_forecast_3months = Station3_ARIMAForecast(wheat_price_df['Settlement Price'], 90, 1, 30, 90, 180)

    # ARIMA Trading Volume prediction
    corn_volume_arima_forecast_7days = Station3_ARIMAForecast(corn_price_df['CVol'], 7, 1, 30, 7, 30)
    corn_volume_arima_forecast_14days = Station3_ARIMAForecast(corn_price_df['CVol'], 14, 1, 30, 14, 60)
    corn_volume_arima_forecast_1month = Station3_ARIMAForecast(corn_price_df['CVol'], 30, 1, 30, 30, 90)
    corn_volume_arima_forecast_3months = Station3_ARIMAForecast(corn_price_df['CVol'], 90, 1, 30, 90, 180)
    wheat_volume_arima_forecast_7days = Station3_ARIMAForecast(wheat_price_df['CVol'], 7, 1, 30, 7, 30)
    wheat_volume_arima_forecast_14days = Station3_ARIMAForecast(wheat_price_df['CVol'], 14, 1, 30, 14, 60)
    wheat_volume_arima_forecast_1month = Station3_ARIMAForecast(wheat_price_df['CVol'], 30, 1, 30, 30, 90)
    wheat_volume_arima_forecast_3months = Station3_ARIMAForecast(wheat_price_df['CVol'], 90, 1, 30, 90, 180)

    # VADER Sentiment analysis
    corn_mean_sentiment_score = Station3_VADER(corn_news_df)
    wheat_mean_sentiment_score = Station3_VADER(wheat_news_df)

    # Sentiment Prediction
    corn_sentiment_forecast_3days = Station3_ARIMAForecast(corn_mean_sentiment_score['compound'], 3, 1, 30, 3, 14)
    corn_sentiment_forecast_7days = Station3_ARIMAForecast(corn_mean_sentiment_score['compound'], 7, 1, 30, 7, 14)
    corn_sentiment_forecast_14days = Station3_ARIMAForecast(corn_mean_sentiment_score['compound'], 14, 1, 30, 14, 30)
    corn_sentiment_forecast_1month = Station3_ARIMAForecast(corn_mean_sentiment_score['compound'], 30, 1, 30, 30,
                                                            len(corn_mean_sentiment_score))
    wheat_sentiment_forecast_3days = Station3_ARIMAForecast(wheat_mean_sentiment_score['compound'], 3, 1, 30, 3, 14)
    wheat_sentiment_forecast_7days = Station3_ARIMAForecast(wheat_mean_sentiment_score['compound'], 7, 1, 30, 7, 14)
    wheat_sentiment_forecast_14days = Station3_ARIMAForecast(wheat_mean_sentiment_score['compound'], 14, 1, 30, 14, 30)
    wheat_sentiment_forecast_1month = Station3_ARIMAForecast(wheat_mean_sentiment_score['compound'], 30, 1, 30, 30,
                                                             len(wheat_mean_sentiment_score))

    # Station #4
    # Return forecast in %
    Station4_ReturnForecast(corn_price_df['Settlement Price'][-1], corn_price_arima_forecast_7days[-1],
                            corn_price_arima_forecast_14days[-1], corn_price_arima_forecast_1month[-1],
                            corn_price_arima_forecast_3months[-1], 'Corn', ['7 days', '14 days', '1 month', '3 months'])
    Station4_ReturnForecast(wheat_price_df['Settlement Price'][-1], wheat_price_arima_forecast_7days[-1],
                            wheat_price_arima_forecast_14days[-1], wheat_price_arima_forecast_1month[-1],
                            wheat_price_arima_forecast_3months[-1], 'Wheat',
                            ['7 days', '14 days', '1 month', '3 months'])

    # ARIMA Price Forecast line plots
    Station4_ARIMALinePlot(corn_price_arima_forecast_7days, corn_price_arima_forecast_14days,
                           corn_price_arima_forecast_1month, corn_price_arima_forecast_3months,
                           [30, 60, 90, 180], 'Corn Settlement Price', ['7 days', '14 days', '1 month', '3 months'])
    Station4_ARIMALinePlot(wheat_price_arima_forecast_7days, wheat_price_arima_forecast_14days,
                           wheat_price_arima_forecast_1month, wheat_price_arima_forecast_3months,
                           [30, 60, 90, 180], 'Wheat Settlement Price', ['7 days', '14 days', '1 month', '3 months'])

    # ARIMA Trading Volume Forecast bar plots
    Station4_ARIMABarPlot(corn_volume_arima_forecast_7days, corn_volume_arima_forecast_14days,
                          corn_volume_arima_forecast_1month, corn_volume_arima_forecast_3months,
                          [30, 60, 90, 180], 'Corn Trading Volume', ['7 days', '14 days', '1 month', '3 months'])
    Station4_ARIMABarPlot(wheat_volume_arima_forecast_7days, wheat_volume_arima_forecast_14days,
                          wheat_volume_arima_forecast_1month, wheat_volume_arima_forecast_3months,
                          [30, 60, 90, 180], 'Wheat Trading Volume', ['7 days', '14 days', '1 month', '3 months'])

    # Sentiment Analysis mean
    Station4_MeanSentiment(corn_mean_sentiment_score['compound'].values, 'Corn Sentiment', [3, 7, 14, 30],
                           ['last 3 Days', 'last 7 days', 'last 14 days', 'last 1 month', 'the whole data period'])
    Station4_MeanSentiment(wheat_mean_sentiment_score['compound'].values, 'Wheat Sentiment', [3, 7, 14, 30],
                           ['last 3 Days', 'last 7 days', 'last 14 days', 'last 1 month', 'the whole data period'])

    # ARIMA Sentiment Forecast line plots
    Station4_ARIMALinePlot(corn_sentiment_forecast_3days, corn_sentiment_forecast_7days,
                           corn_sentiment_forecast_14days, corn_sentiment_forecast_1month,
                           [14, 14, 30, len(corn_mean_sentiment_score)], 'Corn Sentiment',
                           ['3 Days', '7 days', '14 days', '1 month'])
    Station4_ARIMALinePlot(wheat_sentiment_forecast_3days, wheat_sentiment_forecast_7days,
                           wheat_sentiment_forecast_14days, wheat_sentiment_forecast_1month,
                           [14, 14, 30, len(corn_mean_sentiment_score)], 'Wheat Sentiment',
                           ['3 Days', '7 days', '14 days', '1 month'])


if __name__ == '__main__':
    main()
