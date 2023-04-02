from tkinter import *
from PIL import Image, ImageTk
from tkinter import messagebox
from collections import Counter
import logging
import time
import pickle
import itertools
import re
import string
import pickle
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import twitter_samples, stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk import FreqDist, classify, NaiveBayesClassifier
import numpy as np
import pandas as pd
import re, string, random
import sklearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import math
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor


nltk.download('stopwords')
#nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('twitter_samples')
nltk.download('all')

root = Tk()  # Main window 
f = Frame(root)
frame1 = Frame(root)
frame2 = Frame(root)
frame3 = Frame(root)
root.title("Stock Market Prediction With Sentiment Analysis")
root.geometry("720x520")

canvas = Canvas(width=720, height=250)
canvas.pack()
filename=('images1.png')
load = Image.open(filename)
load = load.resize((720, 250), Image.ANTIALIAS)
render = ImageTk.PhotoImage(load)
img = Label(image=render)
img.image = render
#photo = PhotoImage(file='landscape.png')
load = Image.open(filename)
img.place(x=1, y=1)
#canvas.create_image(-80, -80, image=img, anchor=NW)


root.configure(background='white')
scrollbar = Scrollbar(root)
scrollbar.pack(side=RIGHT, fill=Y)
stk = StringVar()

def remove_noise(tweet_tokens, stop_words = ()):

    cleaned_tokens = []

    for token, tag in pos_tag(tweet_tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
        token = re.sub("(@[A-Za-z0-9_]+)","", token)

        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens

def get_all_words(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        for token in tokens:
            yield token

def get_tweets_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)

if __name__ == "__main__":

    positive_tweets = twitter_samples.strings('positive_tweets.json')
    negative_tweets = twitter_samples.strings('negative_tweets.json')
    text = twitter_samples.strings('tweets.20150430-223406.json')
    tweet_tokens = twitter_samples.tokenized('positive_tweets.json')[0]

    stop_words = stopwords.words('english')

    positive_tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
    negative_tweet_tokens = twitter_samples.tokenized('negative_tweets.json')

    positive_cleaned_tokens_list = []
    negative_cleaned_tokens_list = []

    for tokens in positive_tweet_tokens:
        positive_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

    for tokens in negative_tweet_tokens:
        negative_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

    all_pos_words = get_all_words(positive_cleaned_tokens_list)

    freq_dist_pos = FreqDist(all_pos_words)
    print(freq_dist_pos.most_common(10))

    positive_tokens_for_model = get_tweets_for_model(positive_cleaned_tokens_list)
    negative_tokens_for_model = get_tweets_for_model(negative_cleaned_tokens_list)

    positive_dataset = [(tweet_dict, 1)
                         for tweet_dict in positive_tokens_for_model]

    negative_dataset = [(tweet_dict, 0)
                         for tweet_dict in negative_tokens_for_model]

    dataset = positive_dataset + negative_dataset

    random.shuffle(dataset)

    train_data = dataset[:7000]
    test_data = dataset[7000:]

    classifier = NaiveBayesClassifier.train(train_data)

    print("Accuracy is:", classify.accuracy(classifier, test_data))

    print(classifier.show_most_informative_features(10))


def window_data(df, window, feature_col_number1, feature_col_number2, feature_col_number3, target_col_number):
    # Create empty lists "X_close", "X_polarity", "X_volume" and y
    X_close = []
    X_polarity = []
    X_Cscore = []
    y = []
    for i in range(len(df) - window):
        
        # Get close, ts_polarity, tw_vol, and target in the loop
        close = df.iloc[i:(i + window), feature_col_number1]
        ts_polarity = df.iloc[i:(i + window), feature_col_number2]
        tw_score = df.iloc[i:(i + window), feature_col_number3]
        target = df.iloc[(i + window), target_col_number]
        
        # Append values in the lists
        X_close.append(close)
        X_polarity.append(ts_polarity)
        X_Cscore.append(tw_score)
        y.append(target)
        
    return np.hstack((X_close,X_polarity,X_Cscore)), np.array(y).reshape(-1, 1)



def condition(x):
    if x > 0:
        return "Positive"
    elif x==0:
        return "neutral"
    else:
        return 'Negative'


def click():
    Name = stk.get()
    stockfilename='./Data/'+Name+'(stockdata).csv'
    tweetsfilename='./Data/'+Name+'2022.csv'
    stocks_df = pd.read_csv(stockfilename)
    tweets_df = pd.read_csv(tweetsfilename, encoding='latin1')

    stocks_df

    #drop days from tweets df that are on weekends and holidays because the stock market is only open on weekdays
    tweets_dropped1 = pd.merge(stocks_df["Date"], tweets_df, on="Date")

    #drop days from tweets df that are on weekends and holidays because the stock market is only open on weekdays
    tweets_dropped = pd.merge(stocks_df["Date"], tweets_df, on="Date").drop(columns=["Date"])

    tweets_dropped.head(1)

    #process tweets into sentiment values for a given day
    tweets_tokenized = tweets_dropped.applymap(lambda x: word_tokenize(x) if not pd.isnull(x) else x)
    tweets_noiseless = tweets_tokenized.applymap(lambda y: remove_noise(y) if not pd.isnull([y]).any() else y)

    #tweets_noiseless = pd.DataFrame(tweets_noiseless)
    tweets_noiseless

    from nltk.tokenize.treebank import TreebankWordDetokenizer
    twd =TreebankWordDetokenizer()

    reconstructedSentence = tweets_noiseless.applymap(lambda z: twd.detokenize(z) if not pd.isnull([z]).any() else z)

    reconstructedSentence

    df1 = pd.DataFrame()

    sent = SentimentIntensityAnalyzer()

    df1 = reconstructedSentence.applymap(lambda x: round(sent.polarity_scores(x)['compound'], 2) if not pd.isnull([x]).any() else x)

    df1

    sentiment_score = df1.mean(axis=1)

    df2 = pd.DataFrame(sentiment_score)
    df2 = pd.concat([tweets_dropped1['Date'],df2],axis=1)
    df2

    prelim = pd.merge(stocks_df.drop(columns=["High", "Low", "Close"]), df2, on='Date', how='left')
    prelim = prelim.replace(np.nan,0)
    prelim.columns = ['Date', 'Open', 'Adj Close', 'Volume', 'Sentiment_score']

    prelim['com_score'] = prelim['Sentiment_score'].apply(condition)
    prelim['com_score'] = prelim['com_score'].replace({'Positive':1,'neutral':2,'Negative':3})
    prelim.head()

    prelim = prelim.set_index('Date')
    prelim.head()

    #prelim['Adj Close'].plot()

    prelim["Pct_change"] = prelim["Adj Close"].pct_change()
    prelim.dropna(inplace = True)
    prelim.head()

    prelim.columns

    prelim = prelim.drop(['Open','Volume'],axis='columns')

    prelim

    window_size = 3

    # Column index 0 is the `Adj Close` column
    # Column index 1 is the `ts_polarity` column
    # Column index 2 is the `twitter_volume` column
    feature_col_number1 = 0
    feature_col_number2 = 1
    feature_col_number3 = 2
    target_col_number = 0
    X, y = window_data(prelim, window_size, feature_col_number1, feature_col_number2, feature_col_number3, target_col_number)

    # Use 90% of the data for training and the remainder for testing
    X_split = int(0.9 * len(X))
    y_split = int(0.9 * len(y))

    X_train = X[: X_split]
    X_test = X[X_split:]
    y_train = y[: y_split]
    y_test = y[y_split:]

    X_train.shape

    # Use the MinMaxScaler to scale data between 0 and 1.
    x_train_scaler = MinMaxScaler()
    x_test_scaler = MinMaxScaler()
    y_train_scaler = MinMaxScaler()
    y_test_scaler = MinMaxScaler()

    # Fit the scaler for the Training Data
    x_train_scaler.fit(X_train)
    y_train_scaler.fit(y_train)

    # Scale the training data
    X_train = x_train_scaler.transform(X_train)
    y_train = y_train_scaler.transform(y_train)

    # Fit the scaler for the Testing Data
    x_test_scaler.fit(X_test)
    y_test_scaler.fit(y_test)

    # Scale the y_test data
    X_test = x_test_scaler.transform(X_test)
    y_test = y_test_scaler.transform(y_test)

    # Create the XG Boost regressor instance
    model = XGBRegressor()
    model.fit(X_train, y_train.ravel())

    predicted = model.predict(X_test)

    # Evaluating the model
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, predicted)))
    print('R-squared :', metrics.r2_score(y_test, predicted))

    predicted_prices = y_test_scaler.inverse_transform(predicted.reshape(-1, 1))
    real_prices = y_test_scaler.inverse_transform(y_test.reshape(-1, 1))
    # Create a DataFrame of Real and Predicted values
    stocks = pd.DataFrame({
        "Real": real_prices.ravel(),
        "Predicted": predicted_prices.ravel()
    }, index = prelim.index[-len(real_prices): ]) 
    stocks.head()
    print(stocks.columns)


    plt.figure(figsize=(25,15), dpi=80, facecolor='w', edgecolor='k')
    ax = plt.gca()  
    plt.plot(stocks['Real'], color = 'red', label = 'Real Stock Price')
    plt.plot(stocks['Predicted'], color = 'blue', label = 'Predicted Stock Price')
    plt.title('Stock Price Prediction', fontsize=40)
    plt.xticks(rotation = 'vertical')
    plt.xlabel('Time', fontsize=40)
    plt.ylabel('Stock Price', fontsize=40)
    plt.legend(loc=2, prop={'size': 25})
    plt.show()
    print("finish")


label1 = Label(root, text="Stock Market Prediction With Sentiment Analysis")
label1.config(font=('Italic', 18, 'bold'), justify=CENTER, background="Yellow", fg="black", anchor="center")
label1.pack(fill=X)


frame2.pack_forget()
frame3.pack_forget()


satisfaction_level = Label(frame2, text="Enter Query Text: ", bg="yellow", fg="Black")
satisfaction_level.grid(row=1, column=1, padx=10)
stk.set("Select Stock")
e1 = OptionMenu(frame2, stk, "Select Option", "AAPL", "AMZN", "TSLA", "META", "GOOG")
e1.grid(row=1, column=2, padx=10)


button5 = Button(frame2, text="Submit", command=click)
button5.grid(row=1, column=3, pady=10,padx=10)


frame2.configure(background="Yellow")
frame2.pack(pady=10)

frame1.configure(background="Red")
frame1.pack(pady=10)

root.mainloop()
