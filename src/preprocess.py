import pandas as pd
import numpy as np
import re

# read the data and print the first 5 rows
df = pd.read_csv("../data/raw/tripadvisor_hotel_reviews.csv")
print(df.head())

# drop rows with missing values in the Review column and print the shape of the dataframe
df.dropna(subset=["Review"], inplace=True)
print(df.shape)

# define a function to clean the text and apply it to the Review column
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# apply the function to the Review column and print the first 5 rows
df['Review'] = df['Review'].apply(clean_text)
print(df.head())

# define a function to label the sentiment and apply it to the Rating column
def sentiment_label(rating):
    if rating >= 4:
        return "positive"
    elif rating == 3:
        return np.nan
    else:
        return "negative"

# apply the function to the Rating column and drop rows with missing values in the Sentiment column
df['Sentiment'] = df['Rating'].apply(sentiment_label) 
df.dropna(subset=["Sentiment"], inplace=True)
print(df.shape)
print(df['Sentiment'].value_counts())
# save the processed data to a new csv file
df.to_csv("../data/processed/tripadvisor_hotel_reviews.csv", index=False)
