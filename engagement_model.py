import tweepy
import pandas as pd
import numpy as np
import re
from collections import Counter
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
### USED TO GET X DATA, WILL ONLY RUN ONCE A MONTH DUE TO X DEV FREE ACCOUNT LIMITS


def authenticate_x_api(bearer_token):
    client = tweepy.Client(bearer_token=bearer_token)
    return client

def extract_hashtags(text):
    return re.findall(r"#\w+", str(text).lower())

def fetch_tweets(client, query, max_results=100):
    tweets = client.search_recent_tweets(query=query, tweet_fields=['created_at', 'public_metrics', 'text'], max_results=max_results)
    tweet_data = []

    for tweet in tweets.data:
        metrics = tweet.public_metrics
        tweet_data.append({
            "text": tweet.text,
            "timestamp": tweet.created_at,
            "likes": metrics.get("like_count", 0),
            "retweets": metrics.get("retweet_count", 0)
        })

    return pd.DataFrame(tweet_data)

def preprocess_and_train(df, weather_features=None):
    if weather_features is None:
        weather_features = ['temperature', 'humidity', 'wind_speed']

    df['engagement'] = df['likes'] + df['retweets']
    df['sentiment'] = df['text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    df['text_length'] = df['text'].apply(lambda x: len(str(x)))
    df['hashtags'] = df['text'].apply(extract_hashtags)

    all_hashtags = [tag for sublist in df['hashtags'] for tag in sublist]
    top_hashtags = [tag for tag, _ in Counter(all_hashtags).most_common(5)]

    for tag in top_hashtags:
        df[f'hashtag_{tag}'] = df['hashtags'].apply(lambda tags: int(tag in tags))

    feature_cols = ['sentiment', 'hour', 'dayofweek', 'text_length']
    feature_cols += [f'hashtag_{tag}' for tag in top_hashtags]
    feature_cols += [col for col in weather_features if col in df.columns]

    df = df.dropna(subset=feature_cols + ['engagement'])

    X = df[feature_cols]
    y = df['engagement']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    return {
        "model": model,
        "r2_score": r2_score(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred
    }


if __name__ == "__main__":
    BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAAJoq1QEAAAAAfyT9IvG9H56BDSqWKzRNNeGRh5Y%3DrHNnc33aiqyaBUvRZHkXHtIzhxg4RgwR1NOaktjFPkToGAzKgm"
    client = authenticate_x_api(BEARER_TOKEN)

    query = "AI OR machine learning OR #DataScience lang:en -is:retweet"
    tweets_df = fetch_tweets(client, query, max_results=100)

    # === Add dummy weather data (you can replace with real data later) ===
    np.random.seed(42)
    tweets_df['temperature'] = np.random.uniform(15, 30, size=len(tweets_df))
    tweets_df['humidity'] = np.random.uniform(30, 80, size=len(tweets_df))
    tweets_df['wind_speed'] = np.random.uniform(0, 10, size=len(tweets_df))

    # Save to CSV
    tweets_df.to_csv("x_posts_with_weather.csv", index=False)
    print("Saved tweets with weather data to x_posts_with_weather.csv")

    # Run your model training if you want
    results = preprocess_and_train(tweets_df)
    print(f"R² Score: {results['r2_score']:.4f}")
    print(f"RMSE: {results['rmse']:.2f}")
