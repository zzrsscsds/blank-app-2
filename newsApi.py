from newsapi import NewsApiClient
import pandas as pd
from datetime import datetime, timedelta

from mergeData import combined_df

## WAS USED FOR THE NEWS API, FULL IMPLEMENTATION WAS NOT COMPLETED


newsapi = NewsApiClient(api_key="7af7d5e56edc4148aac908f2c9f86ac3")

# Define time range
start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
end_date = datetime.now().strftime('%Y-%m-%d')

# Get news articles
all_articles = newsapi.get_everything(q="*", from_param=start_date, to=end_date, language='en', page_size=100)

# Extract into DataFrame
news_df = pd.DataFrame([{
    "published_at": article['publishedAt'],
    "title": article['title'],
    "description": article['description']
} for article in all_articles['articles']])

# Clean
news_df['published_at'] = pd.to_datetime(news_df['published_at'])
news_df['text'] = news_df['title'] + " " + news_df['description'].fillna('')

# Convert to hourly buckets
combined_df['created_at'] = pd.to_datetime(combined_df['created_at'])
combined_df['hour_bucket'] = combined_df['created_at'].dt.floor('H')

# Group social text and features
social_agg = combined_df.groupby('hour_bucket').agg({
    'text': lambda x: ' '.join(x),
    'sentiment': 'mean',
    'engagement': 'sum'
}).reset_index()
