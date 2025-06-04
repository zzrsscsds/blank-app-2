import pandas as pd
from textblob import TextBlob
from datetime import datetime
import pytz  # To handle timezones
## USED TO MERGE ALL DATA SETS TOGETHER TO CREATE ONE CLEANED UP DATASET FOR MODELS ETC


# Set timezone to NZST for current time
nzst = pytz.timezone('Pacific/Auckland')

# Read all datasets
x_posts = pd.read_csv("data/x_posts_with_weather.csv")
reddit_posts = pd.read_csv("data/reddit_all_recent_posts1.csv")
reddit_two = pd.read_csv("data/reddit_all_recent_posts.csv")
reddit_three = pd.read_csv("data/reddit_fitness_recent_posts2.csv")

# Debug invalid created_at values
print(f"X posts invalid created_at: {x_posts['created_at'].isna().sum()} out of {len(x_posts)} rows")
print(f"Reddit 1 invalid created_at: {reddit_posts['created'].isna().sum()} out of {len(reddit_posts)} rows")
print(f"Reddit 2 invalid created_at: {reddit_two['created'].isna().sum()} out of {len(reddit_two)} rows")
print(f"Reddit 3 invalid created_at: {reddit_three['created'].isna().sum()} out of {len(reddit_three)} rows")

# Function to process Reddit datasets uniformly
def process_reddit(df):
    df['text'] = df['title'].fillna('') + ". " + df['selftext'].fillna('')
    # Parse created and standardize to UTC
    df['created_at'] = pd.to_datetime(df['created'], errors='coerce', utc=True)
    # Fill missing created_at with current NZST time converted to UTC
    current_time = datetime.now(nzst).astimezone(pytz.UTC).strftime('%Y-%m-%d %H:%M:%S %Z')
    invalid_dates = df['created_at'].isna().sum()
    if invalid_dates > 0:
        print(f"Filling {invalid_dates} invalid created_at in Reddit dataset with {current_time}")
        df['created_at'] = df['created_at'].fillna(pd.Timestamp(current_time, tz='UTC'))
    df['hour_of_day'] = df['created_at'].dt.hour
    df['is_weekend'] = df['created_at'].dt.dayofweek >= 5
    df['engagement'] = df['score'] + df['comments']
    df['sentiment'] = df['text'].apply(lambda x: TextBlob(x).sentiment.polarity)
    return df[["created_at", "text", "sentiment", "hour_of_day", "is_weekend", "engagement"]]

# Process each Reddit dataset
reddit_cleaned = process_reddit(reddit_posts)
print(f"Reddit 1 date range: {reddit_cleaned['created_at'].min()} to {reddit_cleaned['created_at'].max()}")
reddit_two_cleaned = process_reddit(reddit_two)
print(f"Reddit 2 date range: {reddit_two_cleaned['created_at'].min()} to {reddit_two_cleaned['created_at'].max()}")
reddit_three_cleaned = process_reddit(reddit_three)
print(f"Reddit 3 date range: {reddit_three_cleaned['created_at'].min()} to {reddit_three_cleaned['created_at'].max()}")

# Process X posts
x_posts['created_at'] = pd.to_datetime(x_posts['created_at'], errors='coerce')
# Standardize X posts to UTC and fill missing
current_time = datetime.now(nzst).astimezone(pytz.UTC).strftime('%Y-%m-%d %H:%M:%S %Z')
invalid_x_dates = x_posts['created_at'].isna().sum()
if invalid_x_dates > 0:
    print(f"Filling {invalid_x_dates} invalid created_at in X posts with {current_time}")
    x_posts['created_at'] = x_posts['created_at'].fillna(pd.Timestamp(current_time, tz='UTC'))
x_posts['created_at'] = x_posts['created_at'].dt.tz_convert('UTC')  # Ensure UTC
print(f"X posts date range: {x_posts['created_at'].min()} to {x_posts['created_at'].max()}")
x_posts_cleaned = x_posts[["created_at", "text", "sentiment", "hour_of_day", "is_weekend", "engagement"]]

# Combine all four datasets into one DataFrame
combined_df = pd.concat([x_posts_cleaned, reddit_cleaned, reddit_two_cleaned, reddit_three_cleaned], ignore_index=True)
# Drop rows with missing text or engagement, but not created_at
combined_df.dropna(subset=["text", "engagement"], inplace=True)
# Final check and fill for created_at
invalid_combined_dates = combined_df['created_at'].isna().sum()
if invalid_combined_dates > 0:
    current_time = datetime.now(nzst).astimezone(pytz.UTC).strftime('%Y-%m-%d %H:%M:%S %Z')
    print(f"Filling {invalid_combined_dates} invalid created_at in combined dataset with {current_time}")
    combined_df['created_at'] = combined_df['created_at'].fillna(pd.Timestamp(current_time, tz='UTC'))
print(f"Invalid created_at in combined dataset: {combined_df['created_at'].isna().sum()} out of {len(combined_df)} rows")
print(f"Combined dataset date range: {combined_df['created_at'].min()} to {combined_df['created_at'].max()}")

# Save combined dataset
combined_df.to_csv("data/combined_social_data.csv", index=False)

# Print only the created_at column
print(combined_df['created_at'])
