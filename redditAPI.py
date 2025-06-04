import praw
import pandas as pd
from datetime import datetime

reddit = praw.Reddit(
    client_id='v5b2CYNg37amXniM43bNmQ',
    client_secret='cqVeL5VR-vENbiLAjnfC-xoRn45qaQ',
    user_agent="MyRedditSentimentApp/0.1 by noahcrampton"
)

subreddit = reddit.subreddit("fitness")
posts = []

for post in subreddit.new(limit=1500):
    posts.append({
        "title": post.title,
        "score": post.score,
        "comments": post.num_comments,
        "created": datetime.utcfromtimestamp(post.created_utc).strftime('%Y-%m-%d %H:%M:%S'),
        "url": post.url,
        "selftext": post.selftext,
        "subreddit": str(post.subreddit)
    })

df = pd.DataFrame(posts).drop_duplicates(subset="title")
df.sort_values(by="created", ascending=False, inplace=True)

print("Columns:", df.columns.tolist())
print(f"Latest post created: {df['created'].max()}")
print(f"Earliest post created: {df['created'].min()}")
print(f"Total unique posts: {len(df)}")

df.to_csv("data/reddit_fitness_recent_posts2.csv", index=False, date_format='%Y-%m-%d %H:%M:%S')
print("Saved to reddit_fitness_recent_posts2.csv")
