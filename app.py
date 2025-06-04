import streamlit as st
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from prophet import Prophet
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import VAR
from xgboost import XGBRegressor
import logging
from datetime import datetime, timedelta
from newsapi import NewsApiClient
import warnings



# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Download NLTK resources
nltk_resources = [
    'punkt', 'punkt_tab', 'stopwords', 'vader_lexicon',
    'averaged_perceptron_tagger', 'wordnet', 'omw-1.4'
]
for res in nltk_resources:
    try:
        nltk.data.find(res)
    except LookupError:
        nltk.download(res, quiet=True)

# Streamlit page configuration
st.set_page_config(page_title="Social Trends Forecaster", layout="wide")
st.markdown("""
    <style>
        .main { background-color: #fafafa; }
        h1 { color: #1f77b4; }
        .stButton>button { background-color: #1f77b4; color: white; border-radius: 5px; }
        .stDownloadButton>button { background-color: #2ca02c; color: white; border-radius: 5px; }
    </style>
""", unsafe_allow_html=True)

st.title("📊 Real-Time Social Media Trend Forecaster")

# Initialize VADER
try:
    sid = SentimentIntensityAnalyzer()
    test_score = sid.polarity_scores("I love this! It's amazing!")
    logger.debug(f"VADER test score: {test_score}")
    if test_score['compound'] == 0:
        st.warning("VADER sentiment analyzer may not be functioning correctly.")
except Exception as e:
    st.error(f"Failed to initialize VADER: {e}")
    st.stop()

@st.cache_data
def load_combined_data(_version):
    try:
        df = pd.read_csv("data/combined_social_data.csv")
        df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
        required_columns = ['created_at', 'text']
        if not all(col in df.columns for col in required_columns):
            st.error(f"Data file missing required columns: {required_columns}")
            return pd.DataFrame()
        logger.debug(f"Loaded CSV with shape: {df.shape}, Columns: {list(df.columns)}")
        st.write(f"Raw dataset date range: {df['created_at'].min()} to {df['created_at'].max()}")
        return df
    except FileNotFoundError:
        st.error("Data file 'combined_social_data.csv' not found.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return pd.DataFrame()

combined_df = load_combined_data(_version=datetime.now().timestamp())

@st.cache_data
def load_recent_news():
    try:
        newsapi = NewsApiClient(api_key="7af7d5e56edc4148aac908f2c9f86ac3")
        start_date = (datetime.now() - timedelta(days=14)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')
        articles = newsapi.get_everything(q="*", from_param=start_date, to=end_date, language='en', page_size=100)
        news_df = pd.DataFrame([{
            "published_at": a['publishedAt'],
            "title": a['title'],
            "description": a['description']
        } for a in articles['articles']])
        news_df['published_at'] = pd.to_datetime(news_df['published_at'], utc=True)
        news_df['text'] = news_df['title'].fillna('') + " " + news_df['description'].fillna('')
        # Compute sentiment and topics
        news_df['sentiment'] = news_df['text'].apply(compute_sentiment)
        news_df['topic'] = extract_topics(news_df['text'].tolist())
        return news_df
    except Exception as e:
        st.warning(f"Failed to fetch news: {e}")
        return pd.DataFrame()

def compute_sentiment(text):
    try:
        text = str(text).strip()
        if not text or text.lower() in ['nan', 'none', '']:
            logger.debug(f"Empty or invalid text: {text}")
            return 0.0
        scores = sid.polarity_scores(text)
        compound_score = scores['compound']
        logger.debug(f"Text: {text[:50]}... Sentiment: {compound_score}")
        return compound_score
    except Exception as e:
        logger.error(f"Sentiment computation error for text '{text[:50]}...': {e}")
        return 0.0

def add_extra_features(df):
    df['emoji_count'] = df['text'].str.count(r'[😀-🙏]')
    df['question_flag'] = df['text'].str.contains(r'\?').astype(int)
    df['text_length_log'] = np.log1p(df['text'].apply(len))
    df['capital_word_count'] = df['text'].str.findall(r'\b[A-Z]{2,}\b').apply(len)
    df['punctuation_count'] = df['text'].str.count(r'[.!?]')
    df['text_length'] = df['text'].apply(len)
    df['hashtag_count'] = df['text'].apply(lambda x: str(x).count('#'))
    df['is_media'] = df['text'].str.contains('https://t.co', na=False).astype(int)
    df['hour'] = df['created_at'].dt.hour
    df['is_weekend'] = df['created_at'].dt.weekday.isin([5, 6]).astype(int)
    return df

def extract_topics(texts):
    stop_words = set(stopwords.words('english')) - {'run', 'pump'}
    processed_texts = [
        " ".join([
            word for word in word_tokenize(doc.lower())
            if (word.isalnum() or word.startswith('#') or word in ['🦵🏽', '💪🏽']) and word not in stop_words
        ]) for doc in texts if isinstance(doc, str) and doc.strip()
    ]
    processed_texts = [doc for doc in processed_texts if len(doc.strip().split()) > 1]
    if len(processed_texts) < 5:
        return [0] * len(texts)
    try:
        vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=1000)
        dtm = vectorizer.fit_transform(processed_texts)
        lda = LatentDirichletAllocation(n_components=3, random_state=42)
        lda.fit(dtm)
        topics = lda.transform(dtm).argmax(axis=1)
        padded_topics = [topics[i] if i < len(topics) else 0 for i in range(len(texts))]
        return padded_topics
    except Exception as e:
        logger.error(f"Topic modeling error: {e}")
        return [0] * len(texts)

def drop_constant_columns(df):
    return df.loc[:, df.nunique() > 1]





warnings.filterwarnings("ignore")  # For cleaner output

def preprocess_and_train(df: pd.DataFrame, model_choice: str = "RandomForest"):
    try:
        features = ['sentiment', 'text_length', 'hashtag_count', 'is_media', 'hour', 'is_weekend']
        X = df[features].fillna(df[features].median())
        y = df['engagement'].fillna(df['engagement'].median())

        if len(X) < 30:
            raise ValueError("Not enough data for training (minimum 30 rows recommended).")

        y_log = np.log1p(y)

        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train_log, y_test_log = train_test_split(X_scaled, y_log, test_size=0.2, random_state=42)

        if model_choice == "RandomForest":
            model = RandomForestRegressor(random_state=42)
            param_dist = {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['auto', 'sqrt', 'log2']
            }
        else:
            model = GradientBoostingRegressor(random_state=42)
            param_dist = {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 1.0],
                'min_samples_split': [2, 5, 10]
            }

        search = RandomizedSearchCV(
            model, param_distributions=param_dist, n_iter=20,
            scoring='neg_mean_squared_error', cv=3, n_jobs=-1, random_state=42
        )
        search.fit(X_train, y_train_log)
        best_model = search.best_estimator_

        y_pred_log = best_model.predict(X_test)
        y_pred = np.expm1(y_pred_log)
        y_test = np.expm1(y_test_log)

        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)

        return {
            'r2_score': r2,
            'rmse': rmse,
            'mae': mae,
            'X_test': pd.DataFrame(X_test, columns=features),
            'y_test': y_test,
            'y_pred': y_pred,
            'model': best_model
        }

    except Exception as e:
        print(f"Error in training model: {e}")
        return None


def hybrid_prophet_xgb(df, forecast_periods=24):
    try:
        logger.info("Starting hybrid Prophet+XGBoost prediction")
        df['created_at'] = pd.to_datetime(df['created_at']).dt.tz_localize(None)
        prophet_df = df[['created_at', 'engagement']].copy()
        prophet_df = prophet_df.rename(columns={'created_at': 'ds', 'engagement': 'y'})
        prophet_df = prophet_df.dropna()

        m = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=False)
        m.fit(prophet_df)

        future = m.make_future_dataframe(periods=forecast_periods, freq='H')
        forecast = m.predict(future)

        prophet_features = forecast[['ds', 'trend', 'weekly', 'daily']].copy()
        merged_df = df.merge(prophet_features, left_on='created_at', right_on='ds', how='left').drop(columns=['ds'])

        features = ['sentiment', 'hour', 'is_weekend', 'trend', 'weekly', 'daily']
        merged_df = merged_df.dropna(subset=features + ['engagement'])
        if len(merged_df) < 10:
            raise ValueError("Not enough data for XGBoost training.")

        X = merged_df[features]
        y = merged_df['engagement']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        xgb_model = XGBRegressor(n_estimators=150, max_depth=6, learning_rate=0.1, subsample=0.8, random_state=42)
        xgb_model.fit(X_train, y_train)
        y_pred = xgb_model.predict(X_test)

        importances = xgb_model.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)

        st.subheader("Feature Importance (XGBoost)")
        fig, ax = plt.subplots()
        ax.barh(importance_df['Feature'], importance_df['Importance'], color='orange')
        ax.set_title("Feature Importance")
        ax.invert_yaxis()
        st.pyplot(fig)
        st.subheader("Residual Analysis")
        residuals = y_test - y_pred
        fig1, ax1 = plt.subplots()
        ax1.hist(residuals, bins=20, color='salmon', edgecolor='black')
        ax1.set_title("Distribution of Residuals")
        st.pyplot(fig1)
        fig2, ax2 = plt.subplots()
        ax2.scatter(y_pred, residuals, alpha=0.5)
        ax2.axhline(0, color='gray', linestyle='--')
        ax2.set_xlabel("Predicted Engagement")
        ax2.set_ylabel("Residuals")
        ax2.set_title("Residuals vs Predicted")
        st.pyplot(fig2)
        st.subheader("Prediction Confidence Interval (Bootstrap)")
        from sklearn.utils import resample
        n_iterations = 100
        predictions = []
        for _ in range(n_iterations):
            X_bs, y_bs = resample(X_train, y_train)
            xgb_model.fit(X_bs, y_bs)
            pred_bs = xgb_model.predict(X_test)
            predictions.append(pred_bs)
        pred_array = np.array(predictions)
        lower = np.percentile(pred_array, 2.5, axis=0)
        upper = np.percentile(pred_array, 97.5, axis=0)
        fig, ax = plt.subplots()
        ax.plot(y_test.index, y_pred, label='Prediction', color='blue')
        ax.fill_between(y_test.index, lower, upper, color='lightblue', alpha=0.4, label='95% CI')
        ax.plot(y_test.index, y_test, label='Actual', color='red')
        ax.set_title("Predicted vs Actual with Confidence Interval")
        ax.legend()
        st.pyplot(fig)        

        future_df = forecast[forecast['ds'] > prophet_df['ds'].max()].copy()
        future_df['created_at'] = pd.to_datetime(future_df['ds'])
        future_df['hour'] = future_df['ds'].dt.hour
        future_df['is_weekend'] = (future_df['ds'].dt.dayofweek >= 5).astype(int)
        future_df['sentiment'] = 0.0
        future_features = future_df[['sentiment', 'hour', 'is_weekend', 'trend', 'weekly', 'daily']]

        future_engagement_pred = xgb_model.predict(future_features)

        future_summary = {
            "average_engagement": float(np.mean(future_engagement_pred)),
            "max_engagement": float(np.max(future_engagement_pred)),
            "max_engagement_time": future_df['ds'].iloc[np.argmax(future_engagement_pred)].strftime('%Y-%m-%d %H:%M:%S'),
            "min_engagement": float(np.min(future_engagement_pred)),
            "min_engagement_time": future_df['ds'].iloc[np.argmin(future_engagement_pred)].strftime('%Y-%m-%d %H:%M:%S'),
            "trend": "increasing" if future_engagement_pred[-1] > future_engagement_pred[0] else "decreasing"
        }

        return {
            "prophet_forecast": forecast[['ds', 'yhat']],
            "xgb_model": xgb_model,
            "xgb_features": features,
            "r2_score": r2_score(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "X_test": X_test.assign(created_at=merged_df.loc[X_test.index, 'created_at']),
            "y_test": y_test,
            "y_pred": y_pred,
            "future_dates": future_df['ds'],
            "future_engagement_pred": future_engagement_pred,
            "future_summary": future_summary
        }
    except Exception as e:
        logger.error(f"Hybrid model error: {str(e)}", exc_info=True)
        return None

def predict_headline_features(social_df, news_df, forecast_periods=7):
    try:
        social_df['date'] = pd.to_datetime(social_df['date']).dt.tz_localize(None)
        news_df['date'] = pd.to_datetime(news_df['date']).dt.tz_localize(None)

        merged_df = pd.merge(social_df, news_df, on='date', how='inner', suffixes=('_social', '_news'))
        features = ['sentiment_social', 'engagement', 'topic_social', 'is_media']
        target_sentiment = 'sentiment_news'
        target_topic = 'topic_news'

        X = merged_df[features].fillna(0)
        y_sentiment = merged_df[target_sentiment].fillna(0)
        y_topic = merged_df[target_topic].astype(int)

        if len(X) < 10:
            raise ValueError("Not enough data for headline prediction (minimum 10 rows required).")

        X_train, X_test, y_train_sentiment, y_test_sentiment = train_test_split(X, y_sentiment, test_size=0.2, random_state=42)
        _, _, y_train_topic, y_test_topic = train_test_split(X, y_topic, test_size=0.2, random_state=42)

        sentiment_model = RandomForestRegressor(n_estimators=100, random_state=42)
        sentiment_model.fit(X_train, y_train_sentiment)
        y_pred = sentiment_model.predict(X_test)
        sentiment_rmse = np.sqrt(mean_squared_error(y_test_sentiment, y_pred))

        topic_model = RandomForestClassifier(n_estimators=100, random_state=42)
        topic_model.fit(X_train, y_train_topic)
        topic_accuracy = topic_model.score(X_test, y_test_topic)

        future_dates = pd.date_range(start=social_df['date'].max() + pd.Timedelta(days=1), periods=forecast_periods, freq='D')
        future_social_df = pd.DataFrame({
            'date': future_dates,
            'sentiment_social': social_df['sentiment_social'].mean(),
            'engagement': social_df['engagement'].mean(),
            'topic_social': social_df['topic_social'].mode()[0],
            'is_media': social_df['is_media'].mean()
        })

        future_X = future_social_df[features].fillna(0)
        future_sentiment = sentiment_model.predict(future_X)
        future_topics = topic_model.predict(future_X)

        return {
            'future_dates': future_dates,
            'future_sentiment': future_sentiment,
            'future_topics': future_topics,
            'sentiment_rmse': sentiment_rmse,
            'topic_accuracy': topic_accuracy
        }
    except Exception as e:
        st.error(f"Headline prediction error: {e}")
        return None


def generate_headline(sentiment, topic):
    topic_keywords = {
        0: "Fitness Trends",
        1: "Workout Tips", #used for fitness topic to test
        2: "Health News"
    }
    if sentiment > 0.2:
        tone = "Positive Update"
    elif sentiment < -0.2:
        tone = "Concerning News"
    else:
        tone = "Neutral Report"
    topic_name = topic_keywords.get(topic, "General News")
    return f"{tone}: {topic_name} Expected to Gain Attention"

if combined_df.empty:
    st.warning("No data loaded. Please check 'combined_social_data.csv'.")
    st.stop()

st.info(f"Raw dataset size: {combined_df.shape[0]} rows, {combined_df.shape[1]} columns")
st.write(f"Raw dataset date range: {combined_df['created_at'].min()} to {combined_df['created_at'].max()}")
st.subheader("Sample of Raw Dataset")
st.dataframe(combined_df[['created_at', 'text']].tail(5))  # Show last 5 rows

st.sidebar.title("Filter Settings")
keyword = st.sidebar.text_input("Enter a topic keyword (To see all data ensure input box is empty and press enter):", "Fitness").lower().replace("#", "")
show_news = st.sidebar.checkbox("📰 Show Latest News Headlines")
model_choice = st.sidebar.selectbox("Select Regression Model", ["RandomForest", "GradientBoosting"])

filtered_df = combined_df[combined_df['text'].str.lower().str.contains(keyword, na=False)].copy()
st.info(f"Dataset size after keyword filtering: {filtered_df.shape[0]} rows, {filtered_df.shape[1]} columns")
if filtered_df.empty:
    st.warning(f"No posts found for '{keyword}'. Check if the keyword exists in the data.")
    st.stop()

if show_news:
    news_df = load_recent_news()
    if not news_df.empty:
        st.subheader("🗞 Recent News Highlights")
        for i, row in news_df.head(5).iterrows():
            st.markdown(f"**{row['published_at'].strftime('%Y-%m-%d %H:%M')}** - {row['title']}")
    else:
        st.info("No recent news available.")

st.info(f"Filtered dataset size: {filtered_df.shape[0]} rows, {filtered_df.shape[1]} columns")
st.write(f"Filtered dataset date range: {filtered_df['created_at'].min()} to {filtered_df['created_at'].max()}")
st.subheader("Sample of Filtered Dataset")
st.dataframe(filtered_df[['created_at', 'text']].tail(5))  # Show last 5 rows

filtered_df = filtered_df.dropna(subset=['text', 'created_at'])
filtered_df['text'] = filtered_df['text'].astype(str).replace('', np.nan).dropna()
filtered_df = add_extra_features(filtered_df)
filtered_df['sentiment'] = filtered_df['text'].apply(compute_sentiment)
if (filtered_df['sentiment'] == 0).all():
    st.warning("All sentiment scores are 0. Possible issues with text data or VADER analyzer.")
filtered_df['engagement'] = pd.to_numeric(filtered_df.get('engagement', 0), errors='coerce').fillna(0)
if 'engagement' not in filtered_df.columns or filtered_df['engagement'].sum() == 0:
    engagement_cols = [col for col in ['likes', 'retweets', 'shares'] if col in filtered_df.columns]
    if engagement_cols:
        filtered_df['engagement'] = filtered_df[engagement_cols].sum(axis=1)
    else:
        filtered_df['engagement'] = 0
filtered_df['topic'] = extract_topics(filtered_df['text'].tolist())
st.success(f"✅ Total filtered posts: {filtered_df.shape[0]}")

time_df = filtered_df.groupby(filtered_df['created_at'].dt.floor('H')).agg({
    'sentiment': 'mean',
    'engagement': 'sum',
    'topic': lambda x: x.mode()[0] if not x.mode().empty else 0,
    'hour': lambda x: x.mode()[0] if not x.mode().empty else 0,
    'is_media': 'mean'
}).dropna()

st.header("📈 Topic-Driven Engagement Forecasting")

st.header("📊 Dataset Overview")
st.info(f"Raw dataset size: {combined_df.shape[0]} rows, {combined_df.shape[1]} columns")
st.info(f"Filtered dataset size: {filtered_df.shape[0]} rows, {filtered_df.shape[1]} columns")

st.subheader("📊 Engagement & Sentiment Over Time")
if not time_df.empty and len(time_df) > 1:
    chart_data = pd.DataFrame({
        'Time': time_df.index,
        'Engagement': time_df['engagement'].fillna(0),
        'Sentiment': time_df['sentiment'].fillna(0)
    }).set_index('Time')
    st.line_chart(chart_data[['Engagement', 'Sentiment']])
else:
    st.warning("Insufficient data for engagement and sentiment trends.")

st.subheader("💬 Sentiment Distribution")
if filtered_df['sentiment'].notnull().sum() > 0:
    sentiment_binned = pd.cut(filtered_df['sentiment'], bins=10)
    sentiment_counts = sentiment_binned.value_counts().sort_index()
    chart_data = pd.DataFrame({
        'Sentiment Range': [str(interval) for interval in sentiment_counts.index],
        'Count': sentiment_counts
    }).set_index('Sentiment Range')
    st.bar_chart(chart_data['Count'])
else:
    st.warning("No sentiment data available to display distribution.")

st.subheader("📌 Topic vs. Average Engagement")
if filtered_df['topic'].nunique() > 1:
    chart_data = filtered_df.groupby('topic')['engagement'].mean()
    st.bar_chart(chart_data)
else:
    st.warning("Insufficient topic diversity for engagement analysis.")

st.subheader("🌐 Word Cloud")
text = " ".join(filtered_df['text'].dropna().tolist())
if text.strip():
    wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=100).generate(text)
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)
else:
    st.warning("No valid text for word cloud. Using sample text.")
    wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=100).generate(
        "fitness gym workout motivation health")
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

st.subheader("🔍 Popular Subtopics")
stop_words = set(stopwords.words('english')) - {'run', 'pump'}
texts = filtered_df['text'].dropna().tolist()
processed_texts = [
    " ".join([
        word for word in word_tokenize(doc.lower())
        if (word.isalnum() or word.startswith('#') or word in ['🦵🏽', '💪🏽']) and word not in stop_words
    ]) for doc in texts if isinstance(doc, str) and doc.strip()
]
processed_texts = [doc for doc in processed_texts if len(doc.strip().split()) > 1]
if len(processed_texts) >= 5:
    try:
        vectorizer = CountVectorizer(max_df=0.9, min_df=2, max_features=1000)
        dtm = vectorizer.fit_transform(processed_texts)
        lda = LatentDirichletAllocation(n_components=3, random_state=42)
        lda.fit(dtm)
        words = vectorizer.get_feature_names_out()
        for i, topic_dist in enumerate(lda.components_):
            topic_words = [words[i] for i in topic_dist.argsort()[-5:][::-1]]
            st.write(f"**Topic {i + 1}:** {', '.join(topic_words)}")
    except Exception as e:
        st.warning(f"Failed to perform topic modeling: {e}")
        hashtags = filtered_df['text'].str.findall(r'#\w+').explode().value_counts().head(5)
        st.write("**Top Hashtags**: " + ", ".join(hashtags.index if not hashtags.empty else ["No hashtags found"]))
else:
    hashtags = filtered_df['text'].str.findall(r'#\w+').explode().value_counts().head(5)
    st.write("**Top Hashtags**: " + ", ".join(hashtags.index if not hashtags.empty else ["No hashtags found"]))

st.subheader("⏰ Optimal Posting Times")
hourly_engagement = filtered_df.groupby('hour')['engagement'].mean().reset_index()
if not hourly_engagement.empty:
    chart_data = pd.DataFrame({
        'Hour of Day': hourly_engagement['hour'].astype(str),
        'Average Engagement': hourly_engagement['engagement']
    }).set_index('Hour of Day')
    st.bar_chart(chart_data['Average Engagement'])
else:
    st.warning("No data available for optimal posting times.")




st.subheader("🧠 Time Series Regression with VAR")
try:
    model_data = time_df[['engagement', 'sentiment', 'topic', 'hour', 'is_media']]
    model_data = drop_constant_columns(model_data)
    if len(model_data.columns) < 2:
        st.warning("Not enough variable columns for VAR model after dropping constants.")
    elif len(model_data) < 2:
        st.warning("Not enough data points for VAR model.")
    else:
        model = VAR(model_data)
        results = model.fit(maxlags=1)
        forecast = results.forecast(model_data.values[-1:], steps=24)
        forecast_df = pd.DataFrame(forecast, columns=model_data.columns)
        st.line_chart(forecast_df[['engagement']])
        with st.expander("Show VAR Coefficients"):
            st.dataframe(results.params)
except Exception as e:
    st.warning(f"VAR model error: {e}")

st.subheader("Hybrid Prophet + XGBoost Forecast")
try:
    with st.spinner("Training hybrid model..."):
        result = hybrid_prophet_xgb(filtered_df, forecast_periods=24)  # Predict next 24 hours
    if result:
        st.success("Hybrid model trained successfully!")
        st.write(f"**R² Score**: {result['r2_score']:.2f}")
        st.write(f"**RMSE**: {result['rmse']:.2f}")

        st.subheader("📈 Historical Prophet Forecast")
        fig = plt.figure(figsize=(12, 6))
        plt.plot(result['prophet_forecast']['ds'], result['prophet_forecast']['yhat'], label='Prophet Forecast')
        plt.xlabel('Time')
        plt.ylabel('Engagement')
        plt.title('Prophet Time Series Forecast (Historical)')
        plt.legend()
        st.pyplot(fig)

        st.subheader("🎯 Future Engagement Prediction (Next 24 Hours)")
        future_df = pd.DataFrame({
            'Date': result['future_dates'],
            'Predicted Engagement': result['future_engagement_pred']
        })
        future_df['Date'] = pd.to_datetime(future_df['Date'])
        future_df.set_index('Date', inplace=True)
        st.line_chart(future_df['Predicted Engagement'])

        st.write("**Future Engagement Predictions Table**")
        st.dataframe(future_df.reset_index())

        st.subheader("📊 Summary of Future Engagement Predictions")
        summary = result['future_summary']
        st.write(f"- **Average Predicted Engagement**: {summary['average_engagement']:.2f}")
        st.write(f"- **Maximum Predicted Engagement**: {summary['max_engagement']:.2f}")
        st.write(f"  - **Peak Time**: {summary['max_engagement_time']}")
        st.write(f"- **Minimum Predicted Engagement**: {summary['min_engagement']:.2f}")
        st.write(f"  - **Low Point Time**: {summary['min_engagement_time']}")
        st.write(f"- **Trend Over the Period**: {summary['trend'].capitalize()}")

        st.subheader("📉 Test Set Comparison")
        chart_df = result['X_test'].copy()
        chart_df['Predicted Engagement'] = result['y_pred']
        chart_df['Actual Engagement'] = result['y_test'].values
        rmse = np.sqrt(mean_squared_error(chart_df['Actual Engagement'], chart_df['Predicted Engagement']))
        mae = mean_absolute_error(chart_df['Actual Engagement'], chart_df['Predicted Engagement'])
        r2 = r2_score(chart_df['Actual Engagement'], chart_df['Predicted Engagement'])

        st.subheader("📊 Summary Statistics")
        col1, col2, col3 = st.columns(3)
        col1.metric("RMSE", f"{rmse:.2f}")
        col2.metric("MAE", f"{mae:.2f}")
        col3.metric("R² Score", f"{r2:.2f}")
        if 'created_at' in chart_df.columns and not chart_df['created_at'].isnull().any():
            chart_df = chart_df.set_index('created_at')
            st.line_chart(chart_df[['Predicted Engagement', 'Actual Engagement']])
        else:
            st.line_chart(chart_df[['Predicted Engagement', 'Actual Engagement']])
            st.warning("Using post indices as x-axis because 'created_at' is missing or invalid.")
except Exception as e:
    st.error(f"Hybrid model error: {e}")

st.subheader("📰 Predicted News Headlines Based on Engagement Trends")
daily_social_df = filtered_df.groupby(filtered_df['created_at'].dt.floor('D')).agg({
    'sentiment': 'mean',
    'engagement': 'sum',
    'topic': lambda x: x.mode()[0] if not x.mode().empty else 0,
    'is_media': 'mean'
}).dropna()
daily_social_df.index.name = 'date'
daily_social_df.reset_index(inplace=True)
daily_social_df['date'] = pd.to_datetime(daily_social_df['date']).dt.tz_localize(None)
st.write("**Daily Social Media Data (Sample):**", daily_social_df.head())

if daily_social_df.empty:
    st.warning("No aggregated social media data. Using sample data.")
    daily_social_df = pd.read_csv("2025-05-30T04-41_export.csv")
    daily_social_df['date'] = pd.to_datetime(daily_social_df['date'])
    daily_social_df['is_media'] = 0  # Add missing column with default value
    st.write("**Sample Social Media Data (Fallback):**", daily_social_df.head())

news_df = load_recent_news()
if not news_df.empty:
    news_daily = news_df.groupby(news_df['published_at'].dt.floor('D')).agg({
        'sentiment': 'mean',
        'topic': lambda x: x.mode()[0] if not x.mode().empty else 0
    }).dropna()
    news_daily.index.name = 'date'
    news_daily.reset_index(inplace=True)
    news_daily['date'] = pd.to_datetime(news_daily['date']).dt.tz_localize(None)
    st.write("**Daily News Data (Sample):**", news_daily.head())
else:
    st.warning("No news data available for headline prediction.")
    merged_df = pd.DataFrame()

if not news_df.empty:
    merged_df = pd.merge(daily_social_df, news_daily, on='date', how='inner', suffixes=('_social', '_news'))
    st.write(f"**Merged Data Size**: {merged_df.shape[0]} rows")
    st.write("**Merged Data (Sample):**", merged_df.head())
else:
    merged_df = pd.DataFrame()

if not merged_df.empty:
    try:
        with st.spinner("Predicting future headlines..."):
            headline_result = predict_headline_features(daily_social_df, news_df, forecast_periods=7)  # Predict next 7 days
        if headline_result:
            st.success("Headline prediction completed!")
            st.write(f"**Sentiment Prediction RMSE**: {headline_result['sentiment_rmse']:.2f}")
            st.write(f"**Topic Prediction Accuracy**: {headline_result['topic_accuracy']:.2f}")

            future_headlines = pd.DataFrame({
                'Date': headline_result['future_dates'],
                'Predicted Sentiment': headline_result['future_sentiment'],
                'Predicted Topic': headline_result['future_topics']
            })
            future_headlines['Headline'] = future_headlines.apply(
                lambda row: generate_headline(row['Predicted Sentiment'], row['Predicted Topic']), axis=1
            )
            st.write("**Predicted Headlines for the Next 7 Days**")
            st.dataframe(future_headlines[['Date', 'Headline', 'Predicted Sentiment']])

            st.subheader("📈 Predicted Headline Sentiment Trend")
            sentiment_trend = pd.DataFrame({
                'Date': headline_result['future_dates'],
                'Predicted Sentiment': headline_result['future_sentiment']
            }).set_index('Date')
            st.line_chart(sentiment_trend['Predicted Sentiment'])
    except Exception as e:
        st.error(f"Failed to predict headlines: {e}")
else:
    st.warning("No merged social and news data available for headline prediction.")
st.subheader("📈 Predict Engagement (Regression)")
try:
    with st.spinner("Training regression model..."):
        result = preprocess_and_train(filtered_df, model_choice)
    if result:
        st.success(f"✅ {model_choice} model trained successfully!")
        st.write(f"**R² Score**: {result['r2_score']:.2f}")
        st.write(f"**RMSE**: {result['rmse']:.2f}")
        chart_df = result['X_test'].copy()
        chart_df['Predicted Engagement'] = result['y_pred']
        chart_df['Actual Engagement'] = result['y_test'].values

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(chart_df['Actual Engagement'], chart_df['Predicted Engagement'], alpha=0.6, edgecolors='w',
                   color='tab:purple')
        ax.plot([chart_df['Actual Engagement'].min(), chart_df['Actual Engagement'].max()],
                [chart_df['Actual Engagement'].min(), chart_df['Actual Engagement'].max()], 'r--', lw=2,
                label='Perfect Prediction')
        ax.set_title('Predicted vs Actual Engagement')
        ax.set_xlabel('Actual Engagement')
        ax.set_ylabel('Predicted Engagement')
        ax.legend()
        ax.grid(True)

        st.pyplot(fig)


except Exception as e:
    st.error(f"Regression model error: {e}")

st.subheader("🔍 Sample Posts")
display_cols = [col for col in ['created_at', 'text', 'sentiment', 'engagement'] if col in filtered_df.columns]
st.dataframe(filtered_df[display_cols].tail(10))

st.download_button("📥 Download Data", filtered_df.to_csv(index=False), file_name="filtered_topic_data.csv")
