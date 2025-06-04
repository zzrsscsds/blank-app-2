# Prophet Forecasting on Full Dataset (Global Time Series)
st.subheader("📅 48-Hour Engagement Forecast (Global)")
try:
    from prophet import Prophet

    global_ts = combined_df.copy()
    global_ts['created_at'] = pd.to_datetime(global_ts['created_at'], errors='coerce')
    global_ts = global_ts.dropna(subset=['created_at', 'engagement'])
    global_ts = global_ts.groupby(global_ts['created_at'].dt.floor('h')).agg({
        'engagement': 'sum'
    }).reset_index()

    if len(global_ts) >= 20:
        df_prophet = global_ts.rename(columns={
            'created_at': 'ds',
            'engagement': 'y'
        })

        prophet_model = Prophet()
        prophet_model.fit(df_prophet)

        future = prophet_model.make_future_dataframe(periods=48, freq='h')
        forecast = prophet_model.predict(future)

        st.subheader("🔮 Global Forecast of Engagement (Next 48 Hours)")
        fig1 = prophet_model.plot(forecast)
        st.pyplot(fig1)

        st.subheader("📈 Components of Global Trend")
        fig2 = prophet_model.plot_components(forecast)
        st.pyplot(fig2)
    else:
        st.warning("📉 Not enough global time series data for Prophet prediction. Try with more complete dataset.")

except ImportError:
    st.error("Prophet library not found. Please run: pip install prophet")
except Exception as e:
    st.error(f"Prophet model error: {e}")

# Enhanced Regression Features for Engagement Prediction
st.subheader("📈 Predicting Engagement (Enhanced Features)")
filtered_df = filtered_df.rename(columns={"created_at": "timestamp"})
try:
    with st.spinner("Training enhanced regression model..."):
        filtered_df['hour'] = filtered_df['timestamp'].dt.hour
        filtered_df['is_weekend'] = filtered_df['timestamp'].dt.dayofweek >= 5
        filtered_df['text_length'] = filtered_df['text'].apply(len)
        filtered_df['hashtag_count'] = filtered_df['text'].apply(lambda x: x.count('#'))
        filtered_df['is_media'] = filtered_df['text'].str.contains('https://t.co', na=False).astype(int)

        features = ['sentiment', 'text_length', 'hashtag_count', 'is_media', 'hour', 'is_weekend']
        X = filtered_df[features].fillna(0)
        y = filtered_df['engagement'].fillna(0)

        if len(X) < 10:
            raise ValueError("Not enough data for regression training (min 10 rows required).")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.success("✅ Enhanced Model trained successfully!")
        st.write(f"**R² Score**: {r2_score(y_test, y_pred):.2f}")
        st.write(f"**RMSE**: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")

        chart_df = X_test.copy()
        chart_df['Predicted Engagement'] = y_pred
        chart_df['Actual Engagement'] = y_test.values

        st.line_chart(chart_df[['Predicted Engagement', 'Actual Engagement']])

except Exception as e:
    st.error(f"❌ Enhanced model error: {e}")
