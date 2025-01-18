import streamlit as st
import pandas as pd
import numpy as np
from binance.client import Client
import talib
import pandas_ta as ta
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
from lightgbm import LGBMClassifier
from sklearn.ensemble import IsolationForest
import plotly.graph_objects as go
from collections import Counter

# إعداد واجهة API لموقع Binance
API_KEY = 'G9KZLu0K1tfL4EeXLRdkpGVp4BY2eZJeqo1BKbS8jmFuK1nYbaSzp9alF1sFA57p'
API_SECRET = '6ZyQMlUZ2MhXWv3OiBOWLFOfILqAA5odVduOqVlY87muiXxDhRVan3IBDDs9viG5'
client = Client(API_KEY, API_SECRET)

# عنوان التطبيق
st.title("تحليل العملات الرقمية المتقدم")

# بيانات تسجيل الدخول
users = {
    "marwan": "marwan2025",
    "admin": "1234",
    "user1": "password1",
    "user2": "password2"
}

# وظيفة التحقق من تسجيل الدخول
def login(username, password):
    if username in users and users[username] == password:
        return True
    return False

# صفحة تسجيل الدخول
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    # صفحة تسجيل الدخول
    st.subheader("تسجيل الدخول")
    username = st.text_input("اسم المستخدم")
    password = st.text_input("كلمة المرور", type="password")
    if st.button("تسجيل الدخول"):
        if login(username, password):
            st.session_state["authenticated"] = True
            st.success("تم تسجيل الدخول بنجاح! يرجى الانتظار...")
            # تأخير بسيط لمحاكاة إعادة التحميل
            st.session_state["reload"] = True
else:
    # التحقق من إعادة التوجيه بعد تسجيل الدخول
    if "reload" in st.session_state and st.session_state["reload"]:
        del st.session_state["reload"]
        st.query_params["authenticated"] = "true"

    # عرض التطبيق إذا كان المستخدم مصادقًا
    st.subheader("مرحبًا بك في واجهة تحليل العملات الرقمية")
    if st.button("تسجيل الخروج"):
        st.session_state["authenticated"] = False
        st.query_params["authenticated"] = "false"
        st.warning("تم تسجيل الخروج بنجاح!")

    # جلب بيانات العملات من Binance
    def get_binance_data(symbol, interval, lookback):
        klines = client.get_historical_klines(symbol, interval, lookback)
        columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']
        df = pd.DataFrame(klines, columns=columns)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df

    # تحليل المؤشرات الفنية
    @st.cache_data
    def analyze_indicators(df):
        # إضافة مؤشر القوة النسبية (RSI)
        df['RSI'] = talib.RSI(df['close'], timeperiod=14)

        # إضافة مؤشر MACD
        df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)

        # حساب Stochastic RSI
        stoch_rsi = ta.stochrsi(df['close'], length=14)
        if stoch_rsi is not None and not stoch_rsi.empty:
            df['Stoch_RSI_K'] = stoch_rsi.iloc[:, 0]  # أول عمود عادةً يكون %K
            df['Stoch_RSI_D'] = stoch_rsi.iloc[:, 1]  # ثاني عمود عادةً يكون %D

        # المتوسطات المتحركة
        df['SMA_50'] = talib.SMA(df['close'], timeperiod=50)
        df['SMA_200'] = talib.SMA(df['close'], timeperiod=200)

        # بولينجر باند
        df['upper_band'], df['middle_band'], df['lower_band'] = talib.BBANDS(df['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)

        # ATR لحساب التقلبات
        df['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)

        # نسبة التغير والمعدلات الملساء
        df['pct_change'] = df['close'].pct_change()
        df['close_smoothed'] = df['close'].rolling(window=10).mean()

        # CMF و Bollinger Band Width
        df['CMF'] = ta.cmf(df['high'], df['low'], df['close'], df['volume'], length=20)
        df['BB_width'] = df['upper_band'] - df['lower_band']

        # ميزات زمنية
        df['week_of_month'] = df.index.day // 7
        return df

    # إزالة القيم الشاذة
    def remove_outliers(df, columns):
        iso = IsolationForest(contamination=0.01, random_state=42)
        outliers = iso.fit_predict(df[columns])
        return df[outliers != -1]

    # تحليل تعلم الآلة
    @st.cache_data
    def machine_learning_analysis(df):
        # تنظيف البيانات
        df = df.dropna(subset=['RSI', 'MACD', 'MACD_signal', 'SMA_50', 'SMA_200', 'close']).copy()
        df['target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)

        # إزالة القيم الشاذة
        df = remove_outliers(df, ['close', 'volume'])

        # إضافة ميزات جديدة
        df['SMA_ratio'] = df['SMA_50'] / df['SMA_200']
        df['day_of_week'] = df.index.dayofweek

        # ميزات الإدخال
        features = df[['RSI', 'MACD', 'SMA_ratio', 'day_of_week', 'pct_change', 'close_smoothed', 'ATR', 'CMF', 'BB_width']].dropna()
        target = df['target']

        features, target = features.align(target, join='inner', axis=0)

        # التحقق من عدد العينات
        if len(features) < 10:
            st.warning("عدد البيانات المتاحة غير كافٍ لتحليل تعلم الآلة.")
            return None, None

        # معيارية البيانات
        scaler = MinMaxScaler()
        features = scaler.fit_transform(features)

        # تقسيم البيانات
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

        # عرض نسبة الفئات
        st.write("Distribution before resampling:", Counter(y_train))

        # تحسين التوازن
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)

        # تدريب النموذج
        model = LGBMClassifier(
        random_state=42,
        verbose=-1,
        n_estimators=500,
        learning_rate=0.05,
        max_depth=7,
        num_leaves=31
        )
        model.fit(X_train, y_train)

        # التنبؤ وحساب الدقة
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        return accuracy, model

    # اختيار العملة والتحليل
    coins = [symbol['symbol'] for symbol in client.get_exchange_info()['symbols'] if symbol['status'] == 'TRADING']
    selected_coin = st.selectbox("اختر العملة للتحليل", coins)
    interval = st.selectbox("اختر الإطار الزمني", [Client.KLINE_INTERVAL_1HOUR, Client.KLINE_INTERVAL_4HOUR, Client.KLINE_INTERVAL_1DAY])

    if selected_coin:
        st.subheader(f"تحليل العملة: {selected_coin}")
        with st.spinner("جاري جلب البيانات وتحليلها..."):
            df = get_binance_data(selected_coin, interval, "1 years ago UTC")
            if not df.empty:
                analyzed_data = analyze_indicators(df)
                if analyzed_data is not None and not analyzed_data.empty:
                    accuracy, model = machine_learning_analysis(analyzed_data)
                    if accuracy is not None and model is not None:
                        st.subheader("الرسوم البيانية والمؤشرات الفنية")
                        st.plotly_chart(go.Figure(data=[
                            go.Candlestick(x=analyzed_data.index, open=analyzed_data['open'], high=analyzed_data['high'], low=analyzed_data['low'], close=analyzed_data['close'])
                        ]))
                        st.line_chart(analyzed_data[['RSI']], use_container_width=True)
                        st.line_chart(analyzed_data[['MACD', 'MACD_signal']], use_container_width=True)
                        st.line_chart(analyzed_data[['SMA_50', 'SMA_200']], use_container_width=True)
                        st.line_chart(analyzed_data[['upper_band', 'lower_band']], use_container_width=True)

                        st.subheader("التحليل الفني والتوصيات")
                        st.info(f"دقة النموذج التنبؤي: {accuracy:.2f}")

                        rsi_value = analyzed_data['RSI'].iloc[-1]
                        macd_value = analyzed_data['MACD'].iloc[-1]
                        macd_signal = analyzed_data['MACD_signal'].iloc[-1]
                        sma_50 = analyzed_data['SMA_50'].iloc[-1]
                        sma_200 = analyzed_data['SMA_200'].iloc[-1]

                        if rsi_value > 70:
                            st.warning(f"RSI: {rsi_value:.2f} (شراء مفرط)")
                        elif rsi_value < 30:
                            st.success(f"RSI: {rsi_value:.2f} (بيع مفرط)")
                        else:
                            st.info(f"RSI: {rsi_value:.2f} (حركة عادية)")

                        if macd_value > macd_signal:
                            st.success(f"MACD: {macd_value:.2f} (اتجاه صاعد)")
                        else:
                            st.warning(f"MACD: {macd_value:.2f} (اتجاه هابط)")

                        if sma_50 > sma_200:
                            st.success("تقاطع إيجابي: السعر في اتجاه صاعد")
                        else:
                            st.warning("تقاطع سلبي: السعر في اتجاه هابط")
                    else:
                        st.warning("عدد البيانات المتاحة غير كافٍ لإجراء التحليل التنبؤي.")
                else:
                    st.warning("فشل تحليل البيانات، يرجى التحقق من البيانات.")
            else:
                st.error("لا توجد بيانات كافية للعملة المختارة.")
