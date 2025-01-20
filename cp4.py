import streamlit as st
import pandas as pd
import numpy as np
import pandas_ta as ta
import google.generativeai as genai
import os
from dotenv import load_dotenv
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import time
nltk.download('vader_lexicon')

load_dotenv()

# إعداد مفتاح API لـ Google Gemini
GOOGLE_API_KEY = "AIzaSyCfVZg-ZmATsBzrdI5mxvk_YdIA0YJXtMo"
if not GOOGLE_API_KEY:
    st.error("API key لـ Google Gemini لم يتم العثور عليه! تأكد من تعيينه كمتغير بيئة.")
    st.stop()
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-pro')

# وظيفة لجلب بيانات العملة من CoinGecko
@st.cache_data(ttl=300)
def fetch_coingecko_data(pair, timeframe):
    try:
        pair = pair.lower().replace("/", "-")
        if timeframe == "1m":
            timeframe = "1"
        elif timeframe == "5m":
           timeframe = "5"
        elif timeframe == "15m":
           timeframe = "15"
        elif timeframe == "1h":
           timeframe ="60"
        elif timeframe =="4h":
           timeframe = "240"
        elif timeframe == "1d":
            timeframe = "1440"
        else:
          timeframe = "1440"
        url = f"https://api.coingecko.com/api/v3/coins/{pair}/ohlc?vs_currency=usd&days=1&interval={timeframe}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        ohlcv = response.json()
        if not ohlcv:
            st.warning(f"لا توجد بيانات لزوج العملات {pair} في CoinGecko.")
            return None
        data = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close"])
        data["timestamp"] = pd.to_datetime(data["timestamp"], unit="ms")
        data.set_index("timestamp", inplace=True)
        return data
    except requests.exceptions.RequestException as e:
        if e.response.status_code == 429:
            st.error(f"حدث خطأ 429: تجاوزت الحد المسموح به للطلبات. يرجى الانتظار والمحاولة مرة أخرى.")
            time.sleep(1)  # انتظر ثانية واحدة قبل المحاولة مرة أخرى
            return fetch_coingecko_data(pair, timeframe) #حاول مرة اخرى بعد الانتظار
        else:
           st.error(f"حدث خطأ في الاتصال بالإنترنت أثناء جلب بيانات الأسعار من CoinGecko: {e}")
           return None
    except Exception as e:
       st.error(f"حدث خطأ أثناء جلب بيانات الأسعار من CoinGecko: {e}")
       return None


# تحليل البيانات باستخدام EMA
def analyze_ema(data):
    try:
        data["EMA_20"] = ta.ema(data["close"], length=20)
        data["EMA_50"] = ta.ema(data["close"], length=50)
        data["EMA_100"] = ta.ema(data["close"], length=100)
        data["EMA_200"] = ta.ema(data["close"], length=200)
        recommendation = ""
        if data["close"].iloc[-1] > data["EMA_200"].iloc[-1]:
            recommendation = "السعر أعلى من EMA 200. الاتجاه صاعد."
        else:
            recommendation = "السعر أقل من EMA 200. الاتجاه هابط."
        return recommendation
    except Exception as e:
        return f"خطأ أثناء تحليل EMA: {e}"

# تحليل البيانات باستخدام RSI
def analyze_rsi(data):
    try:
        data["RSI"] = ta.rsi(data["close"], length=14)
        current_rsi = data["RSI"].iloc[-1]
        recommendation = ""
        if current_rsi < 30:
            recommendation = "السعر أقل من قيمته الحقيقية. فرصة شراء محتملة."
        elif current_rsi > 70:
            recommendation = "السعر أعلى من قيمته الحقيقية. احتمال تصحيح السعر."
        else:
            recommendation = "السعر في نطاق طبيعي."
        return f"قيمة RSI الحالية: {current_rsi:.2f}. {recommendation}"
    except Exception as e:
        return f"خطأ أثناء تحليل RSI: {e}"

# تحليل البيانات باستخدام Price Action
def analyze_price_action(data):
    try:
        range_high = data["high"].max()
        range_low = data["low"].min()
        range_mid = (range_high + range_low) / 2
        return f"أعلى قيمة: {range_high}, أقل قيمة: {range_low}, متوسط النطاق: {range_mid:.2f}"
    except Exception as e:
        return f"خطأ أثناء تحليل حركة الأسعار: {e}"

# تحليل معدل التمويل (تمثيلي)
def analyze_funding_rate(pair):
    # ملاحظة: ليس لدى CoinGecko بيانات معدل التمويل، لذلك نعرض رسالة توضيحية
    return "لا تتوفر بيانات معدل التمويل من CoinGecko."
# وظيفة لحساب نقاط الارتكاز
def calculate_pivot_points(data):
    try:
        pivot_point = (data['high'].iloc[-1] + data['low'].iloc[-1] + data['close'].iloc[-1]) / 3
        resistance1 = (2 * pivot_point) - data['low'].iloc[-1]
        support1 = (2 * pivot_point) - data['high'].iloc[-1]
        resistance2 = pivot_point + (data['high'].iloc[-1] - data['low'].iloc[-1])
        support2 = pivot_point - (data['high'].iloc[-1] - data['low'].iloc[-1])
        return pivot_point, resistance1, support1, resistance2, support2
    except Exception as e:
      st.error(f"خطأ أثناء حساب نقاط الارتكاز: {e}")
      return None, None, None, None, None

# وظيفة لتحديد كسر المقاومة وتحديد الاتجاه
def analyze_breakout(data, pivot_point, resistance1, support1, resistance2, support2):
    try:
        last_close = data['close'].iloc[-1]
        last_ema = data['EMA_200'].iloc[-1]
        recommendation = ""
        if last_close > resistance1:
           if last_close > resistance2:
            recommendation = "تم كسر مستوى المقاومة الثاني. اتجاه صعود قوي."
           else:
            recommendation = "تم كسر مستوى المقاومة الأول. اتجاه صعود محتمل."
        elif last_close < support1:
            recommendation = "تم كسر مستوى الدعم الأول. اتجاه هبوط محتمل."
        elif last_close < support2:
            recommendation = "تم كسر مستوى الدعم الثاني. اتجاه هبوط قوي"
        else:
             recommendation = "السعر ضمن نطاق الدعم والمقاومة."
        
        return recommendation
    except Exception as e:
      st.error(f"خطأ أثناء تحليل كسر المقاومة: {e}")
      return ""

# وظيفة لجلب بيانات الأخبار والمشاعر من الإنترنت
@st.cache_data(ttl=3600)
def fetch_news_and_sentiment(pair):
    try:
        search_query = f"news sentiment {pair}"
        url = f"https://www.google.com/search?q={search_query}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # رفع استثناء للأخطاء
        soup = BeautifulSoup(response.content, 'html.parser')
        # استخدام محددات أكثر تحديدًا للعثور على العناوين
        news_headlines_elements = soup.select('div.yuRUbf h3')  # محدد لتحديد العناوين بشكل أفضل
        news_headlines = [h3.text for h3 in news_headlines_elements]
        
        if news_headlines:
            sentiment_analyzer = SentimentIntensityAnalyzer()
            sentiments = [sentiment_analyzer.polarity_scores(headline)['compound'] for headline in news_headlines]
            average_sentiment = np.mean(sentiments) if sentiments else 0
            return news_headlines, average_sentiment
        else:
            st.warning("لم يتم العثور على أي عناوين إخبارية.")
            return [], 0
    except requests.exceptions.RequestException as e:
        st.error(f"حدث خطأ في الاتصال بالإنترنت أثناء جلب الأخبار: {e}")
        return [], 0
    except Exception as e:
        st.error(f"حدث خطأ أثناء جلب الأخبار والمشاعر: {e}")
        return [], 0

# وظيفة لرسم الرسوم البيانية
def plot_indicators(data, pair, pivot_point, resistance1, support1, resistance2, support2):
  try:
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, subplot_titles=(f'Price Action for {pair}', 'RSI', 'EMA'))
        fig.add_trace(go.Candlestick(x=data.index, open=data['open'], high=data['high'], low=data['low'], close=data['close'], name = "Candlestick"), row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data["RSI"], mode='lines', name = "RSI"), row=2, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data["EMA_20"], mode='lines', name = "EMA_20"), row=3, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data["EMA_50"], mode='lines', name = "EMA_50"), row=3, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data["EMA_100"], mode='lines', name = "EMA_100"), row=3, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data["EMA_200"], mode='lines', name = "EMA_200"), row=3, col=1)

        # إضافة خطوط نقاط الارتكاز
        if pivot_point is not None:
           fig.add_trace(go.Scatter(x=data.index, y=[pivot_point] * len(data.index), mode='lines', name="Pivot Point", line=dict(color="black", dash="dash")), row=1, col=1)
           fig.add_trace(go.Scatter(x=data.index, y=[resistance1] * len(data.index), mode='lines', name="R1", line=dict(color="red", dash="dash")), row=1, col=1)
           fig.add_trace(go.Scatter(x=data.index, y=[support1] * len(data.index), mode='lines', name="S1", line=dict(color="green", dash="dash")), row=1, col=1)
           fig.add_trace(go.Scatter(x=data.index, y=[resistance2] * len(data.index), mode='lines', name="R2", line=dict(color="red", dash="dash")), row=1, col=1)
           fig.add_trace(go.Scatter(x=data.index, y=[support2] * len(data.index), mode='lines', name="S2", line=dict(color="green", dash="dash")), row=1, col=1)
        fig.update_layout(xaxis_rangeslider_visible=False, height=800, title_text=f"تحليل {pair}")
        st.plotly_chart(fig, use_container_width=True)
  except Exception as e:
      st.error(f"حدث خطأ أثناء رسم الرسوم البيانية: {e}")

# إرسال النتائج إلى الذكاء الاصطناعي والحصول على الرد
def send_to_ai(content):
    try:
        response = model.generate_content(content)
        return response.text
    except Exception as e:
        return f"خطأ أثناء الاتصال بـ Gemini API: {e}"

# إعداد واجهة المستخدم
def main():
    st.title("برنامج التحليل الفني للعملات الرقمية")

    # اختيار العملة
    st.sidebar.header("إعدادات البرنامج")
    crypto_pair = st.sidebar.text_input("اختر زوج العملات (مثل BTC/USDT):", "BTC/USDT")

    # اختيار الفترة الزمنية
    timeframe = st.sidebar.selectbox("اختر الإطار الزمني:", ["1m", "5m", "15m", "1h", "4h", "1d"], index=4)

    # تحميل البيانات
    st.write(f"جلب البيانات لـ {crypto_pair} ...")
    data = fetch_coingecko_data(crypto_pair, timeframe)

    if data is not None:
        # تحليل البيانات باستخدام EMA
        st.subheader("نتائج التحليل باستخدام مؤشر EMA")
        ema_analysis = analyze_ema(data)
        st.write(ema_analysis)

        # تحليل البيانات باستخدام RSI
        st.subheader("نتائج التحليل باستخدام مؤشر RSI")
        rsi_analysis = analyze_rsi(data)
        st.write(rsi_analysis)

        # تحليل البيانات باستخدام Price Action
        st.subheader("نتائج تحليل حركة الأسعار (Price Action)")
        price_action_analysis = analyze_price_action(data)
        st.write(price_action_analysis)
        
        # حساب نقاط الارتكاز
        pivot_point, resistance1, support1, resistance2, support2 = calculate_pivot_points(data)
        if pivot_point is not None:
            st.subheader("نقاط الارتكاز ومستويات الدعم والمقاومة")
            st.write(f"نقطة الارتكاز: {pivot_point:.2f}")
            st.write(f"المقاومة 1: {resistance1:.2f}")
            st.write(f"الدعم 1: {support1:.2f}")
            st.write(f"المقاومة 2: {resistance2:.2f}")
            st.write(f"الدعم 2: {support2:.2f}")

        # تحليل كسر المقاومة
        st.subheader("تحليل كسر المقاومة/الدعم")
        breakout_analysis = analyze_breakout(data, pivot_point, resistance1, support1, resistance2, support2)
        st.write(breakout_analysis)

        # رسم الرسوم البيانية
        plot_indicators(data, crypto_pair, pivot_point, resistance1, support1, resistance2, support2)

        # تحليل البيانات باستخدام Funding Rate
        st.subheader("نتائج التحليل باستخدام معدل التمويل (Funding Rate)")
        funding_rate_analysis = analyze_funding_rate(crypto_pair)
        st.write(funding_rate_analysis)
        # جلب بيانات الأخبار والمشاعر من الإنترنت
        st.subheader("تحليل الأخبار والمشاعر")
        news_headlines, average_sentiment = fetch_news_and_sentiment(crypto_pair)
        if news_headlines:
          st.write("العناوين الإخبارية:")
          for headline in news_headlines:
            st.write(f"- {headline}")
          st.write(f"متوسط الشعور: {average_sentiment:.2f}")
        else:
            st.write("لم يتم العثور على أي عناوين إخبارية أو حدث خطأ.")
        # إرسال النتائج إلى الذكاء الاصطناعي
        st.subheader("إرسال النتائج إلى الذكاء الاصطناعي")
        ai_content = f"EMA Analysis: {ema_analysis}\nRSI Analysis: {rsi_analysis}\nPrice Action Analysis: {price_action_analysis}\nFunding Rate Analysis: {funding_rate_analysis}\nPivot Points: {pivot_point}, R1: {resistance1}, S1: {support1}, R2: {resistance2}, S2: {support2}, Breakout Analysis: {breakout_analysis} "
        if news_headlines:
           ai_content +=f" ,Average Sentiment: {average_sentiment} , News Headlines : {news_headlines}"
        else:
           ai_content +=" , News data not available"
        ai_response = send_to_ai(ai_content)
        st.write("تم إرسال النتائج بنجاح.")
        st.subheader("رد الذكاء الاصطناعي")
        st.write(ai_response)
    else:
        st.write("تعذر جلب البيانات. تحقق من زوج العملات أو الإعدادات.")

if __name__ == "__main__":
    main()
