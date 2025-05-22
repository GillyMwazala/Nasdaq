import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# App configurations
st.set_page_config(
    page_title="Trading View AI Analyst",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS to make it look more like TradingView but with better visibility
st.markdown("""
<style>
    /* General styling */
    .main {
        background-color: #131722;
        color: #ffffff; /* Brighter text for better readability */
    }
    
    /* Headers styling for better sectioning */
    h1, h2, h3 {
        color: #f8f9fa;
        font-weight: 600;
        margin-bottom: 0.8rem;
        padding-bottom: 0.3rem;
        border-bottom: 1px solid #2a2e39;
    }
    
    h1 {
        color: #29b6f6;
        font-size: 2.2rem;
    }
    
    h2 {
        color: #ffeb3b;
        font-size: 1.8rem;
    }
    
    h3 {
        color: #f06292;
        font-size: 1.4rem;
    }
    
    /* Button styling for better visibility */
    .stButton>button {
        background-color: #2962ff;
        color: white;
        font-weight: 600;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 4px;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        background-color: #1e40af;
        box-shadow: 0 0 15px rgba(41, 98, 255, 0.5);
    }
    
    /* Metrics styling for dashboard numbers */
    .stMetric {
        background-color: #1e222d;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2962ff;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    
    /* Indicator cards for trading signals */
    .css-1r6slb0 {  /* Metrics container */
        background-color: #1e222d;
        border-radius: 10px;
        padding: 12px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #2a2e39;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 20px;
        color: #ffffff;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #2962ff;
        color: white;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #1e222d;
        border-right: 1px solid #2a2e39;
        padding: 1.5rem;
    }
    
    /* Dataframe styling for better data readability */
    .dataframe {
        background-color: #1e222d;
        font-family: 'Courier New', monospace;
        border-collapse: collapse;
    }
    
    .dataframe th {
        background-color: #2962ff;
        color: white;
        padding: 8px;
        text-align: center;
    }
    
    .dataframe td {
        padding: 6px;
        border: 1px solid #2a2e39;
    }
    
    /* Warning/Status banner */
    .warning-banner {
        background-color: rgba(255, 87, 34, 0.2);
        border-left: 4px solid #ff5722;
        padding: 10px 15px;
        border-radius: 0 5px 5px 0;
        margin: 10px 0;
    }
    
    .success-banner {
        background-color: rgba(0, 200, 83, 0.2);
        border-left: 4px solid #00c853;
        padding: 10px 15px;
        border-radius: 0 5px 5px 0;
        margin: 10px 0;
    }
    
    /* Button styling */
    div.stButton > button {
        background-color: #2962ff;
        color: white;
        border-radius: 4px;
        border: none;
        padding: 10px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        font-weight: 600;
    }
    
    div.stButton > button:hover {
        background-color: #1e53e5;
    }
    
    /* Metrics styling */
    [data-testid="stMetric"] {
        background-color: #2a2e39;
        border-radius: 5px;
        padding: 15px !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
        color: #ffffff !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 14px !important;
        color: #ffffff !important;
    }
    
    [data-testid="stMetricValue"] {
        font-size: 28px !important;
        font-weight: 600 !important;
        color: #ffffff !important;
    }
    
    [data-testid="stMetricDelta"] {
        color: #ffffff !important;
    }
    
    /* Info block styling */
    .stAlert {
        background-color: #2a2e39;
        border-radius: 4px;
        border-left-color: #2962ff;
        color: #ffffff !important;
    }
    
    /* Text styling */
    h1, h2, h3 {
        color: #ffffff !important;
    }
    
    .css-10trblm {
        color: #ffffff !important;
    }
    
    .stMarkdown p {
        color: #ffffff !important;
    }
    
    /* Container styling */
    [data-testid="stVerticalBlock"] > div {
        border-radius: 5px;
        padding: 10px 0;
    }
    
    /* All text elements should be white for visibility */
    span, p, div, h1, h2, h3, h4, h5, h6, li, a {
        color: #ffffff !important;
    }
    
    /* Ensure selectbox text is visible */
    .st-bq {
        background-color: #2a2e39;
        color: white !important;
    }
    
    /* Fix code highlighting */
    code {
        color: #56d1d8 !important;
    }
    /* Additional explicit text color for Streamlit widgets */
    .stTextInput input, .stTextArea textarea, .stNumberInput input {
        color: #ffffff !important;
        background-color: #2a2e39 !important; /* Ensure input backgrounds are also dark */
    }
    .stSelectbox div[data-baseweb="select"] > div, /* Selected value in selectbox */
    .stMultiSelect div[data-baseweb="select"] > div /* Selected values in multiselect */
     {
        color: #ffffff !important;
    }
    
    /* Dropdown menu items for selectbox/multiselect */
    div[data-baseweb="popover"] ul[role="listbox"] li {
        background-color: #1e222d !important;
        color: #ffffff !important;
    }
    div[data-baseweb="popover"] ul[role="listbox"] li:hover {
        background-color: #2962ff !important;
    }

    /* Ensure radio and checkbox labels are white */
    .stRadio label, .stCheckbox label {
        color: #ffffff !important;
    }
    
    /* Ensure st.info/success/warning/error text is explicitly white if not covered by .stAlert */
    .stAlert > div > div { /* Targeting the inner div that often holds the text */
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar with professional styling
with st.sidebar:
    st.title("🚀 AI Trading View Pro")
    
    # Asset selection with more visual appeal
    st.header("💹 Market Selection")
    col1, col2 = st.columns([3, 1])
    with col1:
        symbol = st.selectbox(
            "Select Asset",
            options=["^IXIC"],  # MODIFIED: NASDAQ-100 Ticker
            format_func=lambda x: "NASDAQ-100" if x == "^IXIC" else x, # MODIFIED: Display friendly name
            index=0
        )
    with col2:
        st.markdown("###") # Spacing
        if symbol == "^IXIC": # MODIFIED: Check for NASDAQ ticker
            st.markdown("📈") # MODIFIED: Generic chart icon for NASDAQ
        # Add other conditions if more symbols are added in the future
        # else:
        #     st.markdown("❓") # Default icon
    
    # Time frame selection with better organization
    st.header("⏱️ Time Settings")
    col1_time, col2_time = st.columns(2) # Renamed to avoid conflict with outer col1, col2
    with col1_time:
        timeframe = st.selectbox(
            "Timeframe",
            options=["1d", "1h", "15m", "5m", "1m"],
            index=0
        )
    # with col2_time: # This col2 was for period_options which is defined later, removing for clarity
    #     # Dynamic period options based on timeframe
    #     # This period_options definition was overwritten later, so removing this instance
    #     pass 
    
    # Chart type
    chart_type = st.radio(
        "Chart Type",
        options=["Candlestick", "OHLC", "Line"],
        index=0
    )
    
    # Period of data to fetch
    period_options = {
        "1d": ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"], # Added max, consistent with later definition
        "1h": ["2d", "5d", "1mo", "3mo", "6mo"], # Adjusted options
        "15m": ["1d", "5d", "1mo", "3mo"], # Adjusted options
        "5m": ["1d", "5d", "7d"], # Adjusted options
        "1m": ["1d", "2d", "5d"]
    }
    
    period = st.selectbox(
        "Select Period",
        options=period_options.get(timeframe, ["1mo", "3mo", "6mo", "1y", "max"]), # Default to broader options
        index=0
    )
    
    # Volume display
    show_volume = st.checkbox("Show Volume", True)
    
    # Analysis Options
    st.header("AI Analysis")
    
    # Analysis type selection
    analysis_type = st.multiselect(
        "Select Analysis",
        options=[
            "Trend Analysis", 
            "Support/Resistance", 
            "Fair Value Gaps", 
            "Price Action", 
            "Market Sentiment",
            "Entry/Exit Points",
            "Volume Analysis",
            "Liquidity Zones"
        ],
        default=["Trend Analysis", "Support/Resistance", "Entry/Exit Points", "Price Action", "Market Sentiment"]
    )
    
    # Run analysis button
    run_analysis = st.button("Analyze Market", use_container_width=True)

# Helper Functions

@st.cache_data(ttl=60)  # Cache data for 1 minute only to get fresh updates
def fetch_market_data(ticker, interval, period):
    """Fetch market data with error handling"""
    try:
        st.info(f"Fetching latest market data for {ticker}...")
        data = yf.download(tickers=ticker, interval=interval, period=period)
        if data.empty:
            st.error(f"No data available for {ticker}")
            return None
        st.success(f"Successfully loaded latest {ticker} data")
        return data
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

def find_support_resistance(df, n_levels=5):
    """Find support and resistance levels using price extremes"""
    if 'Close' not in df.columns or df.empty:
        return [], []
        
    prices = df['Close'].values
    
    supports = []
    resistances = []
    
    for i in range(2, len(prices) - 2):
        if prices[i] < prices[i-1] and prices[i] < prices[i-2] and \
           prices[i] < prices[i+1] and prices[i] < prices[i+2]:
            supports.append(prices[i])
            
        if prices[i] > prices[i-1] and prices[i] > prices[i-2] and \
           prices[i] > prices[i+1] and prices[i] > prices[i+2]:
            resistances.append(prices[i])
    
    supports = sorted(list(set(supports))) # Remove duplicates
    resistances = sorted(list(set(resistances)), reverse=True) # Remove duplicates
    
    clustered_supports = []
    clustered_resistances = []
    
    if supports:
        threshold = (max(prices) - min(prices)) * 0.01 if (max(prices) - min(prices)) > 0 else 0.01
        current_level = supports[0]
        current_cluster = [current_level]
        for level in supports[1:]:
            if abs(level - current_level) < threshold:
                current_cluster.append(level)
            else:
                clustered_supports.append(sum(current_cluster) / len(current_cluster))
                current_level = level
                current_cluster = [current_level]
        if current_cluster: # Add last cluster
            clustered_supports.append(sum(current_cluster) / len(current_cluster))
    
    if resistances:
        threshold = (max(prices) - min(prices)) * 0.01 if (max(prices) - min(prices)) > 0 else 0.01
        current_level = resistances[0]
        current_cluster = [current_level]
        for level in resistances[1:]:
            if abs(level - current_level) < threshold:
                current_cluster.append(level)
            else:
                clustered_resistances.append(sum(current_cluster) / len(current_cluster))
                current_level = level
                current_cluster = [current_level]
        if current_cluster: # Add last cluster
            clustered_resistances.append(sum(current_cluster) / len(current_cluster))
            
    return clustered_supports[:n_levels], clustered_resistances[:n_levels]

def find_fair_value_gaps(df):
    """Identify fair value gaps in price data"""
    if len(df) < 3:
        return []
    
    gaps = []
    
    for i in range(1, len(df) - 1):
        try:
            low_next = df['Low'].iloc[i+1].item() if hasattr(df['Low'].iloc[i+1], 'item') else df['Low'].iloc[i+1]
            high_prev = df['High'].iloc[i-1].item() if hasattr(df['High'].iloc[i-1], 'item') else df['High'].iloc[i-1]
            low_prev = df['Low'].iloc[i-1].item() if hasattr(df['Low'].iloc[i-1], 'item') else df['Low'].iloc[i-1]
            high_next = df['High'].iloc[i+1].item() if hasattr(df['High'].iloc[i+1], 'item') else df['High'].iloc[i+1]
            
            if low_next > high_prev:
                gaps.append({
                    'type': 'bullish', 'datetime': df.index[i],
                    'top': low_next, 'bottom': high_prev,
                    'mid': (low_next + high_prev) / 2
                })
                
            if high_next < low_prev:
                gaps.append({
                    'type': 'bearish', 'datetime': df.index[i],
                    'top': low_prev, 'bottom': high_next,
                    'mid': (low_prev + high_next) / 2
                })
        except (ValueError, AttributeError, TypeError, IndexError) as e:
            st.warning(f"Skipping FVG calculation at index {i} due to: {e}")
            continue
    return gaps

def get_ai_analysis(data, analysis_types, symbol_ticker, timeframe): # MODIFIED: symbol to symbol_ticker
    """Get AI analysis from OpenAI API or fallback to local analysis if API fails"""
    # MODIFIED: Display friendly name for symbol
    display_symbol = "NASDAQ-100" if symbol_ticker == "^IXIC" else symbol_ticker

    try:
        recent_data = data.tail(10).copy()
        price_summary = f"Recent {display_symbol} prices:\n" # MODIFIED: use display_symbol
        
        for idx, row in recent_data.iterrows():
            open_val = float(row['Open'])
            high_val = float(row['High'])
            low_val = float(row['Low'])
            close_val = float(row['Close'])
            vol_val = int(float(row['Volume']))
            timestamp = idx.strftime('%Y-%m-%d %H:%M:%S') if hasattr(idx, 'strftime') else str(idx)
            price_summary += f"- {timestamp}: Open {open_val:.2f}, High {high_val:.2f}, Low {low_val:.2f}, Close {close_val:.2f}, Volume {vol_val}\n"
        
        current_price = float(data['Close'].iloc[-1])
        prev_price = float(data['Close'].iloc[-2]) if len(data['Close']) > 1 else current_price
        price_change = ((current_price / prev_price) - 1) * 100 if prev_price != 0 else 0
        
        high_val = float(data['High'].max())
        low_val = float(data['Low'].min())
        vol_val = int(float(data['Volume'].sum()))
        
        sma9 = float(data['SMA_9'].iloc[-1]) if 'SMA_9' in data.columns and not data['SMA_9'].empty else None
        sma20 = float(data['SMA_20'].iloc[-1]) if 'SMA_20' in data.columns and not data['SMA_20'].empty else None
        sma50 = float(data['SMA_50'].iloc[-1]) if 'SMA_50' in data.columns and not data['SMA_50'].empty else None
        
        macd = float(data['MACD'].iloc[-1]) if 'MACD' in data.columns and not data['MACD'].empty else None
        macd_signal = float(data['MACD_Signal'].iloc[-1]) if 'MACD_Signal' in data.columns and not data['MACD_Signal'].empty else None
        macd_hist = float(data['MACD_Hist'].iloc[-1]) if 'MACD_Hist' in data.columns and not data['MACD_Hist'].empty else None
        
        rsi = float(data['RSI'].iloc[-1]) if 'RSI' in data.columns and not data['RSI'].empty else None
        
        market_stats = (
            f"Current Price: ${current_price:.2f}\n"
            f"24h Change: {price_change:.2f}%\n"
            f"24h High: ${high_val:.2f}\n"
            f"24h Low: ${low_val:.2f}\n"
            f"Volume: {vol_val:,}\n"
        )
        
        supports, resistances = find_support_resistance(data)
        scalar_supports = [float(s.item() if hasattr(s, 'item') else s) for s in supports[:3]]
        scalar_resistances = [float(r.item() if hasattr(r, 'item') else r) for r in resistances[:3]]
        
        support_resistance_info = "Key Support Levels: " + ", ".join([f"${s:.2f}" for s in scalar_supports]) + "\n"
        support_resistance_info += "Key Resistance Levels: " + ", ".join([f"${r:.2f}" for r in scalar_resistances])
        
        fvgs = find_fair_value_gaps(data)
        fvg_info = "Recent Fair Value Gaps:\n"
        if fvgs:
            for fvg in fvgs[-3:]:
                try:
                    fvg_type = fvg['type']
                    bottom = float(fvg['bottom'].item() if hasattr(fvg['bottom'], 'item') else fvg['bottom'])
                    top = float(fvg['top'].item() if hasattr(fvg['top'], 'item') else fvg['top'])
                    mid = float(fvg['mid'].item() if hasattr(fvg['mid'], 'item') else fvg['mid'])
                    fvg_info += f"- {fvg_type.title()} FVG at ${mid:.2f} (range: ${bottom:.2f}-${top:.2f})\n"
                except (KeyError, TypeError, ValueError):
                    continue
        else:
            fvg_info += "No significant fair value gaps detected.\n"
            
        try:
            test_response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hello, this is a test request. Reply with just the word 'ok'."}],
                max_tokens=5
            )
            
            if hasattr(test_response, 'choices') and len(test_response.choices) > 0:
                analysis_request = f"""
                As a professional market analyst, provide insights on {display_symbol} at {timeframe} timeframe.
                {market_stats}
                {support_resistance_info}
                {fvg_info}
                {price_summary}
                For each selected analysis type, provide:
                """ # MODIFIED: use display_symbol
                
                if "Trend Analysis" in analysis_types:
                    analysis_request += """
                    - Trend Analysis: Current trend direction (bullish, bearish, or neutral), trend strength, key trend features, and potential trend continuation or reversal signals. Identify if we're in a larger trend or consolidation. Mention any chart patterns.
                    """
                if "Support/Resistance" in analysis_types:
                    analysis_request += """
                    - Support/Resistance: Identify key price levels where the asset has historically found support or resistance. Analyze the strength of these levels and their significance in the current market context. Focus on the most important levels and their likelihood of holding or breaking.
                    """
                if "Fair Value Gaps" in analysis_types:
                    analysis_request += """
                    - Fair Value Gaps: Identify unfilled price gaps (fair value gaps or imbalances) and explain their significance for the current market structure. Discuss how they might act as magnets for price and their potential as targets.
                    """
                if "Price Action" in analysis_types:
                    analysis_request += """
                    - Price Action: Examine recent candlestick patterns and price behavior. Identify signs of accumulation, distribution, or indecision. Look for candlestick formations that suggest market psychology.
                    """
                if "Market Sentiment" in analysis_types:
                    analysis_request += """
                    - Market Sentiment: Assess the overall market sentiment and institutional positioning. Describe potential institutional intent based on price action and volume. Consider possible manipulation patterns.
                    """
                if "Entry/Exit Points" in analysis_types:
                    analysis_request += """
                    - Entry/Exit Points: Identify specific price levels that could serve as optimal entry and exit points based on current market structure.
                      For each entry point, specify:
                        - The type of entry (e.g., breakout, pullback to support, FVG fill).
                        - Key confirmation signals to watch for (e.g., candlestick pattern, volume spike, indicator crossover).
                        - Conditions that would invalidate this entry setup.
                      Include clear, recommended stop-loss levels (with rationale, e.g., below recent swing low, ATR-based) and at least two profit targets with risk-reward ratios.
                    """
                if "Volume Analysis" in analysis_types:
                    analysis_request += """
                    - Volume Analysis: Examine recent volume patterns and their relationship to price movements. Look for volume spikes, volume divergences, and cumulative volume patterns that might indicate a potential trend reversal or continuation.
                    """
                if "Liquidity Zones" in analysis_types:
                    analysis_request += """
                    - Liquidity Zones: Identify areas where large stop losses or pending orders may be clustered. These areas may be targets for price moves as the market hunts for liquidity. Explain how these zones might influence future price action.
                    """
                analysis_request += """
                Finally, provide a clear, actionable trading recommendation with these elements:
                1. Overall Market Bias: Bullish, Bearish, or Neutral, with a confidence level (Low, Medium, High).
                2. Primary Trade Setup:
                    - Suggested Entry Point(s): Specific price level(s) or zone.
                    - Entry Rationale: Detailed explanation linking to the analysis (e.g., "Entry on pullback to confirmed support at $X, coinciding with bullish divergence on RSI").
                    - Confirmation Signals: What specific chart events or indicator readings would confirm the entry?
                    - Stop Loss: Recommended price level and why (e.g., "SL at $Y, just below the 50-period SMA and recent swing low").
                    - Take Profit Targets: At least 2 price targets (e.g., TP1 at $Z1 targeting nearest resistance, TP2 at $Z2 for further extension).
                    - Risk-Reward Ratio: For each target.
                3. Alternative Scenarios: Briefly mention any secondary setups or what might invalidate the primary view.
                4. Timeframe: Expected duration for the primary setup to play out (e.g., intraday, 1-3 days, 1 week).
                Format your analysis for maximum readability with clear sections, bold text for key terms, and bullet points where appropriate. Be precise and avoid vague statements.
                """
                
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are an expert financial market analyst specializing in technical analysis and trading strategies. Your insights are precise, actionable, and based on sound technical principles. Provide chart analysis like a professional trader would."},
                        {"role": "user", "content": analysis_request}
                    ],
                    max_tokens=1500,
                    temperature=0.3,
                )
                return response.choices[0].message.content
            else:
                st.warning("OpenAI API response invalid. Falling back to local analysis.")
                raise Exception("API response invalid")
                
        except Exception as api_error:
            st.warning(f"OpenAI API call failed: {api_error}. Falling back to local analysis.")
            analysis = f"""
            ## Technical Analysis for {display_symbol} - {timeframe} 
            ### Market Overview
            **Current Price:** ${current_price:.2f}
            **24h Change:** {price_change:.2f}%
            **24h High:** ${high_val:.2f}
            **24h Low:** ${low_val:.2f}
            ### Technical Indicators
            """ # MODIFIED: use display_symbol
            
            if sma9 is not None and sma20 is not None and sma50 is not None:
                analysis += f"""
                **Moving Averages:**
                - SMA 9: ${sma9:.2f}
                - SMA 20: ${sma20:.2f}
                - SMA 50: ${sma50:.2f}
                **Trend Analysis:**
                """
                if sma9 > sma20 and sma20 > sma50: analysis += "- Strong Uptrend - All moving averages aligned bullishly (SMA9 > SMA20 > SMA50)"
                elif sma9 < sma20 and sma20 < sma50: analysis += "- Strong Downtrend - All moving averages aligned bearishly (SMA9 < SMA20 < SMA50)"
                elif sma9 > sma20 and sma20 < sma50: analysis += "- Potential bullish reversal or consolidation."
                elif sma9 < sma20 and sma20 > sma50: analysis += "- Potential bearish reversal or consolidation."
                else: analysis += "- Consolidation/Sideways - No clear trend direction from moving averages"
            
            if macd is not None and macd_signal is not None and macd_hist is not None:
                analysis += f"""
                **MACD Analysis:**
                - MACD Line: {macd:.4f}
                - Signal Line: {macd_signal:.4f}
                - Histogram: {macd_hist:.4f}
                **Momentum:**
                """
                if macd > macd_signal and macd_hist > 0: analysis += "- Bullish momentum is increasing"
                elif macd > macd_signal and macd_hist <= 0: analysis += "- Bullish momentum is emerging" # Corrected condition
                elif macd < macd_signal and macd_hist < 0: analysis += "- Bearish momentum is increasing"
                elif macd < macd_signal and macd_hist >= 0: analysis += "- Bearish momentum is emerging" # Corrected condition
                else: analysis += "- Neutral or transitioning momentum."

            if rsi is not None:
                analysis += f"""
                **RSI Analysis:**
                - Current RSI: {rsi:.2f}
                **Overbought/Oversold:**
                """
                if rsi > 70: analysis += "- Overbought conditions: Potential for a pullback or correction"
                elif rsi < 30: analysis += "- Oversold conditions: Potential for a bounce or recovery"
                else: analysis += f"- RSI in neutral territory ({rsi:.2f})"
            
            analysis += "\n\n### Key Price Levels\n**Support Levels:**\n"
            for s_val in scalar_supports: analysis += f"- ${s_val:.2f} ({( (s_val / current_price) - 1) * 100 if current_price != 0 else 0:.2f}% from current price)\n"
            analysis += "\n**Resistance Levels:**\n"
            for r_val in scalar_resistances: analysis += f"- ${r_val:.2f} ({( (r_val / current_price) - 1) * 100 if current_price != 0 else 0:.2f}% from current price)\n"
            
            analysis += "\n\n### Trade Recommendation\n"
            direction = "Neutral"; confidence = "Low"
            if sma9 is not None and sma20 is not None and sma50 is not None:
                if sma9 > sma20 and (macd is not None and macd > macd_signal) and (rsi is not None and rsi > 50 and rsi < 70):
                    direction = "Bullish"; confidence = "Medium"
                    if sma20 > sma50: confidence = "High"
                elif sma9 < sma20 and (macd is not None and macd < macd_signal) and (rsi is not None and rsi < 50 and rsi > 30):
                    direction = "Bearish"; confidence = "Medium"
                    if sma20 < sma50: confidence = "High"
            
            analysis += f"**Market Direction:** {direction} (Confidence: {confidence})\n**Entry Strategy:**\n"
            if direction == "Bullish":
                entry = current_price * 0.99
                stop_loss = min(scalar_supports[0] if scalar_supports else current_price * 0.95, current_price * 0.95)
                tp1 = current_price * 1.05; tp2 = current_price * 1.10
                rr1 = ((tp1-entry)/(entry-stop_loss)) if (entry-stop_loss) != 0 else 0
                rr2 = ((tp2-entry)/(entry-stop_loss)) if (entry-stop_loss) != 0 else 0
                analysis += f"- **Entry Point:** ${entry:.2f}\n- **Stop Loss:** ${stop_loss:.2f}\n- **Take Profit 1:** ${tp1:.2f} (RR: 1:{rr1:.1f})\n- **Take Profit 2:** ${tp2:.2f} (RR: 1:{rr2:.1f})\n- **Timeframe:** Short to medium-term"
            elif direction == "Bearish":
                entry = current_price * 1.01
                stop_loss = max(scalar_resistances[0] if scalar_resistances else current_price * 1.05, current_price * 1.05)
                tp1 = current_price * 0.95; tp2 = current_price * 0.90
                rr1 = ((entry-tp1)/(stop_loss-entry)) if (stop_loss-entry) != 0 else 0
                rr2 = ((entry-tp2)/(stop_loss-entry)) if (stop_loss-entry) != 0 else 0
                analysis += f"- **Entry Point:** ${entry:.2f}\n- **Stop Loss:** ${stop_loss:.2f}\n- **Take Profit 1:** ${tp1:.2f} (RR: 1:{rr1:.1f})\n- **Take Profit 2:** ${tp2:.2f} (RR: 1:{rr2:.1f})\n- **Timeframe:** Short to medium-term"
            else:
                analysis += "- Wait for clearer market direction.\n- Focus on breakout/breakdown levels.\n- Consider range-bound strategies."
            analysis += "\n\n### Risk Warning\nThis analysis is based on technical indicators only and should not be considered financial advice. Always manage your risk."
            return analysis
    
    except Exception as e:
        st.error(f"Critical error in get_ai_analysis: {e}")
        current_price_fallback = data['Close'].iloc[-1] if not data.empty and 'Close' in data.columns and not data['Close'].empty else "N/A"
        price_change_fallback = "N/A"
        if isinstance(current_price_fallback, (int, float)) and len(data) > 1 and 'Close' in data.columns and len(data['Close']) > 1:
            prev_price_fallback = data['Close'].iloc[-2]
            if isinstance(prev_price_fallback, (int, float)) and prev_price_fallback != 0:
                 price_change_fallback = f"{((current_price_fallback / prev_price_fallback) - 1) * 100:.2f}%"

        return f"""
        ## Basic Market Overview for {display_symbol}
        **Current Price:** ${current_price_fallback:.2f if isinstance(current_price_fallback, (int,float)) else current_price_fallback}
        **24h Change:** {price_change_fallback}
        ### Error Notice
        An error occurred while generating the detailed analysis: {str(e)}
        Please try again later or check different timeframes/symbols.
        """ # MODIFIED: use display_symbol

# Main content
st.title(f"AI Trading View Analysis")

with st.spinner('Fetching market data...'):
    interval_map = {"1m": "1m", "5m": "5m", "15m": "15m", "1h": "60m", "1d": "1d"}
    df = fetch_market_data(ticker=symbol, interval=interval_map[timeframe], period=period)

if df is None or df.empty:
    st.error(f"No data available for {symbol} at {timeframe} timeframe. Please try another symbol or timeframe.")
    st.stop()

df = df.copy()
df_reset = df.reset_index()

if 'Date' in df_reset.columns:
    df_reset.rename(columns={"Date": "Datetime"}, inplace=True)
elif 'Datetime' not in df_reset.columns:
    if pd.api.types.is_datetime64_any_dtype(df_reset.iloc[:, 0]):
        date_col = df_reset.columns[0]
        df_reset.rename(columns={date_col: "Datetime"}, inplace=True)
    else:
        st.error("Could not identify a datetime column in the data.")
        st.stop()

with st.spinner('Calculating trading signals...'):
    df['SMA_9'] = df['Close'].rolling(window=9).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).fillna(0)
    loss = -delta.where(delta < 0, 0).fillna(0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss.where(avg_loss != 0, 0.00001)
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI'].fillna(50, inplace=True)
    
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    buy_signals = []
    sell_signals = []
    
    min_period_for_signals = 50 
    if len(df) > min_period_for_signals:
        for i in range(min_period_for_signals, len(df)):
            try:
                sma9_now = float(df['SMA_9'].iloc[i])
                sma9_prev = float(df['SMA_9'].iloc[i-1])
                sma20_now = float(df['SMA_20'].iloc[i])
                sma20_prev = float(df['SMA_20'].iloc[i-1])
                macd_now = float(df['MACD'].iloc[i])
                rsi_now = float(df['RSI'].iloc[i])
                
                buy_cond = (sma9_prev <= sma20_prev and sma9_now > sma20_now and 
                           rsi_now < 60 and macd_now > 0)
                           
                macd_signal_now = float(df['MACD_Signal'].iloc[i])
                macd_signal_prev = float(df['MACD_Signal'].iloc[i-1])
                macd_prev = float(df['MACD'].iloc[i-1])
                
                sell_cond = (rsi_now > 70 or 
                            (macd_prev >= macd_signal_prev and macd_now < macd_signal_now))
                            
                if buy_cond:
                    buy_signals.append((df.index[i], df['Low'].iloc[i] * 0.998))
                if sell_cond:
                    sell_signals.append((df.index[i], df['High'].iloc[i] * 1.002))
            except (ValueError, TypeError, IndexError, AttributeError) as e:
                st.warning(f"Skipping signal calculation at index {i} due to: {e}")
                continue

supports, resistances = find_support_resistance(df)
fvgs = find_fair_value_gaps(df)

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                   vertical_spacing=0.03, 
                   row_heights=[0.8, 0.2] if show_volume else [1, 0])

if chart_type == "Candlestick":
    fig.add_trace(go.Candlestick(x=df_reset["Datetime"], open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price", increasing_line_color='#26a69a', decreasing_line_color='#ef5350'), row=1, col=1)
elif chart_type == "OHLC":
    fig.add_trace(go.Ohlc(x=df_reset["Datetime"], open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price", increasing_line_color='#26a69a', decreasing_line_color='#ef5350'), row=1, col=1)
else:
    fig.add_trace(go.Scatter(x=df_reset["Datetime"], y=df["Close"], name="Price", line=dict(color='#2962ff', width=2)), row=1, col=1)

fig.add_trace(go.Scatter(x=df_reset["Datetime"], y=df["SMA_9"], line=dict(color="#29b6f6", width=1.5), name="SMA 9"), row=1, col=1)
fig.add_trace(go.Scatter(x=df_reset["Datetime"], y=df["SMA_20"], line=dict(color="#ffeb3b", width=1.5), name="SMA 20"), row=1, col=1)
fig.add_trace(go.Scatter(x=df_reset["Datetime"], y=df["SMA_50"], line=dict(color="#f06292", width=1.5), name="SMA 50"), row=1, col=1)

if buy_signals:
    buy_x, buy_y = [], []
    for signal_time, signal_price in buy_signals:
        if isinstance(signal_time, pd.Timestamp):
            try:
                if signal_time in df.index:
                    loc = df.index.get_loc(signal_time)
                    buy_x.append(df_reset["Datetime"].iloc[loc])
                    buy_y.append(signal_price)
            except KeyError: pass
        else: buy_x.append(signal_time); buy_y.append(signal_price)
    if buy_x:
        fig.add_trace(go.Scatter(x=buy_x, y=buy_y, mode="markers+text", marker=dict(color="#00e676", size=15, symbol="triangle-up", line=dict(width=2, color="#ffffff")), text=["BUY"] * len(buy_x), textposition="bottom center", textfont=dict(color="#ffffff", size=10, family="Arial Black"), name="Buy Signal"), row=1, col=1)

if sell_signals:
    sell_x, sell_y = [], []
    for signal_time, signal_price in sell_signals:
        if isinstance(signal_time, pd.Timestamp):
            try:
                if signal_time in df.index:
                    loc = df.index.get_loc(signal_time)
                    sell_x.append(df_reset["Datetime"].iloc[loc])
                    sell_y.append(signal_price)
            except KeyError: pass
        else: sell_x.append(signal_time); sell_y.append(signal_price)
    if sell_x:
        fig.add_trace(go.Scatter(x=sell_x, y=sell_y, mode="markers+text", marker=dict(color="#ff1744", size=15, symbol="triangle-down", line=dict(width=2, color="#ffffff")), text=["SELL"] * len(sell_x), textposition="top center", textfont=dict(color="#ffffff", size=10, family="Arial Black"), name="Sell Signal"), row=1, col=1)

for level in supports:
    fig.add_shape(type="line", x0=df_reset["Datetime"].iloc[0], x1=df_reset["Datetime"].iloc[-1], y0=level, y1=level, line=dict(color="#26a69a", width=2, dash="dot"), row=1, col=1)
for level in resistances:
    fig.add_shape(type="line", x0=df_reset["Datetime"].iloc[0], x1=df_reset["Datetime"].iloc[-1], y0=level, y1=level, line=dict(color="#ef5350", width=2, dash="dot"), row=1, col=1)

for fvg in fvgs:
    color = "rgba(38, 166, 154, 0.3)" if fvg['type'] == 'bullish' else "rgba(239, 83, 80, 0.3)"
    try:
        fvg_dt = pd.Timestamp(fvg['datetime']) if not isinstance(fvg['datetime'], pd.Timestamp) else fvg['datetime']
        matching_rows = df_reset[df_reset['Datetime'] == fvg_dt]
        if not matching_rows.empty:
            x_idx = matching_rows.index[0]
            if x_idx < len(df_reset) -1 :
                x0_val = df_reset["Datetime"].iloc[x_idx] 
                x1_val = df_reset["Datetime"].iloc[x_idx + 1]
                fig.add_shape(type="rect", x0=x0_val, x1=x1_val, y0=fvg['bottom'], y1=fvg['top'], line=dict(width=0), fillcolor=color, layer="below", row=1, col=1)
        # else: st.warning(f"Could not find FVG datetime {fvg_dt} in chart data.") # Optional warning
    except (IndexError, KeyError, TypeError, ValueError) as e:
        st.warning(f"Skipping FVG plotting due to error: {e} for FVG: {fvg}")
        continue

if show_volume:
    colors = []
    for i in range(len(df)):
        try:
            close_val = df['Close'].iloc[i].item() if hasattr(df['Close'].iloc[i], 'item') else df['Close'].iloc[i]
            open_val = df['Open'].iloc[i].item() if hasattr(df['Open'].iloc[i], 'item') else df['Open'].iloc[i]
            colors.append('#26a69a' if close_val >= open_val else '#ef5350')
        except (AttributeError, IndexError): colors.append('#808080')
    fig.add_trace(go.Bar(x=df_reset["Datetime"], y=df["Volume"], name="Volume", marker_color=colors, opacity=0.5), row=2, col=1)

fig.update_layout(
    title=dict(text=f"{'NASDAQ-100' if symbol == '^IXIC' else symbol} - {timeframe} Chart", font=dict(size=24, color="#ffffff")), # MODIFIED: Display friendly name
    xaxis_title="", yaxis_title=dict(text="Price", font=dict(size=16, color="#ffffff")),
    xaxis_rangeslider_visible=False, height=650, template="plotly_dark",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=12, color="#ffffff")),
    margin=dict(l=10, r=10, t=40, b=10), font=dict(family="Arial, sans-serif", size=14, color="#ffffff"),
    paper_bgcolor="#131722", plot_bgcolor="#131722", hovermode="x unified"
)
fig.update_xaxes(showgrid=True, gridwidth=0.7, gridcolor="#2a2e39", showline=True, linewidth=1.5, linecolor="#ffffff", zeroline=False, title_font=dict(size=14, color="#ffffff"), tickfont=dict(size=12, color="#ffffff"))
fig.update_yaxes(showgrid=True, gridwidth=0.7, gridcolor="#2a2e39", showline=True, linewidth=1.5, linecolor="#ffffff", zeroline=False, title_font=dict(size=14, color="#ffffff"), tickfont=dict(size=12, color="#ffffff"))

config = {'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'drawclosedpath', 'drawcircle', 'drawrect', 'eraseshape'], 'scrollZoom': True}
st.plotly_chart(fig, use_container_width=True, config=config)

col_m1, col_m2, col_m3, col_m4 = st.columns(4) # Renamed metric columns
try:
    current_price = df["Close"].iloc[-1].item() if hasattr(df["Close"].iloc[-1], 'item') else df["Close"].iloc[-1]
    prev_close = df["Close"].iloc[-2].item() if hasattr(df["Close"].iloc[-2], 'item') and len(df["Close"]) > 1 else current_price
    price_change = ((current_price / prev_close) - 1) * 100 if prev_close != 0 else 0
    
    if timeframe in ["1m", "5m", "15m", "1h"] and not df.empty:
        last_timestamp = df.index[-1]
        start_24h_ago = last_timestamp - pd.Timedelta(hours=24)
        df_last_24h = df[df.index >= start_24h_ago]
        if not df_last_24h.empty:
            high_24h = df_last_24h['High'].max()
            low_24h = df_last_24h['Low'].min()
            volume_24h = df_last_24h['Volume'].sum()
        else: # Fallback if no data in last 24h (e.g. market closed)
            high_24h = df['High'].iloc[-1]; low_24h = df['Low'].iloc[-1]; volume_24h = df['Volume'].iloc[-1]
    else: # For daily data or if df_last_24h is empty
        high_24h = df['High'].max(); low_24h = df['Low'].min(); volume_24h = df['Volume'].sum()
    
    high_24h = high_24h.item() if hasattr(high_24h, 'item') else high_24h
    low_24h = low_24h.item() if hasattr(low_24h, 'item') else low_24h
    volume_24h = volume_24h.item() if hasattr(volume_24h, 'item') else volume_24h
    
    col_m1.metric("Current Price", f"${current_price:.2f}", f"{price_change:.2f}%")
    col_m2.metric("24h High", f"${high_24h:.2f}")
    col_m3.metric("24h Low", f"${low_24h:.2f}")
    col_m4.metric("Volume", f"{int(volume_24h):,}")
except Exception as e:
    st.error(f"Error calculating market stats: {str(e)}")
    col_m1.metric("Current Price", "N/A"); col_m2.metric("24h High", "N/A")
    col_m3.metric("24h Low", "N/A"); col_m4.metric("Volume", "N/A")

st.header("Market Analysis")
if run_analysis:
    with st.spinner("AI is analyzing the market..."):
        analysis = get_ai_analysis(df, analysis_type, symbol, timeframe) # Pass symbol (ticker)
    st.markdown(f"""<div style="background-color:#2a2e39; padding:20px; border-radius:5px; margin-top:10px;"><h3 style="color:#d1d4dc;">AI Market Analysis</h3><div style="color:#d1d4dc; white-space: pre-line;">{analysis}</div></div>""", unsafe_allow_html=True)
else:
    st.info("Click the 'Analyze Market' button for AI-powered analysis.")

st.markdown("""<h2 style="color:#ffffff; font-size:24px; font-weight:bold; margin-top:30px; margin-bottom:15px; text-align:center; background-color:#2a2e39; padding:15px; border-radius:5px;">🎯 Key Price Levels</h2>""", unsafe_allow_html=True)
col_sr1, col_sr2 = st.columns(2) # Renamed support/resistance columns
with col_sr1:
    st.markdown("<h4 style='color:#26a69a;'>Support Levels</h4>", unsafe_allow_html=True)
    current_price_for_sr = df["Close"].iloc[-1] if not df.empty else 0 # Define here for scope
    for level in supports[:5]:
        level_value = level.item() if hasattr(level, 'item') else float(level)
        distance_str = "N/A"
        if current_price_for_sr != 0:
            distance = ((level_value / current_price_for_sr) - 1) * 100
            distance_str = f"{distance:.2f}% from current"
        st.markdown(f"<div style='background:#1c2030; padding:10px; margin:5px 0; border-radius:5px; border-left:4px solid #26a69a;'>${level_value:.2f} <span style='color:#9ca3af; float:right;'>({distance_str})</span></div>", unsafe_allow_html=True)
with col_sr2:
    st.markdown("<h4 style='color:#ef5350;'>Resistance Levels</h4>", unsafe_allow_html=True)
    for level in resistances[:5]:
        level_value = level.item() if hasattr(level, 'item') else float(level)
        distance_str = "N/A"
        if current_price_for_sr != 0:
            distance = ((level_value / current_price_for_sr) - 1) * 100
            distance_str = f"{distance:.2f}% from current"
        st.markdown(f"<div style='background:#1c2030; padding:10px; margin:5px 0; border-radius:5px; border-left:4px solid #ef5350;'>${level_value:.2f} <span style='color:#9ca3af; float:right;'>({distance_str})</span></div>", unsafe_allow_html=True)

st.markdown("""<h2 style="color:#ffffff; font-size:24px; font-weight:bold; margin-top:30px; margin-bottom:15px; text-align:center; background-color:#2a2e39; padding:15px; border-radius:5px;">⚡ Fair Value Gaps</h2>""", unsafe_allow_html=True)
if fvgs:
    for fvg in fvgs[-5:]:
        try:
            fvg_type = fvg['type']
            color = "#26a69a" if fvg_type == "bullish" else "#ef5350"
            bottom = fvg['bottom'].item() if hasattr(fvg['bottom'], 'item') else float(fvg['bottom'])
            top = fvg['top'].item() if hasattr(fvg['top'], 'item') else float(fvg['top'])
            mid = fvg['mid'].item() if hasattr(fvg['mid'], 'item') else float(fvg['mid'])
            st.markdown(f"""<div style='background:#2a2e39; padding:15px; margin:10px 0; border-radius:5px; border-left:4px solid {color};'><div style='display:flex; justify-content:space-between;'><span style='color:{color}; font-weight:600; font-size:18px;'>{fvg_type.title()} Fair Value Gap</span><span style='color:#ffffff; font-weight:600;'>Range: ${bottom:.2f} - ${top:.2f}</span></div><div style='margin-top:8px; color:#ffffff; font-size:16px;'>Mid-point: ${mid:.2f}</div></div>""", unsafe_allow_html=True)
        except (KeyError, TypeError, ValueError) as e:
            st.warning(f"Skipping FVG display due to error: {e}")
            continue
else:
    st.info("No significant fair value gaps detected in the current timeframe.")

st.markdown("---")
st.caption("""**Disclaimer**: This trading dashboard is for informational purposes only and does not constitute investment advice. The analysis and signals provided are based on technical indicators and AI algorithms that may not predict future market movements accurately. Always conduct your own research and consider consulting with a licensed financial advisor before making investment decisions.""")

