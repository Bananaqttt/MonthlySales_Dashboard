import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import os

# --- Configuration ---
st.set_page_config(page_title="Monthly Sales Dashboard", layout="wide", page_icon="📊")

# --- THEME MANAGEMENT ---
st.sidebar.header("Appearance")
dark_mode = st.sidebar.toggle("🌙 Dark Mode", value=True)

# --- CUSTOM STYLING ---
def apply_theme_style(is_dark):
    if is_dark:
        # DARK THEME CSS
        st.markdown("""
            <style>
                .stApp { background-color: #0E1117; color: #FAFAFA; }
                [data-testid="stSidebar"] { background-color: #262730; border-right: 1px solid #414141; }
                h1, h2, h3, h4, h5, h6, [data-testid="stMetricValue"] { color: #FAFAFA !important; }
                [data-testid="stMetricLabel"] { color: #A3A8B8 !important; }
                .stDataFrame { border: 1px solid #414141; }
                /* Custom link style */
                a.custom-link { color: #40E0D0 !important; text-decoration: none; font-weight: bold; }
                a.custom-link:hover { text-decoration: underline; }
            </style>
        """, unsafe_allow_html=True)
        return "plotly_dark"
    else:
        # LIGHT THEME CSS
        st.markdown("""
            <style>
                .stApp { background-color: #FFFFFF; color: #333333; }
                [data-testid="stSidebar"] { background-color: #FFFFFF; border-right: 1px solid #E0E0E0; }
                h1, h2, h3, h4, h5, h6, [data-testid="stMetricValue"] { color: #2C3E50 !important; }
                .stDataFrame { border: 1px solid #E0E0E0; }
                /* Custom link style */
                a.custom-link { color: #1E88E5 !important; text-decoration: none; font-weight: bold; }
                a.custom-link:hover { text-decoration: underline; }
            </style>
        """, unsafe_allow_html=True)
        return "plotly_white"

current_theme_template = apply_theme_style(dark_mode)

# --- Chart Color Theme ---
# Single Blue Color for all charts
COLOR_SEQUENCE = ["#1E88E5"] 
px.defaults.template = current_theme_template
px.defaults.color_discrete_sequence = COLOR_SEQUENCE

# --- Constants ---
REQUIRED_COLUMNS = [
    'DATE - OUT', 'Category', 'Particular / Desc.', 'Amount', 'Profit', 
    'SALES PERSON', 'DATE - IN', 'Cash', 'Card'
]

# --- Helper Functions ---
def robust_read_csv(file):
    try:
        preview = pd.read_csv(file, header=None, nrows=20)
        header_row_idx = None
        for i, row in preview.iterrows():
            row_values = [str(val).strip() for val in row.values]
            if 'DATE - OUT' in row_values and 'Category' in row_values:
                header_row_idx = i
                break
        
        if header_row_idx is None:
            return None, "Could not find standard headers (DATE - OUT, Category, etc.)"
        
        file.seek(0)
        df = pd.read_csv(file, header=header_row_idx)
        df = df.iloc[:, :25] 
        return df, None
    except Exception as e:
        return None, str(e)

def clean_currency(series):
    series = series.astype(str).str.replace(',', '').str.replace(' ', '')
    series = series.apply(lambda x: '-' + x.replace('(', '').replace(')', '') if '(' in x else x)
    series = series.replace('-', '0').replace('nan', '0').replace('', '0')
    return pd.to_numeric(series, errors='coerce').fillna(0)

def clean_data(df):
    df.columns = df.columns.str.strip()
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        return None, f"Missing required columns: {', '.join(missing_cols)}"
    
    df = df.dropna(subset=['DATE - OUT'])
    
    cols_to_clean = ['Amount', 'Profit', 'Cash', 'Card']
    for col in cols_to_clean:
        if col in df.columns:
            df[col] = clean_currency(df[col])

    df['Date_Sold'] = pd.to_datetime(df['DATE - OUT'], errors='coerce')
    df['Date_In'] = pd.to_datetime(df['DATE - IN'], errors='coerce')
    df = df.dropna(subset=['Date_Sold']) 

    df['Month_Year'] = df['Date_Sold'].dt.to_period('M').astype(str)
    df['Week_Start'] = df['Date_Sold'].dt.to_period('W').apply(lambda r: r.start_time)
    
    df['Days_To_Sell'] = (df['Date_Sold'] - df['Date_In']).dt.days
    df.loc[(df['Days_To_Sell'] < 0) | (df['Days_To_Sell'] > 1000), 'Days_To_Sell'] = np.nan

    return df, None

# --- Main App ---

st.title("Monthly Sales Dashboard")

# --- FORMAT REMINDER SECTION (View-Only Link) ---
with st.expander("ℹ️ Check Required CSV Format"):
    st.write("Click the link below to view the required column arrangement in a new tab.")

    # 1. Define path logic
    static_folder = "static"
    file_name = "sample_layout.html"
    file_path = os.path.join(static_folder, file_name)

    # 2. Ensure static folder exists
    if not os.path.exists(static_folder):
        os.makedirs(static_folder)

    # 3. Auto-generate the HTML file if missing
    if not os.path.exists(file_path):