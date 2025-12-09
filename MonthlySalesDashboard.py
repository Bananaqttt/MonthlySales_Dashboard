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
        sample_data = pd.DataFrame({
            'DATE - OUT': ['2025-01-15', '2025-01-16'],
            'Category': ['Electronics', 'Clothing'],
            'Particular / Desc.': ['Wireless Mouse', 'Cotton T-Shirt'],
            'Amount': [500, 250],
            'Profit': [150, 100],
            'SALES PERSON': ['Juan', 'Maria'],
            'DATE - IN': ['2025-01-01', '2025-01-10'],
            'Cash': [500, 0],
            'Card': [0, 250]
        })
        
        # Create HTML content with basic styling
        html_table = sample_data.to_html(index=False, border=0, justify="left", classes="styled-table")
        full_html = f"""
        <html>
        <head>
            <title>Sample CSV Layout</title>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; padding: 40px; background-color: #f9f9f9; }}
                h2 {{ color: #333; }}
                .styled-table {{ border-collapse: collapse; margin: 25px 0; font-size: 0.9em; min-width: 400px; box-shadow: 0 0 20px rgba(0, 0, 0, 0.15); background-color: #ffffff; }}
                .styled-table thead tr {{ background-color: #1E88E5; color: #ffffff; text-align: left; }}
                .styled-table th, .styled-table td {{ padding: 12px 15px; border-bottom: 1px solid #dddddd; }}
                .styled-table tbody tr:nth-of-type(even) {{ background-color: #f3f3f3; }}
                .styled-table tbody tr:last-of-type {{ border-bottom: 2px solid #1E88E5; }}
            </style>
        </head>
        <body>
            <h2>Required Excel/CSV Layout</h2>
            <p>Your file must contain these exact headers:</p>
            {html_table}
        </body>
        </html>
        """
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(full_html)

    # 4. Display the link in a container
    with st.container(border=True):
        c1, c2 = st.columns([0.05, 0.95])
        with c1:
            st.markdown("📄")
        with c2:
            st.markdown("**View Sample Layout**")
            # Note: Streamlit maps the 'static' folder to 'app/static/' in the browser
            st.markdown(f'<a href="app/static/{file_name}" target="_blank" class="custom-link">🔗 Open Sample File (New Tab)</a>', unsafe_allow_html=True)

# --- UPLOAD SECTION ---
uploaded_files = st.file_uploader("Upload Monthly Sales CSV", accept_multiple_files=True, type=['csv'])

if uploaded_files:
    all_data = []
    errors = []

    with st.status("Processing Files..."):
        for file in uploaded_files:
            raw_df, error = robust_read_csv(file)
            if error:
                errors.append(f"File {file.name}: {error}")
                continue
            
            clean_df, clean_error = clean_data(raw_df)
            if clean_error:
                errors.append(f"File {file.name}: {clean_error}")
                continue
                
            all_data.append(clean_df)

    if errors:
        for err in errors:
            st.warning(err)
        if not all_data:
            st.stop()

    main_df = pd.concat(all_data, ignore_index=True)
    st.success(f"Analyzed {len(main_df)} transactions.")

    # --- Filters ---
    st.sidebar.header("Filters")
    selected_months = st.sidebar.multiselect("Select Month", main_df['Month_Year'].unique(), default=main_df['Month_Year'].unique())
    
    cats = [c for c in main_df['Category'].unique() if isinstance(c, str)]
    selected_cats = st.sidebar.multiselect("Select Category", cats, default=cats)

    if not selected_cats:
        st.warning("Please select at least one Category.")
        st.stop()

    filtered_df = main_df[
        (main_df['Month_Year'].isin(selected_months)) & 
        (main_df['Category'].isin(selected_cats))
    ]

    # --- MAIN TABS ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Overview", "Inventory", "Profit Matrix", "Team", "Payments"
    ])

    # --- TAB 1: SALES OVERVIEW ---
    with tab1:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Revenue", f"₱{filtered_df['Amount'].sum():,.2f}")
        c2.metric("Total Profit", f"₱{filtered_df['Profit'].sum():,.2f}")
        c3.metric("Avg Sale Value", f"₱{filtered_df['Amount'].mean():,.2f}")

        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            st.subheader("Sales by Month")
            monthly = filtered_df.groupby('Month_Year')['Amount'].sum().reset_index()
            # Removed color argument to keep single blue color
            st.plotly_chart(px.bar(monthly, x='Month_Year', y='Amount'), use_container_width=True)
        
        with col_chart2:
            st.subheader("Weekly Trend")
            weekly = filtered_df.groupby('Week_Start')['Amount'].sum().reset_index()
            st.plotly_chart(px.line(weekly, x='Week_Start', y='Amount', markers=True), use_container_width=True)

        st.subheader("Top Products (By Sales)")
        top_prod = filtered_df.groupby('Particular / Desc.')['Amount'].sum().nlargest(10).reset_index()
        st.plotly_chart(px.bar(top_prod, x='Amount', y='Particular / Desc.', orientation='h'), use_container_width=True)

    # --- TAB 2: INVENTORY HEALTH ---
    with tab2:
        st.markdown("### ⏳ Inventory Shelf Life")
        avg_days = filtered_df['Days_To_Sell'].mean()
        st.metric("Avg. Days on Shelf", f"{avg_days:.1f} Days")

        fig_hist = px.histogram(filtered_df, x='Days_To_Sell', nbins=20, title="Distribution of Days to Sell")
        st.plotly_chart(fig_hist, use_container_width=True)
        
        st.markdown("#### 🐢 Slowest Moving Items")
        slow_movers = filtered_df.sort_values('Days_To_Sell', ascending=False).head(10)
        st.dataframe(slow_movers[['Particular / Desc.', 'Category', 'Date_In', 'Date_Sold', 'Days_To_Sell']], use_container_width=True)

    # --- TAB 3: PROFIT MATRIX ---
    with tab3:
        st.markdown("### 💎 Profitability vs. Volume")
        prod_perf = filtered_df.groupby('Particular / Desc.').agg(
            Total_Sales=('Amount', 'sum'),
            Total_Profit=('Profit', 'sum'),
            Count=('Amount', 'count')
        ).reset_index()
        prod_perf['Margin_Percent'] = (prod_perf['Total_Profit'] / prod_perf['Total_Sales']) * 100
        prod_perf = prod_perf[prod_perf['Total_Sales'] > 0]

        fig_matrix = px.scatter(prod_perf, x='Total_Sales', y='Margin_Percent', size='Count', hover_name='Particular / Desc.', title="Product Profit Matrix")
        fig_matrix.add_hline(y=prod_perf['Margin_Percent'].mean(), line_dash="dash", annotation_text="Avg Margin")
        fig_matrix.add_vline(x=prod_perf['Total_Sales'].mean(), line_dash="dash", annotation_text="Avg Sales")
        st.plotly_chart(fig_matrix, use_container_width=True)

    # --- TAB 4: TEAM SCORECARD ---
    with tab4:
        st.markdown("### 🏆 Salesperson Performance")
        team_perf = filtered_df.groupby('SALES PERSON').agg(
            Total_Sales=('Amount', 'sum'),
            Total_Profit=('Profit', 'sum'),
            Transactions=('Amount', 'count')
        ).reset_index()
        team_perf['Avg_Ticket'] = team_perf['Total_Sales'] / team_perf['Transactions']
        team_perf['Efficiency'] = (team_perf['Total_Profit'] / team_perf['Total_Sales']) * 100
        
        c_team1, c_team2 = st.columns(2)
        with c_team1:
            st.plotly_chart(px.bar(team_perf, x='SALES PERSON', y='Total_Sales', title="Revenue by Person"), use_container_width=True)
        with c_team2:
            st.plotly_chart(px.bar(team_perf, x='SALES PERSON', y='Efficiency', title="Profit Efficiency (%)"), use_container_width=True)
        st.dataframe(team_perf.style.format({"Total_Sales": "₱{:,.2f}", "Total_Profit": "₱{:,.2f}", "Avg_Ticket": "₱{:,.2f}", "Efficiency": "{:.1f}%"}), use_container_width=True)

    # --- TAB 5: PAYMENTS ---
    with tab5:
        st.markdown("### 💳 Payment Analysis")
        total_cash = filtered_df['Cash'].sum()
        total_card = filtered_df['Card'].sum()
        pay_df = pd.DataFrame({'Method': ['Cash', 'Card'], 'Amount': [total_cash, total_card]})
        
        c_pay1, c_pay2 = st.columns([1, 2])
        with c_pay1:
            st.metric("Cash Sales", f"₱{total_cash:,.2f}")
            st.metric("Card Sales", f"₱{total_card:,.2f}")
        with c_pay2:
            # Pie charts require distinct colors
            fig_pie = px.pie(pay_df, names='Method', values='Amount', title="Revenue Share", hole=0.5, color_discrete_sequence=["#1E88E5", "#90CAF9"])
            st.plotly_chart(fig_pie, use_container_width=True)