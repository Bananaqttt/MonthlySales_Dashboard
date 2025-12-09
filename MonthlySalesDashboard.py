import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# --- Configuration ---
st.set_page_config(page_title="Monthly Sales Dashboard", layout="wide", page_icon="📊")

# --- CUSTOM STYLING ---
def apply_custom_style():
    st.markdown("""
        <style>
            /* Force entire app background to white */
            .stApp {
                background-color: #FFFFFF;
                color: #333333;
            }
            /* Force Sidebar background to white (with a thin border) */
            [data-testid="stSidebar"] {
                background-color: #FFFFFF;
                border-right: 1px solid #E0E0E0;
            }
            /* Adjust headings to a dark slate color */
            h1, h2, h3, h4, h5, h6 {
                color: #2C3E50;
            }
            /* Metric containers styling */
            [data-testid="stMetricValue"] {
                color: #2C3E50;
            }
            /* Make dataframe headers contrast better */
            .stDataFrame {
                border: 1px solid #E0E0E0;
            }
            /* Customizing the Expander */
            .streamlit-expanderHeader {
                background-color: #F0F2F6;
                color: #31333F;
            }
        </style>
    """, unsafe_allow_html=True)

apply_custom_style()

# --- Chart Color Theme ---
# Teal, Slate, and Cool Blues for a professional look on white
COLOR_SEQUENCE = ["#008080", "#20B2AA", "#40E0D0", "#708090", "#778899", "#B0C4DE"]
px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = COLOR_SEQUENCE

# --- Constants ---
REQUIRED_COLUMNS = [
    'DATE - OUT', 'Category', 'Particular / Desc.', 'Amount', 'Profit', 
    'SALES PERSON', 'DATE - IN', 'Cash', 'Card'
]

# --- Helper Functions ---

def robust_read_csv(file):
    """
    Attempts to read a CSV by finding the header row dynamically.
    """
    try:
        # Read first few lines to find header
        preview = pd.read_csv(file, header=None, nrows=20)
        
        header_row_idx = None
        for i, row in preview.iterrows():
            row_values = [str(val).strip() for val in row.values]
            if 'DATE - OUT' in row_values and 'Category' in row_values:
                header_row_idx = i
                break
        
        if header_row_idx is None:
            return None, "Could not find standard headers (DATE - OUT, Category, etc.)"
        
        # Reload with correct header
        file.seek(0)
        df = pd.read_csv(file, header=header_row_idx)
        # Filter strictly for the data columns (The Big Left Block)
        df = df.iloc[:, :25] 
        return df, None
    except Exception as e:
        return None, str(e)

def clean_currency(series):
    """Helper to clean string currency to float"""
    series = series.astype(str).str.replace(',', '').str.replace(' ', '')
    series = series.apply(lambda x: '-' + x.replace('(', '').replace(')', '') if '(' in x else x)
    series = series.replace('-', '0').replace('nan', '0').replace('', '0')
    return pd.to_numeric(series, errors='coerce').fillna(0)

def clean_data(df):
    """
    Cleans the dataframe: parses dates, converts numbers, drops invalid rows.
    """
    df.columns = df.columns.str.strip()
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        return None, f"Missing required columns: {', '.join(missing_cols)}"
    
    # Drop rows where Date is empty (removes footer summaries)
    df = df.dropna(subset=['DATE - OUT'])
    
    # Clean Numbers
    cols_to_clean = ['Amount', 'Profit', 'Cash', 'Card']
    for col in cols_to_clean:
        if col in df.columns:
            df[col] = clean_currency(df[col])

    # Parse Dates
    df['Date_Sold'] = pd.to_datetime(df['DATE - OUT'], errors='coerce')
    df['Date_In'] = pd.to_datetime(df['DATE - IN'], errors='coerce')
    df = df.dropna(subset=['Date_Sold']) 

    # Extract Features
    df['Month_Year'] = df['Date_Sold'].dt.to_period('M').astype(str)
    df['Week_Start'] = df['Date_Sold'].dt.to_period('W').apply(lambda r: r.start_time)
    
    # Advanced Feature: Shelf Life
    df['Days_To_Sell'] = (df['Date_Sold'] - df['Date_In']).dt.days
    # Filter out invalid calculations
    df.loc[(df['Days_To_Sell'] < 0) | (df['Days_To_Sell'] > 1000), 'Days_To_Sell'] = np.nan

    return df, None

def forecast_sales(df):
    """
    Simple linear forecast of daily sales.
    """
    # Aggregate by date
    daily_sales = df.groupby('Date_Sold')['Amount'].sum().reset_index()
    daily_sales = daily_sales.sort_values('Date_Sold')
    
    if len(daily_sales) < 3:
        return None, "Not enough daily data points to generate a forecast."

    # Create integer time index for regression
    daily_sales['Time_Index'] = np.arange(len(daily_sales))
    
    # Linear Regression: y = mx + c
    x = daily_sales['Time_Index']
    y = daily_sales['Amount']
    
    try:
        m, c = np.polyfit(x, y, 1)
        
        # Predict next day
        next_idx = daily_sales['Time_Index'].max() + 1
        next_sales = m * next_idx + c
        
        trend_desc = "Increasing 📈" if m > 0 else "Decreasing 📉"
        
        return {
            'next_val': next_sales,
            'trend': trend_desc,
            'slope': m,
            'data': daily_sales,
            'reg_line': m * x + c
        }, None
    except:
        return None, "Could not calculate forecast."

# --- Main App ---

st.title("Monthly Sales Dashboard")

# --- ℹ️ HELP / FORMAT REMINDER SECTION ---
with st.expander("ℹ️ Check Required CSV Format (Click to View)"):
    st.info("Please ensure your file matches the column arrangement below before uploading.")
    st.write("Required Columns: DATE - OUT, Category, Particular / Desc., Amount, Profit, SALES PERSON, DATE - IN, Cash, Card")
    
    # --- [IMAGE PLACEHOLDER] ---
    # Put your image file in the same folder as this script.
    # Uncomment the line below and change 'my_image.png' to your actual file name.
    
    # st.image("my_image.png", caption="Required Excel/CSV Layout", use_container_width=True)
    
    st.markdown("*[Image Placeholder: Add your screenshot code here]*")
    # ---------------------------

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
    st.success(f"Analyzed {len(main_df)} transactions from {len(uploaded_files)} file(s).")

    # --- Filters ---
    st.sidebar.header("Filters")
    
    # Dynamic Month Filter
    available_months = sorted(main_df['Month_Year'].unique())
    selected_months = st.sidebar.multiselect("Select Month", available_months, default=available_months)
    
    # Dynamic Category Filter
    cats = [c for c in main_df['Category'].unique() if isinstance(c, str)]
    selected_cats = st.sidebar.multiselect("Select Category", cats, default=cats)

    if not selected_cats:
        st.warning("Please select at least one Category.")
        st.stop()

    filtered_df = main_df[
        (main_df['Month_Year'].isin(selected_months)) & 
        (main_df['Category'].isin(selected_cats))
    ]

    if filtered_df.empty:
        st.warning("No data matches the selected filters.")
        st.stop()

    # --- MAIN TABS ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Overview", "Inventory", "Profit Matrix", "Team", "Payments"
    ])

    # --- TAB 1: SALES OVERVIEW ---
    with tab1:
        # KPIs
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Revenue", f"₱{filtered_df['Amount'].sum():,.2f}")
        c2.metric("Total Profit", f"₱{filtered_df['Profit'].sum():,.2f}")
        c3.metric("Avg Sale Value", f"₱{filtered_df['Amount'].mean():,.2f}")

        # Charts
        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            st.subheader("Sales by Month")
            monthly = filtered_df.groupby('Month_Year')['Amount'].sum().reset_index()
            st.plotly_chart(px.bar(monthly, x='Month_Year', y='Amount', color='Amount'), use_container_width=True)
        
        with col_chart2:
            st.subheader("Weekly Trend")
            weekly = filtered_df.groupby('Week_Start')['Amount'].sum().reset_index()
            st.plotly_chart(px.line(weekly, x='Week_Start', y='Amount', markers=True), use_container_width=True)

        st.subheader("Top Products (By Sales)")
        top_prod = filtered_df.groupby('Particular / Desc.')['Amount'].sum().nlargest(10).reset_index()
        st.plotly_chart(px.bar(top_prod, x='Amount', y='Particular / Desc.', orientation='h', color='Amount'), use_container_width=True)

        # --- FORECAST SECTION ---
        st.markdown("---")
        st.subheader("🔮 Sales Forecast")
        fc_res, fc_err = forecast_sales(filtered_df)
        
        if fc_res:
            c_fc1, c_fc2 = st.columns([1, 2])
            with c_fc1:
                st.info("Forecast based on linear trend of selected data.")
                st.metric("Projected Next Day Sales", f"₱{fc_res['next_val']:,.2f}", delta=fc_res['trend'])
            with c_fc2:
                # Plot Data + Trend Line
                fc_data = fc_res['data']
                fig_fc = px.scatter(fc_data, x='Date_Sold', y='Amount', title="Daily Sales Trend & Forecast")
                # Add the regression line manually for clarity
                fig_fc.add_scatter(x=fc_data['Date_Sold'], y=fc_res['reg_line'], mode='lines', name='Trend')
                st.plotly_chart(fig_fc, use_container_width=True)
        else:
            st.caption(f"Forecast unavailable: {fc_err}")

    # --- TAB 2: INVENTORY HEALTH ---
    with tab2:
        st.markdown("### ⏳ Inventory Shelf Life")
        
        valid_inventory = filtered_df.dropna(subset=['Days_To_Sell'])
        if not valid_inventory.empty:
            avg_days = valid_inventory['Days_To_Sell'].mean()
            st.metric("Avg. Days on Shelf", f"{avg_days:.1f} Days")

            fig_hist = px.histogram(valid_inventory, x='Days_To_Sell', nbins=20, title="Distribution of Days to Sell", color='Category')
            st.plotly_chart(fig_hist, use_container_width=True)
            
            st.markdown("#### 🐢 Slowest Moving Items")
            slow_movers = valid_inventory.sort_values('Days_To_Sell', ascending=False).head(10)
            st.dataframe(slow_movers[['Particular / Desc.', 'Category', 'Date_In', 'Date_Sold', 'Days_To_Sell']], use_container_width=True)
        else:
            st.info("No Date-In/Date-Out data available for shelf life calculation.")

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

        fig_matrix = px.scatter(
            prod_perf, 
            x='Total_Sales', 
            y='Margin_Percent', 
            size='Count', 
            hover_name='Particular / Desc.', 
            title="Product Profit Matrix", 
            color='Margin_Percent', 
            color_continuous_scale="Teal"
        )
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
            fig_pie = px.pie(pay_df, names='Method', values='Amount', title="Revenue Share", hole=0.5, color_discrete_sequence=["#008080", "#778899"])
            st.plotly_chart(fig_pie, use_container_width=True)

else:
    st.info("Upload CSV files to begin.")