import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# --- Configuration ---
st.set_page_config(page_title="Advanced Sales Dashboard", layout="wide")

# --- Constants ---
# We verify these exist in the file
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
        # We look for a row that contains "DATE - OUT" and "Category"
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
        # We slice the first ~25 columns to avoid the right-side summary blocks
        df = df.iloc[:, :25] 
        
        return df, None
    except Exception as e:
        return None, str(e)

def clean_currency(series):
    """Helper to clean string currency to float"""
    series = series.astype(str).str.replace(',', '').str.replace(' ', '')
    # Handle negative values in accounting format like (200)
    series = series.apply(lambda x: '-' + x.replace('(', '').replace(')', '') if '(' in x else x)
    # Handle dashes as 0
    series = series.replace('-', '0').replace('nan', '0').replace('', '0')
    return pd.to_numeric(series, errors='coerce').fillna(0)

def clean_data(df):
    """
    Cleans the dataframe: parses dates, converts numbers, drops invalid rows.
    """
    # 1. Check for required columns
    # We use a loose check (case insensitive) just in case
    df.columns = df.columns.str.strip() # Clean whitespace from headers
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        return None, f"Missing required columns: {', '.join(missing_cols)}"
    
    # 2. Drop summary rows (rows where DATE - OUT is empty)
    df = df.dropna(subset=['DATE - OUT'])
    
    # 3. Clean Currency/Numeric Columns
    cols_to_clean = ['Amount', 'Profit', 'Cash', 'Card']
    for col in cols_to_clean:
        if col in df.columns:
            df[col] = clean_currency(df[col])

    # 4. Parse Dates
    df['Date_Sold'] = pd.to_datetime(df['DATE - OUT'], errors='coerce')
    df['Date_In'] = pd.to_datetime(df['DATE - IN'], errors='coerce')
    
    # Drop rows where Sale Date is invalid (likely footer rows)
    df = df.dropna(subset=['Date_Sold']) 

    # 5. Extract Features
    df['Month_Year'] = df['Date_Sold'].dt.to_period('M').astype(str)
    df['Week_Start'] = df['Date_Sold'].dt.to_period('W').apply(lambda r: r.start_time)
    
    # 6. Advanced Features
    # Shelf Life (Days to Sell)
    df['Days_To_Sell'] = (df['Date_Sold'] - df['Date_In']).dt.days
    # Filter out negative days (data entry errors) or crazy high numbers
    df.loc[(df['Days_To_Sell'] < 0) | (df['Days_To_Sell'] > 1000), 'Days_To_Sell'] = np.nan

    return df, None

# --- Main App ---

st.title("🚀 Supercharged Sales Dashboard")
st.markdown("Upload your monthly sales CSV files to unlock inventory, profit, and performance insights.")

uploaded_files = st.file_uploader("Drop CSV files here", accept_multiple_files=True, type=['csv'])

if uploaded_files:
    all_data = []
    errors = []

    # --- Loading Phase ---
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

    # --- Merging Data ---
    main_df = pd.concat(all_data, ignore_index=True)
    st.success(f"Analyzed {len(main_df)} transactions.")

    # --- Filters ---
    st.sidebar.header("Filters")
    selected_months = st.sidebar.multiselect("Select Month", main_df['Month_Year'].unique(), default=main_df['Month_Year'].unique())
    selected_cats = st.sidebar.multiselect("Select Category", main_df['Category'].unique(), default=main_df['Category'].unique())

    if not selected_cats:
        st.warning("Please select at least one Category.")
        st.stop()

    filtered_df = main_df[
        (main_df['Month_Year'].isin(selected_months)) & 
        (main_df['Category'].isin(selected_cats))
    ]

    # --- MAIN TABS ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Sales Overview", 
        "📦 Inventory Health", 
        "💎 Profit Matrix", 
        "🏆 Team Scorecard", 
        "💳 Payment Tracker"
    ])

    # --- TAB 1: SALES OVERVIEW (Original Request) ---
    with tab1:
        # KPI Row
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Revenue", f"₱{filtered_df['Amount'].sum():,.2f}")
        c2.metric("Total Profit", f"₱{filtered_df['Profit'].sum():,.2f}")
        c3.metric("Avg Sale Value", f"₱{filtered_df['Amount'].mean():,.2f}")

        # Charts
        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            st.subheader("Sales by Month")
            monthly = filtered_df.groupby('Month_Year')['Amount'].sum().reset_index()
            st.plotly_chart(px.bar(monthly, x='Month_Year', y='Amount'), use_container_width=True)
        
        with col_chart2:
            st.subheader("Weekly Trend")
            weekly = filtered_df.groupby('Week_Start')['Amount'].sum().reset_index()
            st.plotly_chart(px.line(weekly, x='Week_Start', y='Amount', markers=True), use_container_width=True)

        st.subheader("Top Products (By Sales)")
        top_prod = filtered_df.groupby('Particular / Desc.')['Amount'].sum().nlargest(10).reset_index()
        st.plotly_chart(px.bar(top_prod, x='Amount', y='Particular / Desc.', orientation='h'), use_container_width=True)

    # --- TAB 2: INVENTORY HEALTH (Shelf Life) ---
    with tab2:
        st.markdown("### ⏳ How long does it take to sell items?")
        
        # Avg days to sell metric
        avg_days = filtered_df['Days_To_Sell'].mean()
        st.metric("Avg. Days on Shelf", f"{avg_days:.1f} Days")

        # Histogram
        fig_hist = px.histogram(
            filtered_df, 
            x='Days_To_Sell', 
            nbins=20, 
            title="Distribution of Days to Sell",
            color='Category',
            labels={'Days_To_Sell': 'Days held in inventory'}
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # Slow movers table
        st.markdown("#### 🐢 Slowest Moving Items (Stagnant Stock)")
        slow_movers = filtered_df.sort_values('Days_To_Sell', ascending=False).head(10)
        st.dataframe(slow_movers[['Particular / Desc.', 'Category', 'Date_In', 'Date_Sold', 'Days_To_Sell']], use_container_width=True)

    # --- TAB 3: PROFIT MATRIX ---
    with tab3:
        st.markdown("### 💎 Profitability vs. Volume")
        st.info("Top Right = Cash Cows (High Sales, High Margin). Bottom Left = Low Performers.")
        
        # Group by Product
        prod_perf = filtered_df.groupby('Particular / Desc.').agg(
            Total_Sales=('Amount', 'sum'),
            Total_Profit=('Profit', 'sum'),
            Count=('Amount', 'count')
        ).reset_index()
        
        # Calculate Margin %
        prod_perf['Margin_Percent'] = (prod_perf['Total_Profit'] / prod_perf['Total_Sales']) * 100
        prod_perf = prod_perf[prod_perf['Total_Sales'] > 0] # Avoid div by zero errors

        fig_matrix = px.scatter(
            prod_perf,
            x='Total_Sales',
            y='Margin_Percent',
            size='Count',
            hover_name='Particular / Desc.',
            title="Product Profit Matrix",
            color='Margin_Percent',
            color_continuous_scale=px.colors.sequential.Viridis
        )
        # Add reference lines
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
        
        st.dataframe(
            team_perf.style.format({
                "Total_Sales": "₱{:,.2f}", 
                "Total_Profit": "₱{:,.2f}",
                "Avg_Ticket": "₱{:,.2f}",
                "Efficiency": "{:.1f}%"
            }),
            use_container_width=True
        )
        
        c_team1, c_team2 = st.columns(2)
        with c_team1:
            st.plotly_chart(px.bar(team_perf, x='SALES PERSON', y='Total_Sales', title="Revenue by Person"), use_container_width=True)
        with c_team2:
            st.plotly_chart(px.bar(team_perf, x='SALES PERSON', y='Efficiency', title="Profit Efficiency (%)"), use_container_width=True)

    # --- TAB 5: PAYMENT TRACKER ---
    with tab5:
        st.markdown("### 💳 Cash vs. Card Analysis")
        
        total_cash = filtered_df['Cash'].sum()
        total_card = filtered_df['Card'].sum()
        
        pay_df = pd.DataFrame({
            'Method': ['Cash', 'Card'],
            'Amount': [total_cash, total_card]
        })
        
        c_pay1, c_pay2 = st.columns([1, 2])
        
        with c_pay1:
            st.metric("Cash Sales", f"₱{total_cash:,.2f}")
            st.metric("Card Sales", f"₱{total_card:,.2f}")
            
        with c_pay2:
            fig_pie = px.pie(pay_df, names='Method', values='Amount', title="Revenue Share by Payment Method", hole=0.4)
            st.plotly_chart(fig_pie, use_container_width=True)

else:
    st.info("Upload CSV files to begin.")