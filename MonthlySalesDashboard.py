import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# --- Configuration ---
st.set_page_config(page_title="Monthly Sales Dashboard", layout="wide", page_icon="📊")

# --- THEME MANAGEMENT ---
# We place this at the top of the sidebar so it loads first
st.sidebar.header("Appearance")
dark_mode = st.sidebar.toggle("🌙 Dark Mode", value=True)

# --- CUSTOM STYLING ---
def apply_theme_style(is_dark):
    if is_dark:
        # DARK THEME CSS
        st.markdown("""
            <style>
                /* Force entire app background to dark */
                .stApp {
                    background-color: #0E1117;
                    color: #FAFAFA;
                }
                /* Force Sidebar background to dark gray */
                [data-testid="stSidebar"] {
                    background-color: #262730;
                    border-right: 1px solid #414141;
                }
                /* Adjust headings to white/light gray */
                h1, h2, h3, h4, h5, h6 {
                    color: #FAFAFA !important;
                }
                /* Metric containers styling */
                [data-testid="stMetricValue"] {
                    color: #FAFAFA !important;
                }
                [data-testid="stMetricLabel"] {
                    color: #A3A8B8 !important;
                }
                /* Make dataframe borders blend in */
                .stDataFrame {
                    border: 1px solid #414141;
                }
            </style>
        """, unsafe_allow_html=True)
        return "plotly_dark"
        
    else:
        # LIGHT THEME CSS (Your original code)
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
                    color: #2C3E50 !important;
                }
                /* Metric containers styling */
                [data-testid="stMetricValue"] {
                    color: #2C3E50 !important;
                }
                /* Make dataframe headers contrast better */
                .stDataFrame {
                    border: 1px solid #E0E0E0;
                }
            </style>
        """, unsafe_allow_html=True)
        return "plotly_white"

# Apply the styles and get the correct plotly template string
current_theme_template = apply_theme_style(dark_mode)

# --- Chart Color Theme ---
# We use a color sequence that looks decent on both, but you can change this if needed
COLOR_SEQUENCE = ["#008080", "#20B2AA", "#40E0D0", "#708090", "#778899", "#B0C4DE"]
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

# --- FORMAT REMINDER SECTION (UPDATED) ---
with st.expander("ℹ️ Check Required CSV Format (Click to View)"):
    st.info("Please ensure your file matches the column arrangement below before uploading.")
    
    # -------------------------------------------------------------
    # REPLACE THE LINE BELOW WITH YOUR IMAGE FILENAME/PATH
    # st.image("example_format.png", caption="Required Excel/CSV Layout", use_column_width=True)
    st.markdown("**[PLACEHOLDER: Insert your image code here]**") 
    # -------------------------------------------------------------

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
            st.plotly_chart(px.bar(monthly, x='Month_Year', y='Amount', color='Amount'), use_container_width=True)
        
        with col_chart2:
            st.subheader("Weekly Trend")
            weekly = filtered_df.groupby('Week_Start')['Amount'].sum().reset_index()
            st.plotly_chart(px.line(weekly, x='Week_Start', y='Amount', markers=True), use_container_width=True)

        st.subheader("Top Products (By Sales)")
        top_prod = filtered_df.groupby('Particular / Desc.')['Amount'].sum().nlargest(10).reset_index()
        st.plotly_chart(px.bar(top_prod, x='Amount', y='Particular / Desc.', orientation='h', color='Amount'), use_container_width=True)

    # --- TAB 2: INVENTORY HEALTH ---
    with tab2:
        st.markdown("### ⏳ Inventory Shelf Life")
        avg_days = filtered_df['Days_To_Sell'].mean()
        st.metric("Avg. Days on Shelf", f"{avg_days:.1f} Days")

        fig_hist = px.histogram(filtered_df, x='Days_To_Sell', nbins=20, title="Distribution of Days to Sell", color='Category')
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

        fig_matrix = px.scatter(prod_perf, x='Total_Sales', y='Margin_Percent', size='Count', hover_name='Particular / Desc.', title="Product Profit Matrix", color='Margin_Percent', color_continuous_scale="Teal")
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