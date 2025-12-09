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
            /* 1. Force entire app background to white */
            .stApp {
                background-color: #FFFFFF;
                color: #333333;
            }
            
            /* 2. Off-white Top Header/Accent */
            header[data-testid="stHeader"] {
                background-color: #F9F9F9;
                border-bottom: 1px solid #E0E0E0;
            }

            /* 3. White Sidebar with border */
            [data-testid="stSidebar"] {
                background-color: #FFFFFF;
                border-right: 1px solid #E0E0E0;
            }
            /* Sidebar Text Fix */
            [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label {
                color: #333333 !important;
            }

            /* 4. Custom File Uploader Styling */
            [data-testid="stFileUploader"] section {
                background-color: #FFFFFF;
                border: 2px dashed #B0C4DE;
                border-radius: 10px;
                padding: 20px;
            }
            [data-testid="stFileUploader"] section > div {
                color: #555;
            }

            /* 5. General Text Colors */
            h1, h2, h3, h4, h5, h6, p, li, label, .stMarkdown {
                color: #2C3E50 !important;
            }
            [data-testid="stMetricValue"] {
                color: #2C3E50;
            }
            [data-testid="stMetricLabel"] {
                color: #7F8C8D;
            }
            
            /* 6. Dataframe Borders & Text */
            .stDataFrame {
                border: 1px solid #E0E0E0;
            }
            [data-testid="stDataFrame"] div {
                color: #333333;
            }
            
            /* 7. Expander Styling */
            .streamlit-expanderHeader {
                background-color: #F0F2F6;
                color: #31333F;
            }
        </style>
    """, unsafe_allow_html=True)

apply_custom_style()

# --- Chart Color Theme ---
COLOR_SEQUENCE = ["#008080", "#20B2AA", "#40E0D0", "#708090", "#778899", "#B0C4DE"]

# --- Constants ---
REQUIRED_COLUMNS = [
    'DATE - OUT', 'Category', 'Particular / Desc.', 'Amount', 'Profit', 
    'SALES PERSON', 'DATE - IN', 'Cash', 'Card'
]

# --- Helper Functions ---

def style_chart(fig):
    """
    Applies a strict white theme to Plotly figures to override Streamlit defaults.
    """
    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="#2C3E50"),
        xaxis=dict(showgrid=True, gridcolor="#F0F0F0"),
        yaxis=dict(showgrid=True, gridcolor="#F0F0F0"),
        margin=dict(t=50, l=10, r=10, b=10)
    )
    return fig

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

def forecast_sales(df):
    daily_sales = df.groupby('Date_Sold')['Amount'].sum().reset_index()
    daily_sales = daily_sales.sort_values('Date_Sold')
    
    if len(daily_sales) < 3:
        return None, "Not enough data for forecast."

    daily_sales['Time_Index'] = np.arange(len(daily_sales))
    x = daily_sales['Time_Index']
    y = daily_sales['Amount']
    
    try:
        m, c = np.polyfit(x, y, 1)
        next_idx = daily_sales['Time_Index'].max() + 1
        next_sales = m * next_idx + c
        trend_desc = "Increasing 📈" if m > 0 else "Decreasing 📉"
        
        return {
            'next_val': next_sales,
            'trend': trend_desc,
            'data': daily_sales,
            'reg_line': m * x + c
        }, None
    except:
        return None, "Calculation error."

# --- Main App ---

st.title("Monthly Sales Dashboard")

# --- ℹ️ HELP / FORMAT REMINDER SECTION ---
with st.expander("ℹ️ Check Required CSV Format (Click to View)"):
    st.info("Please ensure your file matches the column arrangement below before uploading.")
    st.write("Required Columns: DATE - OUT, Category, Particular / Desc., Amount, Profit, SALES PERSON, DATE - IN, Cash, Card")
    st.markdown("**[PLACEHOLDER: Insert your format_guide.png here]**")

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
    
    available_months = sorted(main_df['Month_Year'].unique())
    selected_months = st.sidebar.multiselect("Select Month", available_months, default=available_months)
    
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

    # --- TAB 1: OVERVIEW ---
    with tab1:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Revenue", f"₱{filtered_df['Amount'].sum():,.2f}")
        c2.metric("Total Profit", f"₱{filtered_df['Profit'].sum():,.2f}")
        c3.metric("Avg Sale Value", f"₱{filtered_df['Amount'].mean():,.2f}")

        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            st.subheader("Sales by Month")
            monthly = filtered_df.groupby('Month_Year')['Amount'].sum().reset_index()
            fig = px.bar(monthly, x='Month_Year', y='Amount', color='Amount', color_continuous_scale='Teal')
            st.plotly_chart(style_chart(fig), use_container_width=True, theme=None)
        
        with col_chart2:
            st.subheader("Weekly Trend")
            weekly = filtered_df.groupby('Week_Start')['Amount'].sum().reset_index()
            fig = px.line(weekly, x='Week_Start', y='Amount', markers=True, color_discrete_sequence=COLOR_SEQUENCE)
            st.plotly_chart(style_chart(fig), use_container_width=True, theme=None)

        st.subheader("Top Products (By Sales)")
        top_prod = filtered_df.groupby('Particular / Desc.')['Amount'].sum().nlargest(10).reset_index()
        fig = px.bar(top_prod, x='Amount', y='Particular / Desc.', orientation='h', color='Amount', color_continuous_scale='Teal')
        st.plotly_chart(style_chart(fig), use_container_width=True, theme=None)

        st.markdown("---")
        st.subheader("🔮 Sales Forecast")
        fc_res, fc_err = forecast_sales(filtered_df)
        if fc_res:
            c_fc1, c_fc2 = st.columns([1, 2])
            with c_fc1:
                st.metric("Projected Next Day Sales", f"₱{fc_res['next_val']:,.2f}", delta=fc_res['trend'])
            with c_fc2:
                fc_data = fc_res['data']
                fig_fc = px.scatter(fc_data, x='Date_Sold', y='Amount', title="Daily Sales Trend", color_discrete_sequence=['#20B2AA'])
                fig_fc.add_scatter(x=fc_data['Date_Sold'], y=fc_res['reg_line'], mode='lines', name='Trend', line=dict(color='#2C3E50'))
                st.plotly_chart(style_chart(fig_fc), use_container_width=True, theme=None)
        else:
            st.caption(fc_err)

    # --- TAB 2: INVENTORY ---
    with tab2:
        st.markdown("### ⏳ Inventory Shelf Life")
        valid_inventory = filtered_df.dropna(subset=['Days_To_Sell'])
        if not valid_inventory.empty:
            avg_days = valid_inventory['Days_To_Sell'].mean()
            st.metric("Avg. Days on Shelf", f"{avg_days:.1f} Days")
            fig_hist = px.histogram(valid_inventory, x='Days_To_Sell', nbins=20, title="Distribution of Days to Sell", color='Category', color_discrete_sequence=COLOR_SEQUENCE)
            st.plotly_chart(style_chart(fig_hist), use_container_width=True, theme=None)
            st.dataframe(valid_inventory.sort_values('Days_To_Sell', ascending=False).head(10)[['Particular / Desc.', 'Category', 'Days_To_Sell']], use_container_width=True)
        else:
            st.info("No sufficient data.")

    # --- TAB 3: PROFIT MATRIX ---
    with tab3:
        st.markdown("### 💎 Profitability vs. Volume")
        prod_perf = filtered_df.groupby('Particular / Desc.').agg(
            Total_Sales=('Amount', 'sum'), Total_Profit=('Profit', 'sum'), Count=('Amount', 'count')
        ).reset_index()
        prod_perf['Margin_Percent'] = (prod_perf['Total_Profit'] / prod_perf['Total_Sales']) * 100
        prod_perf = prod_perf[prod_perf['Total_Sales'] > 0]
        
        fig_matrix = px.scatter(prod_perf, x='Total_Sales', y='Margin_Percent', size='Count', hover_name='Particular / Desc.', title="Product Profit Matrix", color='Margin_Percent', color_continuous_scale="Teal")
        fig_matrix.add_hline(y=prod_perf['Margin_Percent'].mean(), line_dash="dash", annotation_text="Avg Margin", line_color="#7F8C8D")
        fig_matrix.add_vline(x=prod_perf['Total_Sales'].mean(), line_dash="dash", annotation_text="Avg Sales", line_color="#7F8C8D")
        st.plotly_chart(style_chart(fig_matrix), use_container_width=True, theme=None)

    # --- TAB 4: TEAM ---
    with tab4:
        st.markdown("### 🏆 Team Scorecard")
        team_perf = filtered_df.groupby('SALES PERSON').agg(
            Total_Sales=('Amount', 'sum'), Total_Profit=('Profit', 'sum'), Transactions=('Amount', 'count')
        ).reset_index()
        team_perf['Efficiency'] = (team_perf['Total_Profit'] / team_perf['Total_Sales']) * 100
        
        c_team1, c_team2 = st.columns(2)
        with c_team1:
            fig = px.bar(team_perf, x='SALES PERSON', y='Total_Sales', color='Total_Sales', color_continuous_scale='Teal')
            st.plotly_chart(style_chart(fig), use_container_width=True, theme=None)
        with c_team2:
            fig = px.bar(team_perf, x='SALES PERSON', y='Efficiency', color='Efficiency', color_continuous_scale='Teal')
            st.plotly_chart(style_chart(fig), use_container_width=True, theme=None)

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
            st.plotly_chart(style_chart(fig_pie), use_container_width=True, theme=None)

else:
    st.info("Upload CSV files to begin.")