import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# --- Configuration ---
st.set_page_config(page_title="Monthly Sales Dashboard", layout="wide", page_icon="📊")

# --- CUSTOM STYLING ---
def apply_custom_style():
    st.markdown("""
        <style>
            /* Nuclear option - force white everywhere */
            * {
                background-color: transparent !important;
            }
            
            .stApp, .main, .block-container, section[data-testid="stSidebar"], 
            header[data-testid="stHeader"], [data-testid="stVerticalBlock"],
            [data-testid="stHorizontalBlock"] {
                background-color: #FFFFFF !important;
            }
            
            /* Sidebar specific */
            section[data-testid="stSidebar"] > div {
                background-color: #FFFFFF !important;
            }
            
            section[data-testid="stSidebar"] * {
                color: #1a1a1a !important;
            }
            
            /* Main content text */
            .stApp *, .main *, h1, h2, h3, h4, h5, h6, p, span, div, label {
                color: #1a1a1a !important;
            }
            
            /* Metrics */
            [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
                color: #1a1a1a !important;
            }
            
            /* Tabs */
            button[data-baseweb="tab"] {
                background-color: #f5f5f5 !important;
                color: #1a1a1a !important;
            }
            
            button[data-baseweb="tab"][aria-selected="true"] {
                background-color: #FFFFFF !important;
                border-bottom: 3px solid #008080 !important;
            }
            
            /* File uploader */
            [data-testid="stFileUploader"] {
                background-color: #f9f9f9 !important;
                border: 2px dashed #cccccc !important;
            }
            
            /* Success/Info/Warning */
            .stSuccess, .stInfo, .stWarning {
                background-color: #e8f5e9 !important;
                color: #1a1a1a !important;
            }
            
            /* Expander */
            [data-testid="stExpander"] {
                background-color: #f5f5f5 !important;
                border: 1px solid #e0e0e0 !important;
            }
            
            /* Status */
            [data-testid="stStatusWidget"] {
                background-color: #FFFFFF !important;
            }
        </style>
    """, unsafe_allow_html=True)

apply_custom_style()

# --- Chart Color Theme ---
TEAL_COLORS = ["#008080", "#20B2AA", "#40E0D0", "#5F9EA0", "#66CDAA"]

# --- Constants ---
REQUIRED_COLUMNS = [
    'DATE - OUT', 'Category', 'Particular / Desc.', 'Amount', 'Profit', 
    'SALES PERSON', 'DATE - IN', 'Cash', 'Card'
]

# --- Helper Functions ---

def style_chart(fig):
    """Force white background on all Plotly charts"""
    fig.update_layout(
        template="plotly_white",  # Use white template
        paper_bgcolor="rgba(255,255,255,1)",
        plot_bgcolor="rgba(255,255,255,1)",
        font=dict(color="#1a1a1a", size=12, family="Arial"),
        title_font=dict(color="#1a1a1a", size=16),
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(230,230,230,0.5)",
            linecolor="#d0d0d0",
            tickfont=dict(color="#1a1a1a"),
            title_font=dict(color="#1a1a1a")
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(230,230,230,0.5)",
            linecolor="#d0d0d0",
            tickfont=dict(color="#1a1a1a"),
            title_font=dict(color="#1a1a1a")
        ),
        legend=dict(
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#d0d0d0",
            borderwidth=1,
            font=dict(color="#1a1a1a")
        ),
        margin=dict(t=60, l=60, r=40, b=60),
        coloraxis=dict(
            colorbar=dict(
                bgcolor="rgba(255,255,255,0.9)",
                tickfont=dict(color="#1a1a1a"),
                tickcolor="#1a1a1a"
            )
        )
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
            return None, "Could not find standard headers"
        
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
        return None, f"Missing columns: {', '.join(missing_cols)}"
    
    df = df.dropna(subset=['DATE - OUT'])
    
    for col in ['Amount', 'Profit', 'Cash', 'Card']:
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
        return None, "Not enough data"

    daily_sales['Time_Index'] = np.arange(len(daily_sales))
    x = daily_sales['Time_Index']
    y = daily_sales['Amount']
    
    try:
        m, c = np.polyfit(x, y, 1)
        next_idx = x.max() + 1
        next_sales = m * next_idx + c
        trend_desc = "Increasing 📈" if m > 0 else "Decreasing 📉"
        
        return {
            'next_val': next_sales,
            'trend': trend_desc,
            'data': daily_sales,
            'reg_line': m * x + c
        }, None
    except:
        return None, "Calculation error"

# --- Main App ---
st.title("📊 Monthly Sales Dashboard")

with st.expander("ℹ️ Check Required CSV Format"):
    st.write("**Required Columns:** DATE - OUT, Category, Particular / Desc., Amount, Profit, SALES PERSON, DATE - IN, Cash, Card")

uploaded_files = st.file_uploader("Upload Monthly Sales CSV", accept_multiple_files=True, type=['csv'])

if uploaded_files:
    all_data = []
    errors = []

    with st.status("Processing Files..."):
        for file in uploaded_files:
            raw_df, error = robust_read_csv(file)
            if error:
                errors.append(f"{file.name}: {error}")
                continue
            
            clean_df, clean_error = clean_data(raw_df)
            if clean_error:
                errors.append(f"{file.name}: {clean_error}")
                continue
            all_data.append(clean_df)

    if errors:
        for err in errors:
            st.warning(err)
        if not all_data:
            st.stop()

    main_df = pd.concat(all_data, ignore_index=True)
    st.success(f"✅ Analyzed {len(main_df)} transactions from {len(uploaded_files)} file(s)")

    # --- Filters ---
    st.sidebar.header("📋 Filters")
    
    available_months = sorted(main_df['Month_Year'].unique())
    selected_months = st.sidebar.multiselect("Select Month", available_months, default=available_months)
    
    cats = [c for c in main_df['Category'].unique() if isinstance(c, str)]
    selected_cats = st.sidebar.multiselect("Select Category", cats, default=cats)

    if not selected_cats:
        st.warning("Select at least one Category")
        st.stop()

    filtered_df = main_df[
        (main_df['Month_Year'].isin(selected_months)) & 
        (main_df['Category'].isin(selected_cats))
    ]

    if filtered_df.empty:
        st.warning("No data matches filters")
        st.stop()

    # --- TABS ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Overview", "📦 Inventory", "💎 Profit Matrix", "👥 Team", "💳 Payments"])

    # --- TAB 1: OVERVIEW ---
    with tab1:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Revenue", f"₱{filtered_df['Amount'].sum():,.2f}")
        col2.metric("Total Profit", f"₱{filtered_df['Profit'].sum():,.2f}")
        col3.metric("Avg Sale", f"₱{filtered_df['Amount'].mean():,.2f}")

        st.write("")
        
        c1, c2 = st.columns(2)
        
        with c1:
            st.subheader("Sales by Month")
            monthly = filtered_df.groupby('Month_Year')['Amount'].sum().reset_index()
            fig1 = px.bar(monthly, x='Month_Year', y='Amount', 
                         labels={'Amount': 'Sales (₱)', 'Month_Year': 'Month'})
            fig1.update_traces(marker_color='#008080')
            st.plotly_chart(style_chart(fig1), use_container_width=True, theme=None)
        
        with c2:
            st.subheader("Weekly Trend")
            weekly = filtered_df.groupby('Week_Start')['Amount'].sum().reset_index()
            fig2 = px.line(weekly, x='Week_Start', y='Amount', markers=True,
                          labels={'Amount': 'Sales (₱)', 'Week_Start': 'Week'})
            fig2.update_traces(line_color='#008080', marker=dict(size=8))
            st.plotly_chart(style_chart(fig2), use_container_width=True, theme=None)

        st.write("")
        st.subheader("Top 10 Products")
        top_prod = filtered_df.groupby('Particular / Desc.')['Amount'].sum().nlargest(10).reset_index()
        top_prod = top_prod.sort_values('Amount', ascending=True)
        fig3 = px.bar(top_prod, x='Amount', y='Particular / Desc.', orientation='h',
                     labels={'Amount': 'Sales (₱)', 'Particular / Desc.': 'Product'})
        fig3.update_traces(marker_color='#008080')
        st.plotly_chart(style_chart(fig3), use_container_width=True, theme=None)

        st.write("")
        st.subheader("🔮 Sales Forecast")
        fc_res, fc_err = forecast_sales(filtered_df)
        if fc_res:
            c1, c2 = st.columns([1, 2])
            with c1:
                st.metric("Next Day Projection", f"₱{fc_res['next_val']:,.2f}", delta=fc_res['trend'])
            with c2:
                fc_data = fc_res['data']
                fig4 = go.Figure()
                fig4.add_trace(go.Scatter(x=fc_data['Date_Sold'], y=fc_data['Amount'],
                    mode='markers', name='Actual', marker=dict(color='#20B2AA', size=8)))
                fig4.add_trace(go.Scatter(x=fc_data['Date_Sold'], y=fc_res['reg_line'],
                    mode='lines', name='Trend', line=dict(color='#008080', width=2, dash='dash')))
                fig4.update_layout(xaxis_title="Date", yaxis_title="Sales (₱)")
                st.plotly_chart(style_chart(fig4), use_container_width=True, theme=None)
        else:
            st.info(fc_err)

    # --- TAB 2: INVENTORY ---
    with tab2:
        st.subheader("⏳ Inventory Shelf Life")
        valid_inv = filtered_df.dropna(subset=['Days_To_Sell'])
        if not valid_inv.empty:
            avg_days = valid_inv['Days_To_Sell'].mean()
            st.metric("Avg Days on Shelf", f"{avg_days:.1f} Days")
            
            st.write("")
            fig5 = px.histogram(valid_inv, x='Days_To_Sell', nbins=25, color='Category',
                               labels={'Days_To_Sell': 'Days to Sell'},
                               color_discrete_sequence=TEAL_COLORS)
            st.plotly_chart(style_chart(fig5), use_container_width=True, theme=None)
            
            st.write("")
            st.write("**Slowest Moving Items**")
            slow = valid_inv.nlargest(10, 'Days_To_Sell')[['Particular / Desc.', 'Category', 'Days_To_Sell']]
            st.dataframe(slow, use_container_width=True, hide_index=True)
        else:
            st.info("No shelf life data available")

    # --- TAB 3: PROFIT MATRIX ---
    with tab3:
        st.subheader("💎 Profitability vs Volume")
        prod_perf = filtered_df.groupby('Particular / Desc.').agg(
            Total_Sales=('Amount', 'sum'), 
            Total_Profit=('Profit', 'sum'), 
            Count=('Amount', 'count')
        ).reset_index()
        prod_perf['Margin_Pct'] = (prod_perf['Total_Profit'] / prod_perf['Total_Sales']) * 100
        prod_perf = prod_perf[prod_perf['Total_Sales'] > 0]
        
        avg_margin = prod_perf['Margin_Pct'].mean()
        avg_sales = prod_perf['Total_Sales'].mean()
        
        fig6 = px.scatter(prod_perf, x='Total_Sales', y='Margin_Pct', size='Count',
                         hover_name='Particular / Desc.',
                         labels={'Total_Sales': 'Sales (₱)', 'Margin_Pct': 'Margin (%)', 'Count': 'Transactions'})
        fig6.update_traces(marker=dict(color='#008080', line=dict(width=1, color='white')))
        fig6.add_hline(y=avg_margin, line_dash="dash", line_color="#999",
                      annotation_text=f"Avg: {avg_margin:.1f}%", annotation_position="right")
        fig6.add_vline(x=avg_sales, line_dash="dash", line_color="#999",
                      annotation_text=f"Avg: ₱{avg_sales:,.0f}", annotation_position="top")
        st.plotly_chart(style_chart(fig6), use_container_width=True, theme=None)

    # --- TAB 4: TEAM ---
    with tab4:
        st.subheader("🏆 Team Performance")
        team_perf = filtered_df.groupby('SALES PERSON').agg(
            Total_Sales=('Amount', 'sum'), 
            Total_Profit=('Profit', 'sum'), 
            Transactions=('Amount', 'count')
        ).reset_index()
        team_perf['Efficiency'] = (team_perf['Total_Profit'] / team_perf['Total_Sales']) * 100
        team_perf = team_perf.sort_values('Total_Sales', ascending=False)
        
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Total Sales**")
            fig7 = px.bar(team_perf, x='SALES PERSON', y='Total_Sales',
                         labels={'Total_Sales': 'Sales (₱)', 'SALES PERSON': 'Team Member'})
            fig7.update_traces(marker_color='#008080')
            st.plotly_chart(style_chart(fig7), use_container_width=True, theme=None)
            
        with c2:
            st.write("**Profit Efficiency (%)**")
            fig8 = px.bar(team_perf, x='SALES PERSON', y='Efficiency',
                         labels={'Efficiency': 'Efficiency (%)', 'SALES PERSON': 'Team Member'})
            fig8.update_traces(marker_color='#20B2AA')
            st.plotly_chart(style_chart(fig8), use_container_width=True, theme=None)

    # --- TAB 5: PAYMENTS ---
    with tab5:
        st.subheader("💳 Payment Methods")
        total_cash = filtered_df['Cash'].sum()
        total_card = filtered_df['Card'].sum()
        
        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("💵 Cash", f"₱{total_cash:,.2f}")
            st.metric("💳 Card", f"₱{total_card:,.2f}")
            pct = (total_cash/(total_cash+total_card))*100 if (total_cash+total_card)>0 else 0
            st.metric("Cash %", f"{pct:.1f}%")
            
        with c2:
            pay_df = pd.DataFrame({'Method': ['Cash', 'Card'], 'Amount': [total_cash, total_card]})
            fig9 = px.pie(pay_df, names='Method', values='Amount', hole=0.5)
            fig9.update_traces(marker=dict(colors=['#008080', '#5F9EA0']),
                              textfont=dict(color='white', size=14))
            st.plotly_chart(style_chart(fig9), use_container_width=True, theme=None)

else:
    st.info("📁 Upload CSV files to begin")