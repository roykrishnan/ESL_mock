import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, date, timedelta

# Load data from CSV file
@st.cache_data
def load_data():
    """Load the bakery transaction data from CSV file"""
    try:
        df = pd.read_csv('Bakery.csv')
        # Ensure DateTime column is properly formatted
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        df['Date'] = df['DateTime'].dt.date
        df['Hour'] = df['DateTime'].dt.hour
        df['DayOfWeek'] = df['DateTime'].dt.day_name()
        return df
    except FileNotFoundError:
        st.error("Bakery.csv file not found at the specified path. Please ensure the file exists.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

# Calculate sales KPIs
def calculate_sales_kpis(df, start_date, end_date, selected_item, selected_daypart, selected_daytype):
    df_filtered = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    
    if selected_item != "All Items":
        df_filtered = df_filtered[df_filtered['Items'] == selected_item]
    
    if selected_daypart != "All Dayparts":
        df_filtered = df_filtered[df_filtered['Daypart'] == selected_daypart]
    
    if selected_daytype != "All Day Types":
        df_filtered = df_filtered[df_filtered['DayType'] == selected_daytype]
    
    if df_filtered.empty:
        return {}, pd.DataFrame(), pd.DataFrame()
    
    # Calculate basic metrics
    total_transactions = df_filtered['TransactionNo'].nunique()
    total_items_sold = len(df_filtered)
    avg_items_per_transaction = total_items_sold / total_transactions if total_transactions > 0 else 0
    
    # Daily metrics
    daily_sales = df_filtered.groupby('Date').agg({
        'TransactionNo': 'nunique',
        'Items': 'count'
    }).rename(columns={'TransactionNo': 'Transactions', 'Items': 'Items_Sold'})
    
    avg_daily_transactions = daily_sales['Transactions'].mean()
    avg_daily_items = daily_sales['Items_Sold'].mean()
    
    # Item popularity
    item_popularity = df_filtered['Items'].value_counts().reset_index()
    item_popularity.columns = ['Item', 'Units_Sold']
    
    # Peak hours analysis
    hourly_sales = df_filtered.groupby('Hour').size().reset_index(name='Sales_Count')
    
    # Transaction size analysis
    transaction_sizes = df_filtered.groupby('TransactionNo').size().reset_index(name='Items_in_Transaction')
    
    kpis = {
        "Sales Overview": {
            "Total Transactions": f"{total_transactions:,}",
            "Total Items Sold": f"{total_items_sold:,}",
            "Avg Items/Transaction": f"{avg_items_per_transaction:.1f}",
            "Days in Period": f"{len(daily_sales):,}"
        },
        "Daily Averages": {
            "Avg Transactions/Day": f"{avg_daily_transactions:.0f}",
            "Avg Items/Day": f"{avg_daily_items:.0f}",
            "Peak Hour": f"{hourly_sales.loc[hourly_sales['Sales_Count'].idxmax(), 'Hour']:02d}:00",
            "Busiest Day Type": df_filtered['DayType'].mode().iloc[0] if not df_filtered.empty else "N/A"
        },
        "Performance": {
            "Most Popular Item": item_popularity.iloc[0]['Item'] if not item_popularity.empty else "N/A",
            "Top Item Sales": f"{item_popularity.iloc[0]['Units_Sold']:,}" if not item_popularity.empty else "0",
            "Avg Transaction Size": f"{transaction_sizes['Items_in_Transaction'].mean():.1f}",
            "Max Transaction Size": f"{transaction_sizes['Items_in_Transaction'].max():,}"
        }
    }
    
    return kpis, item_popularity, daily_sales

# Create trend analysis charts
def create_sales_trends(df, start_date, end_date, selected_item, selected_daypart, selected_daytype):
    df_filtered = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    
    if selected_item != "All Items":
        df_filtered = df_filtered[df_filtered['Items'] == selected_item]
    
    if selected_daypart != "All Dayparts":
        df_filtered = df_filtered[df_filtered['Daypart'] == selected_daypart]
    
    if selected_daytype != "All Day Types":
        df_filtered = df_filtered[df_filtered['DayType'] == selected_daytype]
    
    # Daily sales trends
    daily_trends = df_filtered.groupby('Date').agg({
        'TransactionNo': 'nunique',
        'Items': 'count'
    }).rename(columns={'TransactionNo': 'Transactions', 'Items': 'Items_Sold'}).reset_index()
    
    # Hourly patterns
    hourly_patterns = df_filtered.groupby('Hour').size().reset_index(name='Sales_Count')
    
    # Day of week patterns
    dayofweek_patterns = df_filtered.groupby('DayOfWeek').size().reset_index(name='Sales_Count')
    # Reorder days properly
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dayofweek_patterns['DayOfWeek'] = pd.Categorical(dayofweek_patterns['DayOfWeek'], categories=day_order, ordered=True)
    dayofweek_patterns = dayofweek_patterns.sort_values('DayOfWeek')
    
    return daily_trends, hourly_patterns, dayofweek_patterns

# Main dashboard
def main_dashboard(df):
    # Display logo
    try:
        st.image("logo.png", width=200)
    except:
        pass  # Continue without logo if file not found
    
    st.title("Sweet & Savory Bakery - Sales Analytics Dashboard")
    st.markdown("*Analyze sales patterns and optimize bakery operations*")

    if df.empty:
        st.error("No data available. Please check your Bakery.csv file.")
        return

    # Display data info
    # Display logo in sidebar
    try:
        st.sidebar.image("/Users/rohitkrishnan/Desktop/Assesments:Projects/ESL/BakeryDash/images/logo.png", width=150)
    except:
        pass  # Continue without logo if file not found
    
    st.sidebar.title("Data Overview")
    st.sidebar.write(f"Total Records: {len(df):,}")
    st.sidebar.write(f"Date Range: {df['Date'].min()} to {df['Date'].max()}")
    st.sidebar.write(f"Unique Products: {df['Items'].nunique()}")
    st.sidebar.write(f"Total Transactions: {df['TransactionNo'].nunique():,}")
    
    # Sidebar filters
    st.sidebar.title("Filters")
    
    # Item filter
    items = ["All Items"] + sorted(df['Items'].unique().tolist())
    selected_item = st.sidebar.selectbox("Product Item", items)
    
    # Daypart filter
    dayparts = ["All Dayparts"] + sorted(df['Daypart'].unique().tolist())
    selected_daypart = st.sidebar.selectbox("Day Part", dayparts)
    
    # DayType filter
    daytypes = ["All Day Types"] + sorted(df['DayType'].unique().tolist())
    selected_daytype = st.sidebar.selectbox("Day Type", daytypes)
    
    # Set date range to full dataset
    start_date = df['Date'].min()
    end_date = df['Date'].max()
    
    st.write(f"**Analyzing period:** Full Dataset ({start_date} to {end_date})")

    # Calculate KPIs
    kpis, item_popularity, daily_sales = calculate_sales_kpis(
        df, start_date, end_date, 
        selected_item, selected_daypart, selected_daytype
    )

    # Display KPI sections
    st.subheader("Key Performance Indicators")
    
    for category_name, metrics in kpis.items():
        with st.expander(f"{category_name}", expanded=True):
            cols = st.columns(len(metrics))
            for i, (metric_name, value) in enumerate(metrics.items()):
                with cols[i]:
                    st.metric(metric_name, value)

    # Product Performance Analysis
    st.subheader("Product Performance Analysis")
    
    if not item_popularity.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Top Selling Items**")
            top_items = item_popularity.head(10)
            st.dataframe(top_items, hide_index=True, use_container_width=True)
        
        with col2:
            # Top items pie chart
            top_10_items = item_popularity.head(10)
            fig_pie = px.pie(top_10_items, values='Units_Sold', names='Item',
                            title='Top 10 Items by Sales Volume')
            st.plotly_chart(fig_pie, use_container_width=True)

    # Daypart and DayType Analysis
    with st.expander("Daypart & Day Type Analysis", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            # Sales by daypart
            df_period = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
            daypart_sales = df_period['Daypart'].value_counts().reset_index()
            daypart_sales.columns = ['Daypart', 'Items_Sold']
            
            fig_daypart = px.bar(daypart_sales, x='Daypart', y='Items_Sold',
                               title='Sales by Day Part',
                               labels={'Items_Sold': 'Items Sold'})
            st.plotly_chart(fig_daypart, use_container_width=True)
        
        with col2:
            # Sales by day type
            daytype_sales = df_period['DayType'].value_counts().reset_index()
            daytype_sales.columns = ['DayType', 'Items_Sold']
            
            fig_daytype = px.bar(daytype_sales, x='DayType', y='Items_Sold',
                               title='Sales by Day Type',
                               labels={'Items_Sold': 'Items Sold'})
            st.plotly_chart(fig_daytype, use_container_width=True)

    # Transaction Analysis
    with st.expander("Transaction Analysis", expanded=False):
        df_period = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
        
        # Transaction sizes
        transaction_sizes = df_period.groupby('TransactionNo').size().reset_index(name='Items_in_Transaction')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Transaction Size Distribution**")
            size_distribution = transaction_sizes['Items_in_Transaction'].value_counts().sort_index().reset_index()
            size_distribution.columns = ['Transaction_Size', 'Count']
            
            fig_trans_size = px.bar(size_distribution, x='Transaction_Size', y='Count',
                                   title='Distribution of Transaction Sizes',
                                   labels={'Transaction_Size': 'Items per Transaction', 'Count': 'Number of Transactions'})
            st.plotly_chart(fig_trans_size, use_container_width=True)
        
        with col2:
            st.write("**Transaction Size Statistics**")
            st.metric("Average Transaction Size", f"{transaction_sizes['Items_in_Transaction'].mean():.1f} items")
            st.metric("Most Common Transaction Size", f"{transaction_sizes['Items_in_Transaction'].mode().iloc[0]} items")
            st.metric("Largest Transaction", f"{transaction_sizes['Items_in_Transaction'].max()} items")

    # Product Insights & Recommendations
    st.subheader("Business Insights & Recommendations")
    
    df_period = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    
    if not df_period.empty:
        # Most popular items
        popular_items = df_period['Items'].value_counts().head(5)
        
        # Peak hours
        peak_hour = df_period['Hour'].value_counts().idxmax()
        peak_sales = df_period['Hour'].value_counts().max()
        
        # Best performing day types and parts
        best_daytype = df_period['DayType'].value_counts().idxmax()
        best_daypart = df_period['Daypart'].value_counts().idxmax()
        
        # Generate insights
        insights = []
        
        if not popular_items.empty:
            top_item = popular_items.index[0]
            top_sales = popular_items.iloc[0]
            insights.append(f"**Best Seller:** {top_item} with {top_sales:,} units sold")
        
        insights.append(f"**Peak Hour:** {peak_hour:02d}:00 with {peak_sales:,} items sold")
        insights.append(f"**Best Day Type:** {best_daytype} generates the most sales")
        insights.append(f"**Best Day Part:** {best_daypart} is your busiest period")
        
        # Low performers
        low_performers = df_period['Items'].value_counts().tail(3)
        if not low_performers.empty:
            insights.append(f"**Consider promoting:** {', '.join(low_performers.index.tolist())} (low sales volume)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Key Insights**")
            for insight in insights:
                st.write(f"• {insight}")
        
        with col2:
            st.write("**Lowest Performing Items**")
            if not low_performers.empty:
                st.write("Items with lowest sales volume:")
                for item, sales in low_performers.items():
                    st.write(f"• {item}: {sales:,} units sold")
            else:
                st.write("• All items performing consistently")

    # Detailed Data View
    with st.expander("Raw Data View", expanded=False):
        df_display = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
        
        if selected_item != "All Items":
            df_display = df_display[df_display['Items'] == selected_item]
        if selected_daypart != "All Dayparts":
            df_display = df_display[df_display['Daypart'] == selected_daypart]
        if selected_daytype != "All Day Types":
            df_display = df_display[df_display['DayType'] == selected_daytype]
        
        st.dataframe(df_display.head(100), use_container_width=True)
        st.write(f"Showing first 100 of {len(df_display):,} filtered records")

# Main app
def main():
    st.set_page_config(page_title="Bakery Sales Dashboard", layout="wide")
    
    # Load data from CSV
    df = load_data()
    
    if not df.empty:
        # Display the main dashboard
        main_dashboard(df)
    
        # Footer
        st.markdown("---")
        st.markdown("*Dashboard built with Streamlit | Data loaded from Bakery.csv*")
    else:
        st.write("Please ensure Bakery.csv is available with the following columns:")
        st.write("• TransactionNo - Unique transaction identifier")
        st.write("• Items - Product name")
        st.write("• DateTime - Transaction timestamp")
        st.write("• Daypart - Time period (Morning, Afternoon, etc.)")
        st.write("• DayType - Day category (Weekend, Weekday, etc.)")

if __name__ == "__main__":
    main()