import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, date, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

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
        df['DayOfWeekNum'] = df['DateTime'].dt.dayofweek  # 0=Monday, 6=Sunday
        df['Month'] = df['DateTime'].dt.month
        df['DayOfMonth'] = df['DateTime'].dt.day
        return df
    except FileNotFoundError:
        st.error("Bakery.csv file not found at the specified path. Please ensure the file exists.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

# Prepare data for machine learning
def prepare_prediction_data(df):
    """Prepare historical sales data for predictive modeling"""
    
    # Aggregate daily sales by item
    daily_sales = df.groupby(['Date', 'Items']).size().reset_index(name='Units_Sold')
    daily_sales['Date'] = pd.to_datetime(daily_sales['Date'])
    
    # Add date features
    daily_sales['DayOfWeek'] = daily_sales['Date'].dt.dayofweek
    daily_sales['Month'] = daily_sales['Date'].dt.month
    daily_sales['DayOfMonth'] = daily_sales['Date'].dt.day
    daily_sales['DayOfYear'] = daily_sales['Date'].dt.dayofyear
    
    # Add rolling averages
    daily_sales = daily_sales.sort_values(['Items', 'Date'])
    daily_sales['Units_Sold_7day_avg'] = daily_sales.groupby('Items')['Units_Sold'].rolling(window=7, min_periods=1).mean().reset_index(0, drop=True)
    daily_sales['Units_Sold_30day_avg'] = daily_sales.groupby('Items')['Units_Sold'].rolling(window=30, min_periods=1).mean().reset_index(0, drop=True)
    
    return daily_sales

# Build predictive model for each item
def build_prediction_models(daily_sales):
    """Build simple linear regression models for each item"""
    
    models = {}
    predictions = {}
    
    # Encode categorical variables
    le = LabelEncoder()
    
    for item in daily_sales['Items'].unique():
        item_data = daily_sales[daily_sales['Items'] == item].copy()
        
        if len(item_data) < 7:  # Skip items with insufficient data
            continue
            
        # Prepare features
        features = ['DayOfWeek', 'Month', 'DayOfMonth', 'DayOfYear', 'Units_Sold_7day_avg']
        X = item_data[features].fillna(method='ffill').fillna(method='bfill')
        y = item_data['Units_Sold']
        
        # Train model
        model = LinearRegression()
        model.fit(X, y)
        
        # Generate predictions for next 7 days
        last_date = item_data['Date'].max()
        future_dates = [last_date + timedelta(days=i) for i in range(1, 8)]
        
        future_features = []
        for future_date in future_dates:
            last_7day_avg = item_data['Units_Sold'].tail(7).mean()
            features_row = [
                future_date.weekday(),
                future_date.month,
                future_date.day,
                future_date.timetuple().tm_yday,
                last_7day_avg
            ]
            future_features.append(features_row)
        
        future_X = pd.DataFrame(future_features, columns=features)
        future_pred = model.predict(future_X)
        future_pred = np.maximum(future_pred, 0)  # Ensure non-negative predictions
        
        models[item] = model
        predictions[item] = {
            'dates': future_dates,
            'predictions': future_pred,
            'historical_avg': item_data['Units_Sold'].mean(),
            'recent_trend': item_data['Units_Sold'].tail(7).mean()
        }
    
    return models, predictions

# Calculate inventory recommendations
def calculate_inventory_recommendations(df, predictions):
    """Calculate recommended inventory levels based on predictions and historical data"""
    
    recommendations = []
    
    # Calculate historical statistics
    historical_stats = df.groupby('Items').agg({
        'Items': 'count'
    }).rename(columns={'Items': 'Total_Historical_Sales'})
    
    daily_avg = df.groupby(['Date', 'Items']).size().groupby('Items').agg(['mean', 'std', 'max'])
    daily_avg.columns = ['Avg_Daily_Sales', 'Sales_Std', 'Max_Daily_Sales']
    
    for item, pred_data in predictions.items():
        if item in daily_avg.index:
            avg_daily = daily_avg.loc[item, 'Avg_Daily_Sales']
            std_daily = daily_avg.loc[item, 'Sales_Std']
            max_daily = daily_avg.loc[item, 'Max_Daily_Sales']
            
            # Predicted demand for next 7 days
            predicted_7day = pred_data['predictions'].sum()
            predicted_daily_avg = pred_data['predictions'].mean()
            
            # Safety stock (2 standard deviations)
            safety_stock = 2 * std_daily if not pd.isna(std_daily) else avg_daily * 0.2
            
            # Recommended stock levels
            daily_recommendation = predicted_daily_avg + safety_stock
            weekly_recommendation = predicted_7day + (safety_stock * 7)
            
            recommendations.append({
                'Item': item,
                'Historical_Avg_Daily': round(avg_daily, 1),
                'Predicted_Avg_Daily': round(predicted_daily_avg, 1),
                'Predicted_7Day_Total': round(predicted_7day, 0),
                'Safety_Stock': round(safety_stock, 1),
                'Recommended_Daily_Stock': round(daily_recommendation, 0),
                'Recommended_Weekly_Stock': round(weekly_recommendation, 0),
                'Max_Historical_Daily': round(max_daily, 0)
            })
    
    return pd.DataFrame(recommendations)

# Main inventory prediction dashboard
def main_inventory_dashboard():
    # Display logo
    try:
        st.image('logo.png', width=200)
    except:
        pass  # Continue without logo if file not found
    
    st.title("Sweet & Savory Bakery - Inventory Prediction Dashboard")
    st.markdown("*Predictive analytics for optimal inventory management*")
    
    # Load data
    df = load_data()
    
    if df.empty:
        st.error("No data available. Please check your Bakery.csv file.")
        return
    
    # Sidebar info
    # Display logo in sidebar
    try:
        st.sidebar.image("/Users/rohitkrishnan/Desktop/Assesments:Projects/ESL/BakeryDash/images/logo.png", width=150)
    except:
        pass  # Continue without logo if file not found
    
    st.sidebar.title("Model Information")
    st.sidebar.write("**Prediction Method:** Linear Regression")
    st.sidebar.write("**Features Used:**")
    st.sidebar.write("• Day of week patterns")
    st.sidebar.write("• Monthly seasonality") 
    st.sidebar.write("• Recent sales trends")
    st.sidebar.write("• 7-day moving averages")
    st.sidebar.write("")
    st.sidebar.write("**Safety Stock:** 2 standard deviations")
    st.sidebar.write("**Forecast Period:** 7 days ahead")
    
    # Prepare data and build models
    with st.spinner("Building predictive models..."):
        daily_sales = prepare_prediction_data(df)
        models, predictions = build_prediction_models(daily_sales)
        recommendations = calculate_inventory_recommendations(df, predictions)
    
    if recommendations.empty:
        st.error("Unable to generate predictions. Please check your data.")
        return
    
    # Display inventory recommendations
    st.subheader("Inventory Recommendations")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Items Analyzed", len(recommendations))
    with col2:
        total_weekly_rec = recommendations['Recommended_Weekly_Stock'].sum()
        st.metric("Total Weekly Stock Needed", f"{total_weekly_rec:,.0f} units")
    with col3:
        total_daily_rec = recommendations['Recommended_Daily_Stock'].sum()
        st.metric("Total Daily Stock Needed", f"{total_daily_rec:,.0f} units")
    with col4:
        avg_safety_stock = recommendations['Safety_Stock'].mean()
        st.metric("Avg Safety Stock", f"{avg_safety_stock:.1f} units")
    
    # Detailed recommendations table
    st.subheader("Detailed Stock Recommendations")
    
    # Sort by predicted demand
    recommendations_sorted = recommendations.sort_values('Predicted_7Day_Total', ascending=False)
    
    # Format the dataframe for better display
    display_df = recommendations_sorted.copy()
    display_df = display_df.round(1)
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # Top items that need most inventory
    st.subheader("High Priority Items")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Highest Predicted Demand (Next 7 Days)**")
        top_demand = recommendations_sorted.head(10)[['Item', 'Predicted_7Day_Total']]
        fig_demand = px.bar(top_demand, x='Predicted_7Day_Total', y='Item', 
                           orientation='h', title='Predicted 7-Day Demand by Item')
        fig_demand.update_layout(height=400)
        st.plotly_chart(fig_demand, use_container_width=True)
    
    with col2:
        st.write("**Recommended Weekly Stock Levels**")
        top_stock = recommendations_sorted.head(10)[['Item', 'Recommended_Weekly_Stock']]
        fig_stock = px.bar(top_stock, x='Recommended_Weekly_Stock', y='Item',
                          orientation='h', title='Recommended Weekly Stock by Item',
                          color_discrete_sequence=['orange'])
        fig_stock.update_layout(height=400)
        st.plotly_chart(fig_stock, use_container_width=True)
    
    # Prediction vs Historical Analysis
    st.subheader("Prediction vs Historical Analysis")
    
    # Create comparison chart
    comparison_data = recommendations_sorted.head(15).copy()
    
    fig_comparison = go.Figure()
    fig_comparison.add_trace(go.Bar(
        name='Historical Avg Daily',
        x=comparison_data['Item'],
        y=comparison_data['Historical_Avg_Daily'],
        marker_color='lightblue'
    ))
    fig_comparison.add_trace(go.Bar(
        name='Predicted Avg Daily',
        x=comparison_data['Item'],
        y=comparison_data['Predicted_Avg_Daily'],
        marker_color='darkblue'
    ))
    
    fig_comparison.update_layout(
        title='Historical vs Predicted Daily Sales (Top 15 Items)',
        xaxis_title='Items',
        yaxis_title='Units per Day',
        barmode='group',
        height=500
    )
    fig_comparison.update_xaxes(tickangle=45)
    
    st.plotly_chart(fig_comparison, use_container_width=True)
    
    # Individual item analysis
    st.subheader("Individual Item Forecast")
    
    selected_item = st.selectbox("Select item for detailed forecast:", 
                                sorted(recommendations['Item'].tolist()))
    
    if selected_item in predictions:
        col1, col2 = st.columns(2)
        
        with col1:
            # Show prediction metrics
            item_rec = recommendations[recommendations['Item'] == selected_item].iloc[0]
            st.write(f"**Forecast for {selected_item}**")
            st.metric("Predicted 7-Day Demand", f"{item_rec['Predicted_7Day_Total']:.0f} units")
            st.metric("Recommended Daily Stock", f"{item_rec['Recommended_Daily_Stock']:.0f} units")
            st.metric("Safety Stock Buffer", f"{item_rec['Safety_Stock']:.1f} units")
            st.metric("Historical Daily Average", f"{item_rec['Historical_Avg_Daily']:.1f} units")
        
        with col2:
            # Show 7-day forecast chart
            pred_data = predictions[selected_item]
            forecast_df = pd.DataFrame({
                'Date': pred_data['dates'],
                'Predicted_Demand': pred_data['predictions']
            })
            
            fig_forecast = px.line(forecast_df, x='Date', y='Predicted_Demand',
                                  title=f'7-Day Demand Forecast for {selected_item}',
                                  markers=True)
            fig_forecast.add_hline(y=item_rec['Historical_Avg_Daily'], 
                                  line_dash="dash", line_color="red",
                                  annotation_text="Historical Average")
            st.plotly_chart(fig_forecast, use_container_width=True)
    
    # Model accuracy information
    with st.expander("Model Performance & Methodology", expanded=False):
        st.write("**Predictive Model Details:**")
        st.write("• Uses Linear Regression with time-series features")
        st.write("• Incorporates seasonal patterns (day of week, month)")
        st.write("• Includes recent sales trend momentum")
        st.write("• Applies safety stock buffer for demand uncertainty")
        st.write("")
        st.write("**Limitations:**")
        st.write("• Predictions based on historical patterns only")
        st.write("• Does not account for external factors (weather, events, etc.)")
        st.write("• Assumes consistent supply and business conditions")
        st.write("• Model accuracy improves with more historical data")
        st.write("")
        st.write("**Recommended Usage:**")
        st.write("• Use as starting point for inventory planning")
        st.write("• Adjust based on known upcoming events or promotions")
        st.write("• Monitor actual vs predicted sales to improve accuracy")
        st.write("• Update predictions regularly as new sales data becomes available")

# Main app
def main():
    st.set_page_config(page_title="Bakery Inventory Prediction", layout="wide")
    main_inventory_dashboard()

if __name__ == "__main__":
    main()
