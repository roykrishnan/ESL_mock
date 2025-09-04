import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, date, timedelta
from scipy import stats
import json

# Load historical data to establish priors
@st.cache_data
def load_historical_data():
    """Load historical data to establish smart predictions"""
    try:
        df = pd.read_csv('Bakery.csv')
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        df['Date'] = df['DateTime'].dt.date
        return df
    except Exception as e:
        st.error(f"Error loading historical data: {str(e)}")
        return pd.DataFrame()

# Initialize smart predictions from historical data
def initialize_smart_predictions(df):
    """Calculate smart inventory predictions for each item based on historical data"""
    if df.empty:
        return {}
    
    # Calculate daily sales for each item
    daily_sales = df.groupby(['Date', 'Items']).size().reset_index(name='Units_Sold')
    
    predictions = {}
    for item in daily_sales['Items'].unique():
        item_sales = daily_sales[daily_sales['Items'] == item]['Units_Sold']
        
        # Use statistical analysis to create smart predictions
        sample_mean = item_sales.mean()
        sample_var = item_sales.var()
        
        if sample_var > 0 and sample_mean > 0:
            # Statistical parameters for smart predictions
            beta = sample_mean / sample_var
            alpha = sample_mean * beta
        else:
            # Default values for new items
            alpha = 1.0
            beta = 1.0
        
        predictions[item] = {
            'alpha': alpha,
            'beta': beta,
            'historical_mean': sample_mean,
            'historical_samples': len(item_sales),
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    return predictions

# Update predictions with new sales data
def update_predictions(prior_alpha, prior_beta, new_sales, time_elapsed_hours=1):
    """Update smart predictions with new sales information"""
    
    updated_alpha = prior_alpha + new_sales
    updated_beta = prior_beta + (time_elapsed_hours / 24)
    
    return updated_alpha, updated_beta

# Calculate current inventory recommendations
def calculate_smart_recommendations(predictions, confidence_level=0.95):
    """Calculate inventory recommendations based on current smart predictions"""
    recommendations = []
    
    for item, params in predictions.items():
        alpha = params['alpha']
        beta = params['beta']
        
        # Expected daily demand
        expected_daily_demand = alpha / beta
        
        # Conservative and optimistic estimates
        conservative_estimate = stats.gamma.ppf((1 - confidence_level) / 2, alpha, scale=1/beta)
        optimistic_estimate = stats.gamma.ppf((1 + confidence_level) / 2, alpha, scale=1/beta)
        
        # Recommended stock (be prepared for higher demand)
        recommended_stock = optimistic_estimate
        
        recommendations.append({
            'Item': item,
            'Expected_Daily_Sales': expected_daily_demand,
            'Conservative_Estimate': conservative_estimate,
            'Optimistic_Estimate': optimistic_estimate,
            'Recommended_Daily_Stock': recommended_stock,
            'Confidence_Level': confidence_level,
            'Last_Updated': params['last_updated']
        })
    
    return pd.DataFrame(recommendations)

# Main smart inventory dashboard
def main_smart_dashboard():
    st.title("Sweet & Savory Bakery - Smart Inventory Tracker")
    st.markdown("*Real-time inventory recommendations that learn from your daily sales*")
    
    # Load historical data and initialize predictions
    historical_df = load_historical_data()
    
    if historical_df.empty:
        st.error("No historical data available for smart predictions.")
        return
    
    # Initialize or load smart predictions
    if 'initialized' not in st.session_state:
        historical_predictions = initialize_smart_predictions(historical_df)
        st.session_state.smart_predictions = historical_predictions.copy()
        st.session_state.initialized = True
        st.session_state.daily_sales_log = []
    
    # Ensure all session state variables exist (with proper error handling)
    if 'smart_predictions' not in st.session_state:
        st.session_state.smart_predictions = initialize_smart_predictions(historical_df)
    if 'daily_sales_log' not in st.session_state:
        st.session_state.daily_sales_log = []
    
    # Handle legacy variable names for backwards compatibility
    if 'bayesian_posteriors' in st.session_state and 'smart_predictions' not in st.session_state:
        st.session_state.smart_predictions = st.session_state.bayesian_posteriors
    
    predictions = st.session_state.smart_predictions
    
    # Sidebar information
    # Display logo in sidebar
    try:
        st.sidebar.image('images/logo.png', width=150)
    except:
        pass
    
    st.sidebar.title("How This Works")
    st.sidebar.write("**Smart Learning System**")
    st.sidebar.write("This system learns from your sales patterns and gets smarter over time.")
    st.sidebar.write("")
    st.sidebar.write("**What it does:**")
    st.sidebar.write("1. Analyzes your historical sales")
    st.sidebar.write("2. Records today's sales as you make them")
    st.sidebar.write("3. Updates recommendations in real-time")
    st.sidebar.write("4. Helps you stock the right amounts")
    st.sidebar.write("")
    st.sidebar.write("**The more you use it, the better it gets!**")
    
    # Reset button
    if st.sidebar.button("Reset to Original Recommendations"):
        historical_predictions = initialize_smart_predictions(historical_df)
        st.session_state.smart_predictions = historical_predictions.copy()
        st.session_state.daily_sales_log = []
        st.rerun()
    
    # Current recommendations
    st.subheader("Today's Inventory Recommendations")
    
    recommendations = calculate_smart_recommendations(predictions)
    
    if not recommendations.empty:
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            total_expected = recommendations['Expected_Daily_Sales'].sum()
            st.metric("Expected Total Sales Today", f"{total_expected:.0f} items")
        with col2:
            total_recommended = recommendations['Recommended_Daily_Stock'].sum()
            st.metric("Total Items to Stock", f"{total_recommended:.0f} items")
        with col3:
            st.metric("Items We Track", f"{len(recommendations)}")
        with col4:
            st.metric("Confidence Level", "95%")
        
        # Detailed recommendations table
        display_df = recommendations[['Item', 'Expected_Daily_Sales', 'Conservative_Estimate', 'Optimistic_Estimate', 'Recommended_Daily_Stock']].copy()
        display_df = display_df.round(1)
        display_df.columns = ['Product', 'Expected Sales', 'Low Estimate', 'High Estimate', 'Stock This Many']
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # Sales recording section
    st.subheader("Record Your Sales")
    st.markdown("*Every sale you record helps improve tomorrow's recommendations*")
    
    # Get list of items for selection
    available_items = list(predictions.keys())
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        selected_item = st.selectbox("What did you sell?", available_items)
    
    with col2:
        quantity_sold = st.number_input("How many?", min_value=1, value=1, step=1)
    
    with col3:
        hours_since_last = st.number_input("Hours since last entry:", min_value=0.1, value=1.0, step=0.1)
    
    # Record sale button
    if st.button("Record This Sale", type="primary"):
        if selected_item in predictions:
            # Update predictions
            current_alpha = predictions[selected_item]['alpha']
            current_beta = predictions[selected_item]['beta']
            
            new_alpha, new_beta = update_predictions(
                current_alpha, current_beta, quantity_sold, hours_since_last
            )
            
            # Update the predictions in memory
            st.session_state.smart_predictions[selected_item]['alpha'] = new_alpha
            st.session_state.smart_predictions[selected_item]['beta'] = new_beta
            st.session_state.smart_predictions[selected_item]['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Log the sale
            st.session_state.daily_sales_log.append({
                'Time': datetime.now().strftime('%H:%M'),
                'Product': selected_item,
                'Quantity': quantity_sold,
                'Hours_Since_Last': hours_since_last
            })
            
            st.success(f"Recorded: {quantity_sold} {selected_item}(s) sold!")
            st.rerun()
    
    # Quick sale buttons for popular items
    st.subheader("Quick Sale Buttons")
    st.markdown("*One-click recording for your best sellers*")
    
    # Get top 6 most popular items
    if not recommendations.empty:
        top_items = recommendations.nlargest(6, 'Expected_Daily_Sales')['Item'].tolist()
        
        cols = st.columns(3)
        for i, item in enumerate(top_items):
            col_idx = i % 3
            with cols[col_idx]:
                if st.button(f"Sold 1 {item}", key=f"quick_sale_{item}"):
                    # Record quick sale
                    current_alpha = predictions[item]['alpha']
                    current_beta = predictions[item]['beta']
                    
                    new_alpha, new_beta = update_predictions(current_alpha, current_beta, 1, 1.0)
                    
                    st.session_state.smart_predictions[item]['alpha'] = new_alpha
                    st.session_state.smart_predictions[item]['beta'] = new_beta
                    st.session_state.smart_predictions[item]['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    
                    st.session_state.daily_sales_log.append({
                        'Time': datetime.now().strftime('%H:%M'),
                        'Product': item,
                        'Quantity': 1,
                        'Hours_Since_Last': 1.0
                    })
                    
                    st.success(f"Recorded: 1 {item} sold!")
                    st.rerun()
    
    # Sales confidence visualization
    st.subheader("Sales Confidence Analysis")
    
    if available_items:
        selected_viz_item = st.selectbox("Select product to see detailed sales prediction:", available_items)
        
        if selected_viz_item in predictions:
            alpha = predictions[selected_viz_item]['alpha']
            beta = predictions[selected_viz_item]['beta']
            
            # Generate prediction range
            x = np.linspace(0, stats.gamma.ppf(0.99, alpha, scale=1/beta) * 1.5, 1000)
            y = stats.gamma.pdf(x, alpha, scale=1/beta)
            
            # Create prediction chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Sales Prediction Range',
                                   line=dict(color='blue', width=2)))
            
            # Add expected sales line
            expected = alpha / beta
            fig.add_vline(x=expected, line_dash="dash", line_color="red",
                         annotation_text=f"Most Likely Sales: {expected:.1f}")
            
            # Add confidence range
            lower = stats.gamma.ppf(0.025, alpha, scale=1/beta)
            upper = stats.gamma.ppf(0.975, alpha, scale=1/beta)
            fig.add_vrect(x0=lower, x1=upper, fillcolor="rgba(0,100,255,0.1)",
                         annotation_text="95% Confidence Range")
            
            fig.update_layout(
                title=f'Sales Prediction for {selected_viz_item}',
                xaxis_title='Number of Items Expected to Sell',
                yaxis_title='Confidence Level',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show simple statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Most Likely Sales", f"{expected:.1f} items")
            with col2:
                variance = alpha / (beta ** 2)
                st.metric("Prediction Accuracy", "High" if variance < expected else "Moderate")
            with col3:
                st.metric("Recommendation", f"Stock {upper:.0f} items")
    
    # Today's sales log
    with st.expander("Today's Sales Log", expanded=False):
        if st.session_state.daily_sales_log:
            sales_df = pd.DataFrame(st.session_state.daily_sales_log)
            st.dataframe(sales_df, use_container_width=True, hide_index=True)
            
            # Quick summary
            total_today = sum([sale['Quantity'] for sale in st.session_state.daily_sales_log])
            st.write(f"**Total items sold today: {total_today}**")
        else:
            st.write("No sales recorded yet today. Start recording to improve your recommendations!")
    
    # How recommendations changed
    st.subheader("How Your Recommendations Have Changed")
    
    if available_items and not historical_df.empty:
        comparison_data = []
        historical_predictions = initialize_smart_predictions(historical_df)
        
        for item in available_items[:10]:  # Show top 10 for clarity
            if item in historical_predictions:
                original_prediction = historical_predictions[item]['alpha'] / historical_predictions[item]['beta']
                current_prediction = predictions[item]['alpha'] / predictions[item]['beta']
                
                comparison_data.append({
                    'Product': item,
                    'Original_Prediction': original_prediction,
                    'Current_Prediction': current_prediction,
                    'Change': current_prediction - original_prediction
                })
        
        if comparison_data:
            comp_df = pd.DataFrame(comparison_data)
            
            fig_comp = go.Figure()
            fig_comp.add_trace(go.Bar(name='Original Prediction', x=comp_df['Product'], y=comp_df['Original_Prediction'],
                                     marker_color='lightblue'))
            fig_comp.add_trace(go.Bar(name='Updated Prediction', x=comp_df['Product'], y=comp_df['Current_Prediction'],
                                     marker_color='darkblue'))
            
            fig_comp.update_layout(
                title='How Sales Predictions Have Changed Today',
                xaxis_title='Products',
                yaxis_title='Expected Daily Sales',
                barmode='group',
                height=400
            )
            fig_comp.update_xaxes(tickangle=45)
            
            st.plotly_chart(fig_comp, use_container_width=True)
    
    # Simple explanation
    with st.expander("How This Smart System Helps Your Business", expanded=False):
        st.write("**What This System Does for You:**")
        st.write("• **Reduces Waste**: Helps you stock the right amounts so you don't overbake")
        st.write("• **Prevents Stockouts**: Makes sure you have enough of your popular items")
        st.write("• **Learns Your Patterns**: Gets smarter about your unique customer preferences")
        st.write("• **Adapts to Changes**: Adjusts recommendations based on today's actual sales")
        st.write("")
        st.write("**How to Use It Effectively:**")
        st.write("• Record sales throughout the day, not just at closing")
        st.write("• Use the quick buttons for your most common items")
        st.write("• Check recommendations each morning before baking")
        st.write("• The system works best with consistent daily use")
        st.write("")
        st.write("**Why It Gets Better Over Time:**")
        st.write("• More data = more accurate predictions")
        st.write("• Learns seasonal changes in customer preferences")
        st.write("• Adapts to new products and changing popularity")
        st.write("• Balances historical patterns with recent trends")

# Main app
def main():
    st.set_page_config(page_title="Smart Inventory Tracker", layout="wide")
    main_smart_dashboard()

if __name__ == "__main__":
    main()
