import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, date, timedelta
from scipy import stats
import json
import random

# Load historical data to establish priors
@st.cache_data
def load_historical_data():
    """Load historical data to establish smart predictions"""
    try:
        df = pd.read_csv('/Users/rohitkrishnan/Desktop/Assesments:Projects/ESL/BakeryDash/Bakery.csv')
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        df['Date'] = df['DateTime'].dt.date
        return df
    except Exception as e:
        st.error(f"Error loading historical data: {str(e)}")
        return pd.DataFrame()

# Initialize inventory data
def initialize_inventory(df):
    """Initialize mock inventory data based on historical items"""
    if df.empty:
        return {}
    
    unique_items = df['Items'].unique()
    inventory = {}
    
    for item in unique_items:
        # Mock initial inventory levels (you can adjust these)
        initial_stock = random.randint(15, 50)  # Random initial stock between 15-50
        min_threshold = random.randint(5, 12)   # Low stock threshold
        max_capacity = initial_stock + random.randint(20, 40)  # Maximum capacity
        
        inventory[item] = {
            'current_stock': initial_stock,
            'min_threshold': min_threshold,
            'max_capacity': max_capacity,
            'last_restocked': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'times_restocked_today': 0
        }
    
    return inventory

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

# Inventory management functions
def get_low_stock_items(inventory):
    """Get items that are running low on stock"""
    low_stock = []
    for item, data in inventory.items():
        if data['current_stock'] <= data['min_threshold']:
            low_stock.append({
                'Item': item,
                'Current_Stock': data['current_stock'],
                'Min_Threshold': data['min_threshold'],
                'Status': 'CRITICAL' if data['current_stock'] <= data['min_threshold'] * 0.5 else 'LOW'
            })
    return low_stock

def restock_item(inventory, item_name, quantity):
    """Restock an inventory item"""
    if item_name in inventory:
        old_stock = inventory[item_name]['current_stock']
        new_stock = min(old_stock + quantity, inventory[item_name]['max_capacity'])
        inventory[item_name]['current_stock'] = new_stock
        inventory[item_name]['last_restocked'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        inventory[item_name]['times_restocked_today'] += 1
        return new_stock - old_stock  # Actual quantity added
    return 0

def update_inventory_after_sale(inventory, item_name, quantity_sold):
    """Update inventory levels after a sale"""
    if item_name in inventory:
        inventory[item_name]['current_stock'] = max(0, inventory[item_name]['current_stock'] - quantity_sold)

# Main smart inventory dashboard
def main_smart_dashboard():
    st.title("Sweet & Savory Bakery - Smart Inventory Tracker")
    st.markdown("*Real-time inventory recommendations that learn from your daily sales*")
    
    # Load historical data and initialize predictions
    historical_df = load_historical_data()
    
    if historical_df.empty:
        st.error("No historical data available for smart predictions.")
        return
    
    # Initialize or load smart predictions and inventory
    if 'initialized' not in st.session_state:
        historical_predictions = initialize_smart_predictions(historical_df)
        st.session_state.smart_predictions = historical_predictions.copy()
        st.session_state.inventory = initialize_inventory(historical_df)
        st.session_state.initialized = True
        st.session_state.daily_sales_log = []
    
    predictions = st.session_state.smart_predictions
    inventory = st.session_state.inventory
    
    # Sidebar information
    # Display logo in sidebar
    try:
        st.sidebar.image("/Users/rohitkrishnan/Desktop/Assesments:Projects/ESL/BakeryDash/images/logo.png", width=150)
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
    st.sidebar.write("4. Tracks inventory levels and alerts for low stock")
    st.sidebar.write("5. Helps you stock the right amounts")
    st.sidebar.write("")
    st.sidebar.write("**The more you use it, the better it gets!**")
    
    # Reset button
    if st.sidebar.button("Reset to Original Recommendations"):
        historical_predictions = initialize_smart_predictions(historical_df)
        st.session_state.smart_predictions = historical_predictions.copy()
        st.session_state.inventory = initialize_inventory(historical_df)
        st.session_state.daily_sales_log = []
        st.rerun()
    
    # DYNAMIC INVENTORY TRACKER SECTION
    st.subheader("Live Inventory Tracker")
    
    # Get low stock items
    low_stock_items = get_low_stock_items(inventory)
    
    # Alert for low stock items
    if low_stock_items:
        st.error(f"⚠️ **{len(low_stock_items)} items running low!**")
        
        # Create alerts
        for item_info in low_stock_items:
            status_color = "🔴" if item_info['Status'] == 'CRITICAL' else "🟡"
            st.warning(f"{status_color} **{item_info['Item']}**: {item_info['Current_Stock']} left (minimum: {item_info['Min_Threshold']})")
    else:
        st.success("✅ All items are adequately stocked!")
    
    # Inventory overview
    col1, col2, col3, col4 = st.columns(4)
    
    total_items = len(inventory)
    total_stock = sum([data['current_stock'] for data in inventory.values()])
    low_stock_count = len(low_stock_items)
    critical_stock_count = len([item for item in low_stock_items if item['Status'] == 'CRITICAL'])
    
    with col1:
        st.metric("Total Products", total_items)
    with col2:
        st.metric("Total Items in Stock", total_stock)
    with col3:
        st.metric("Low Stock Alerts", low_stock_count, delta=-low_stock_count if low_stock_count > 0 else None)
    with col4:
        st.metric("Critical Stock", critical_stock_count, delta=-critical_stock_count if critical_stock_count > 0 else None)
    
    # Inventory status table
    inventory_data = []
    for item, data in inventory.items():
        stock_level = data['current_stock']
        threshold = data['min_threshold']
        
        if stock_level <= threshold * 0.5:
            status = "🔴 CRITICAL"
            status_text = "CRITICAL"
        elif stock_level <= threshold:
            status = "🟡 LOW"
            status_text = "LOW"
        else:
            status = "🟢 GOOD"
            status_text = "GOOD"
        
        inventory_data.append({
            'Product': item,
            'Current Stock': stock_level,
            'Min Threshold': threshold,
            'Max Capacity': data['max_capacity'],
            'Status': status,
            'Status_Sort': status_text,  # For sorting
            'Last Restocked': data['last_restocked'],
            'Times Restocked Today': data['times_restocked_today']
        })
    
    inventory_df = pd.DataFrame(inventory_data)
    
    # Sort by status (Critical first, then Low, then Good)
    status_order = {'CRITICAL': 0, 'LOW': 1, 'GOOD': 2}
    inventory_df['Status_Order'] = inventory_df['Status_Sort'].map(status_order)
    inventory_df = inventory_df.sort_values('Status_Order')
    
    # Display inventory table
    display_inventory = inventory_df[['Product', 'Current Stock', 'Min Threshold', 'Max Capacity', 'Status', 'Times Restocked Today']].copy()
    st.dataframe(display_inventory, use_container_width=True, hide_index=True)
    
    # Restocking interface
    st.subheader("Restock Items")
    
    # Quick restock for low stock items
    if low_stock_items:
        st.markdown("**Quick Restock (Low Stock Items)**")
        restock_cols = st.columns(min(3, len(low_stock_items)))
        
        for i, item_info in enumerate(low_stock_items[:3]):  # Show up to 3 items
            with restock_cols[i]:
                item_name = item_info['Item']
                current = item_info['Current_Stock']
                max_cap = inventory[item_name]['max_capacity']
                suggested_restock = max_cap - current
                
                if st.button(f"Restock {item_name}", key=f"quick_restock_{item_name}"):
                    added = restock_item(st.session_state.inventory, item_name, suggested_restock)
                    st.success(f"Added {added} {item_name}(s) to inventory!")
                    st.rerun()
                
                st.write(f"Current: {current} | Suggested: +{suggested_restock}")
    
    # Manual restocking
    st.markdown("**Manual Restocking**")
    restock_col1, restock_col2, restock_col3 = st.columns([2, 1, 1])
    
    available_items = list(inventory.keys())
    
    with restock_col1:
        restock_item_select = st.selectbox("Select item to restock:", available_items, key="manual_restock_item")
    
    with restock_col2:
        if restock_item_select:
            current_stock = inventory[restock_item_select]['current_stock']
            max_capacity = inventory[restock_item_select]['max_capacity']
            max_possible = max_capacity - current_stock
            
            restock_quantity = st.number_input(
                f"Quantity (Max: {max_possible}):", 
                min_value=1, 
                max_value=max_possible, 
                value=min(10, max_possible), 
                step=1,
                key="manual_restock_qty"
            )
    
    with restock_col3:
        st.write("")  # Spacing
        if st.button("Restock Item", type="primary", key="manual_restock_btn"):
            if restock_item_select:
                added = restock_item(st.session_state.inventory, restock_item_select, restock_quantity)
                if added > 0:
                    st.success(f"Added {added} {restock_item_select}(s) to inventory!")
                    st.rerun()
                else:
                    st.error("Unable to restock - may be at maximum capacity!")
    
    # Inventory level visualization
    if available_items:
        st.subheader("Inventory Levels Visualization")
        
        # Create inventory chart
        chart_data = []
        for item, data in inventory.items():
            chart_data.append({
                'Item': item,
                'Current Stock': data['current_stock'],
                'Min Threshold': data['min_threshold'],
                'Max Capacity': data['max_capacity']
            })
        
        chart_df = pd.DataFrame(chart_data)
        
        fig_inventory = go.Figure()
        
        # Add current stock bars
        fig_inventory.add_trace(go.Bar(
            name='Current Stock',
            x=chart_df['Item'],
            y=chart_df['Current Stock'],
            marker_color=['red' if stock <= chart_df.loc[i, 'Min Threshold'] * 0.5 
                         else 'orange' if stock <= chart_df.loc[i, 'Min Threshold']
                         else 'green' for i, stock in enumerate(chart_df['Current Stock'])],
            text=chart_df['Current Stock'],
            textposition='outside'
        ))
        
        # Add threshold line
        fig_inventory.add_trace(go.Scatter(
            name='Min Threshold',
            x=chart_df['Item'],
            y=chart_df['Min Threshold'],
            mode='markers+lines',
            line=dict(color='red', dash='dash'),
            marker=dict(color='red', size=8)
        ))
        
        # Add capacity line
        fig_inventory.add_trace(go.Scatter(
            name='Max Capacity',
            x=chart_df['Item'],
            y=chart_df['Max Capacity'],
            mode='markers+lines',
            line=dict(color='blue', dash='dot'),
            marker=dict(color='blue', size=8)
        ))
        
        fig_inventory.update_layout(
            title='Current Inventory Levels vs Thresholds',
            xaxis_title='Products',
            yaxis_title='Quantity',
            height=500,
            barmode='group'
        )
        fig_inventory.update_xaxes(tickangle=45)
        
        st.plotly_chart(fig_inventory, use_container_width=True)
    
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
    st.markdown("*Every sale you record helps improve tomorrow's recommendations and updates inventory*")
    
    # Get list of items for selection
    available_items = list(predictions.keys())
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        selected_item = st.selectbox("What did you sell?", available_items)
    
    with col2:
        # Show current stock for selected item
        if selected_item and selected_item in inventory:
            current_stock = inventory[selected_item]['current_stock']
            max_sellable = min(current_stock, 20)  # Limit to current stock or 20
            quantity_sold = st.number_input("How many?", min_value=1, max_value=max(1, max_sellable), value=1, step=1)
            st.caption(f"Stock available: {current_stock}")
        else:
            quantity_sold = st.number_input("How many?", min_value=1, value=1, step=1)
    
    with col3:
        hours_since_last = st.number_input("Hours since last entry:", min_value=0.1, value=1.0, step=0.1)
    
    # Record sale button
    if st.button("Record This Sale", type="primary"):
        if selected_item in predictions and selected_item in inventory:
            # Check if we have enough stock
            if inventory[selected_item]['current_stock'] >= quantity_sold:
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
                
                # Update inventory
                update_inventory_after_sale(st.session_state.inventory, selected_item, quantity_sold)
                
                # Log the sale
                st.session_state.daily_sales_log.append({
                    'Time': datetime.now().strftime('%H:%M'),
                    'Product': selected_item,
                    'Quantity': quantity_sold,
                    'Hours_Since_Last': hours_since_last
                })
                
                st.success(f"Recorded: {quantity_sold} {selected_item}(s) sold! Inventory updated.")
                st.rerun()
            else:
                st.error(f"Not enough stock! Only {inventory[selected_item]['current_stock']} {selected_item}(s) available.")
    
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
                # Show stock level in button
                stock_level = inventory[item]['current_stock'] if item in inventory else 0
                button_text = f"Sell 1 {item} (Stock: {stock_level})"
                
                if stock_level > 0:
                    if st.button(button_text, key=f"quick_sale_{item}"):
                        # Record quick sale
                        current_alpha = predictions[item]['alpha']
                        current_beta = predictions[item]['beta']
                        
                        new_alpha, new_beta = update_predictions(current_alpha, current_beta, 1, 1.0)
                        
                        st.session_state.smart_predictions[item]['alpha'] = new_alpha
                        st.session_state.smart_predictions[item]['beta'] = new_beta
                        st.session_state.smart_predictions[item]['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        
                        # Update inventory
                        update_inventory_after_sale(st.session_state.inventory, item, 1)
                        
                        st.session_state.daily_sales_log.append({
                            'Time': datetime.now().strftime('%H:%M'),
                            'Product': item,
                            'Quantity': 1,
                            'Hours_Since_Last': 1.0
                        })
                        
                        st.success(f"Recorded: 1 {item} sold!")
                        st.rerun()
                else:
                    st.button(f"🔴 {item} - OUT OF STOCK", disabled=True, key=f"oos_{item}")
    
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
            
            # Show simple statistics with inventory info
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Most Likely Sales", f"{expected:.1f} items")
            with col2:
                variance = alpha / (beta ** 2)
                st.metric("Prediction Accuracy", "High" if variance < expected else "Moderate")
            with col3:
                st.metric("Recommendation", f"Stock {upper:.0f} items")
            with col4:
                current_stock = inventory[selected_viz_item]['current_stock'] if selected_viz_item in inventory else 0
                st.metric("Current Stock", f"{current_stock} items")
    
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
        st.write("• **Tracks Inventory**: Real-time monitoring of stock levels with low-stock alerts")
        st.write("• **Learns Your Patterns**: Gets smarter about your unique customer preferences")
        st.write("• **Adapts to Changes**: Adjusts recommendations based on today's actual sales")
        st.write("")
        st.write("**How to Use It Effectively:**")
        st.write("• Check inventory alerts each morning and restock as needed")
        st.write("• Record sales throughout the day, not just at closing")
        st.write("• Use the quick buttons for your most common items")
        st.write("• Monitor the inventory visualization to plan restocking")
        st.write("• The system works best with consistent daily use")
        st.write("")
        st.write("**Why It Gets Better Over Time:**")
        st.write("• More data = more accurate predictions")
        st.write("• Learns seasonal changes in customer preferences")
        st.write("• Adapts to new products and changing popularity")
        st.write("• Balances historical patterns with recent trends")
        st.write("• Optimizes inventory levels to reduce waste and stockouts")

# Main app
def main():
    st.set_page_config(page_title="Smart Inventory Tracker", layout="wide")
    main_smart_dashboard()

if __name__ == "__main__":
    main()