import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, date, timedelta


try:
    st.sidebar.image("images/logo.png", width=200)
except Exception:
    pass

# Initialize mock inventory data
@st.cache_data
def initialize_mock_inventory():
    """Initialize mock inventory data for bakery items"""
    bakery_items = [
        'Croissant', 'Baguette', 'Sourdough Bread', 'Chocolate Chip Muffin', 
        'Blueberry Muffin', 'Danish', 'Donut', 'Scone', 'Bagel', 'Cinnamon Roll',
        'Apple Pie', 'Chocolate Cake', 'Vanilla Cupcake', 'Brownie', 'Cookie'
    ]
    
    inventory_data = {}
    for item in bakery_items:
        # Generate realistic starting inventory
        current_stock = np.random.randint(15, 50)
        max_capacity = np.random.randint(80, 120)
        reorder_point = np.random.randint(8, 15)
        
        inventory_data[item] = {
            'current_stock': current_stock,
            'max_capacity': max_capacity,
            'reorder_point': reorder_point,
            'last_updated': datetime.now(),
            'sales_today': 0
        }
    
    return inventory_data

# Update inventory when items are sold
def sell_item(item_name, quantity):
    """Reduce inventory when items are sold"""
    if item_name in st.session_state.inventory:
        current = st.session_state.inventory[item_name]['current_stock']
        new_stock = max(0, current - quantity)
        
        st.session_state.inventory[item_name]['current_stock'] = new_stock
        st.session_state.inventory[item_name]['sales_today'] += quantity
        st.session_state.inventory[item_name]['last_updated'] = datetime.now()
        
        return new_stock
    return 0

# Add inventory when restocking
def add_inventory(item_name, quantity):
    """Add inventory when restocking"""
    if item_name in st.session_state.inventory:
        current = st.session_state.inventory[item_name]['current_stock']
        max_cap = st.session_state.inventory[item_name]['max_capacity']
        new_stock = min(max_cap, current + quantity)
        
        st.session_state.inventory[item_name]['current_stock'] = new_stock
        st.session_state.inventory[item_name]['last_updated'] = datetime.now()
        
        return new_stock
    return 0

# Get inventory status for alerts
def get_inventory_status():
    """Analyze inventory status and generate alerts"""
    if 'inventory' not in st.session_state:
        return [], [], []
    
    low_stock = []
    out_of_stock = []
    well_stocked = []
    
    for item, data in st.session_state.inventory.items():
        current = data['current_stock']
        reorder = data['reorder_point']
        
        if current == 0:
            out_of_stock.append(item)
        elif current <= reorder:
            low_stock.append(item)
        else:
            well_stocked.append(item)
    
    return low_stock, out_of_stock, well_stocked

# Create inventory overview chart
def create_inventory_chart():
    """Create visual chart of current inventory levels"""
    if 'inventory' not in st.session_state:
        return go.Figure()
    
    items = []
    current_stock = []
    reorder_points = []
    colors = []
    
    for item, data in st.session_state.inventory.items():
        items.append(item)
        current_stock.append(data['current_stock'])
        reorder_points.append(data['reorder_point'])
        
        # Color code based on stock level
        if data['current_stock'] == 0:
            colors.append('red')
        elif data['current_stock'] <= data['reorder_point']:
            colors.append('orange')
        else:
            colors.append('green')
    
    fig = go.Figure()
    
    # Add current stock bars
    fig.add_trace(go.Bar(
        x=items,
        y=current_stock,
        name='Current Stock',
        marker_color=colors,
        text=current_stock,
        textposition='outside'
    ))
    
    # Add reorder point line
    fig.add_trace(go.Scatter(
        x=items,
        y=reorder_points,
        mode='lines+markers',
        name='Reorder Point',
        line=dict(color='red', dash='dash'),
        marker=dict(color='red', size=6)
    ))
    
    fig.update_layout(
        title='Current Inventory Levels',
        xaxis_title='Items',
        yaxis_title='Quantity',
        height=500,
        xaxis_tickangle=45,
        showlegend=True
    )
    
    return fig

# Main inventory dashboard
def main_inventory_tracker():
    st.title("Sweet & Savory Bakery - Inventory Tracker")
    st.markdown("*Real-time inventory management with visual tracking and alerts*")
    
    # Initialize inventory if not exists
    if 'inventory' not in st.session_state:
        st.session_state.inventory = initialize_mock_inventory()
    
    # Sidebar information
    st.sidebar.title("Inventory Management")
    st.sidebar.write("**Features:**")
    st.sidebar.write("• Real-time stock tracking")
    st.sidebar.write("• Visual inventory charts")
    st.sidebar.write("• Low stock alerts")
    st.sidebar.write("• Easy sell/restock interface")
    st.sidebar.write("")
    
    if st.sidebar.button("Reset Inventory"):
        st.session_state.inventory = initialize_mock_inventory()
        st.rerun()
    
    # Get current status
    low_stock, out_of_stock, well_stocked = get_inventory_status()
    
    # Summary metrics
    st.subheader("Inventory Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_items = len(st.session_state.inventory) if 'inventory' in st.session_state else 0
        st.metric("Total Items", total_items)
    
    with col2:
        st.metric("Out of Stock", len(out_of_stock), delta_color="inverse")
    
    with col3:
        st.metric("Low Stock", len(low_stock), delta_color="off")
    
    with col4:
        st.metric("Well Stocked", len(well_stocked), delta_color="normal")
    
    # Inventory management interface
    st.subheader("Quick Actions")
    
    if 'inventory' in st.session_state and st.session_state.inventory:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            available_items = sorted(list(st.session_state.inventory.keys()))
            selected_item = st.selectbox("Select Item:", available_items)
        
        with col2:
            action = st.selectbox("Action:", ["Sell", "Restock"])
        
        with col3:
            quantity = st.number_input("Quantity:", min_value=1, value=1, step=1)
        
        with col4:
            st.write("")  # Spacing
            if st.button("Execute Action", type="primary"):
                if action == "Sell":
                    new_stock = sell_item(selected_item, quantity)
                    if new_stock >= 0:
                        st.success(f"Sold {quantity} {selected_item}. Stock now: {new_stock}")
                        if new_stock <= st.session_state.inventory[selected_item]['reorder_point']:
                            st.warning(f"{selected_item} is now low in stock!")
                        if new_stock == 0:
                            st.error(f"{selected_item} is now OUT OF STOCK!")
                else:  # Restock
                    new_stock = add_inventory(selected_item, quantity)
                    st.success(f"Added {quantity} {selected_item}. Stock now: {new_stock}")
                
                st.rerun()
        
        # Show current stock for selected item
        if selected_item in st.session_state.inventory:
            item_data = st.session_state.inventory[selected_item]
            st.info(f"**{selected_item}**: {item_data['current_stock']} in stock | Reorder at: {item_data['reorder_point']} | Sales today: {item_data.get('sales_today', 0)}")
    else:
        st.error("No inventory data available. Please click 'Reset Inventory' in the sidebar.")
    
    # Visual inventory chart
    st.subheader("Inventory Levels Chart")
    fig = create_inventory_chart()
    st.plotly_chart(fig, width='stretch')
    
    # Alerts section
    st.subheader("Inventory Alerts")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if out_of_stock:
            st.error("**URGENT: Out of Stock Items**")
            for item in out_of_stock:
                st.write(f"• {item} - RESTOCK IMMEDIATELY")
        elif low_stock:
            st.warning("**Low Stock Warning**")
            for item in low_stock:
                current = st.session_state.inventory[item]['current_stock']
                reorder = st.session_state.inventory[item]['reorder_point']
                st.write(f"• {item}: {current} left (reorder at {reorder})")
        else:
            st.success("**All Items Well Stocked**")
            st.write("No immediate action needed")
    
    with col2:
        if out_of_stock or low_stock:
            st.write("**Quick Restock Options**")
            critical_items = out_of_stock + low_stock
            
            for item in critical_items[:5]:  # Show top 5 critical items
                item_data = st.session_state.inventory[item]
                needed = max(5, item_data['reorder_point'] - item_data['current_stock'] + 10)
                
                if st.button(f"Restock {item} (+{needed})", key=f"restock_{item}"):
                    add_inventory(item, needed)
                    st.success(f"Restocked {item} with {needed} units!")
                    st.rerun()
        else:
            st.info("No critical restocking needed at this time.")
    
    # Detailed inventory table
    st.subheader("Detailed Inventory Status")
    
    if 'inventory' in st.session_state and st.session_state.inventory:
        # Prepare data for table
        inventory_table = []
        for item, data in st.session_state.inventory.items():
            current = data['current_stock']
            reorder = data['reorder_point']
            max_cap = data['max_capacity']
            
            # Status indicator
            if current == 0:
                status = "OUT OF STOCK"
                status_icon = "🔴"
            elif current <= reorder:
                status = "LOW STOCK"
                status_icon = "🟡"
            else:
                status = "IN STOCK"
                status_icon = "🟢"
            
            stock_percentage = (current / max_cap) * 100 if max_cap > 0 else 0
            
            inventory_table.append({
                'Status': status_icon,
                'Item': item,
                'Current Stock': current,
                'Stock %': f"{stock_percentage:.0f}%",
                'Reorder Point': reorder,
                'Max Capacity': max_cap,
                'Sales Today': data.get('sales_today', 0),
                'Status Text': status
            })
        
        # Display table
        inventory_df = pd.DataFrame(inventory_table)
        display_columns = ['Status', 'Item', 'Current Stock', 'Stock %', 'Reorder Point', 'Sales Today', 'Status Text']
        st.dataframe(inventory_df[display_columns], width='stretch', hide_index=True)
        
        # Sales summary for today
        with st.expander("Today's Sales Summary", expanded=False):
            total_sales_today = sum([data.get('sales_today', 0) for data in st.session_state.inventory.values()])
            st.metric("Total Items Sold Today", total_sales_today)
            
            # Top selling items today
            sales_data = [(item, data.get('sales_today', 0)) for item, data in st.session_state.inventory.items() if data.get('sales_today', 0) > 0]
            if sales_data:
                sales_data.sort(key=lambda x: x[1], reverse=True)
                st.write("**Top Sellers Today:**")
                for item, sales in sales_data[:5]:
                    st.write(f"• {item}: {sales} sold")
            else:
                st.write("No sales recorded yet today.")
    else:
        st.warning("No inventory data available.")

# Main app
def main():
    st.set_page_config(page_title="Inventory Tracker", layout="wide")
    main_inventory_tracker()

if __name__ == "__main__":
    main()
