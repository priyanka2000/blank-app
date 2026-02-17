import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import StringIO


st.set_page_config(page_title="Procurement Planning — Beverage", layout="wide")


def generate_dummy_data():
    np.random.seed(42)
    skus = [f"SKU-{i:03d}" for i in range(1, 21)]
    categories = ["Beverage", "Ingredient", "Pack"]
    packs = ["6-pack", "12-pack", "Bottle", "Sachet"]

    skus_df = pd.DataFrame({
        "sku": skus,
        "description": [f"Product {i}" for i in range(1, 21)],
        "category": np.random.choice(categories, size=20),
        "pack_size": np.random.choice(packs, size=20),
        "uom": "EA",
        "baseline_cost": np.round(np.random.uniform(0.5, 15.0, size=20), 2),
        "safety_stock": np.random.randint(50, 300, size=20),
    })

    suppliers = [f"Supplier-{c}" for c in list("ABCDEFGH")]
    suppliers_df = pd.DataFrame({
        "supplier": suppliers,
        "region": np.random.choice(["North","South","East","West"], size=len(suppliers)),
        "lead_time_days": np.random.randint(3, 30, size=len(suppliers)),
        "moq": np.random.randint(50, 500, size=len(suppliers)),
        "capacity_per_period": np.random.randint(500, 5000, size=len(suppliers)),
        "risk_score": np.round(np.random.uniform(0,1,size=len(suppliers)),2)
    })

    # supplier-price mapping per SKU (random)
    prices = []
    rows = []
    for sku in skus:
        offered = np.random.choice(suppliers, size=4, replace=False)
        for s in offered:
            rows.append({
                "sku": sku,
                "supplier": s,
                "unit_price": np.round(np.random.uniform(0.5, 20.0),2),
                "lead_time": int(suppliers_df.loc[suppliers_df.supplier==s,"lead_time_days"].values[0]),
                "moq": int(suppliers_df.loc[suppliers_df.supplier==s,"moq"].values[0]),
                "capacity": int(suppliers_df.loc[suppliers_df.supplier==s,"capacity_per_period"].values[0]),
            })
    supplier_price_df = pd.DataFrame(rows)

    forecast_periods = pd.date_range(start=pd.Timestamp.today().normalize(), periods=12, freq="MS")
    forecast_rows = []
    for p in forecast_periods:
        for sku in skus:
            forecast_rows.append({"period": p.strftime("%Y-%m"), "sku": sku, "forecast_qty": int(np.random.poisson(200))})
    forecast_df = pd.DataFrame(forecast_rows)

    inventory_df = pd.DataFrame({
        "sku": skus,
        "on_hand": np.random.randint(0, 1000, size=20),
        "allocated": np.random.randint(0, 200, size=20),
    })

    # Simple BOM: each SKU has 1-3 components from among 8 components
    components = [f"COMP-{i:02d}" for i in range(1,9)]
    bom_rows = []
    for sku in skus:
        comps = np.random.choice(components, size=np.random.randint(1,4), replace=False)
        for c in comps:
            bom_rows.append({"sku": sku, "component": c, "qty_per_parent": np.random.randint(1,6)})
    bom_df = pd.DataFrame(bom_rows)

    return {
        "skus": skus_df,
        "suppliers": suppliers_df,
        "supplier_price": supplier_price_df,
        "forecast": forecast_df,
        "inventory": inventory_df,
        "bom": bom_df,
        "periods": list(forecast_periods.strftime("%Y-%m"))
    }


@st.cache_data
def get_data():
    return generate_dummy_data()


data = get_data()


def sidebar_nav():
    st.sidebar.title("Procurement Planner")
    page = st.sidebar.radio("Navigate", [
        "Executive Overview",
        "Procurement Planner",
        "Optimization Results",
        "Scenario Simulation",
        "BOM & Dependencies",
        "Constraints Dashboard",
        "PO Creation",
        "Mobile Ops (Preview)",
        "Business Landscape",
        "Input Data Spec",
        "Output Data Spec",
    ])
    return page


def exec_overview(data):
    st.header("**Executive Overview Dashboard**")
    col1, col2, col3, col4 = st.columns(4)
    total_cost = (data['supplier_price'].groupby('sku').unit_price.min().sum() * 1000) / 10
    supplier_alloc_pct = data['supplier_price'].groupby('supplier').sku.nunique().nlargest(5)
    cost_savings = np.round(np.random.uniform(1,10),2)

    col1.metric("Total Procurement Cost", f"${total_cost:,.0f}")
    col2.metric("Top Supplier Allocation %", f"{supplier_alloc_pct.iloc[0] if len(supplier_alloc_pct)>0 else 0}")
    col3.metric("Cost Savings vs Baseline", f"{cost_savings}%")
    col4.metric("Inventory Coverage (days)", f"{np.random.randint(10,90)}")

    st.subheader("Supplier Risk Heatmap")
    fig = px.scatter(data['suppliers'], x='region', y='supplier', size='capacity_per_period', color='risk_score', color_continuous_scale='OrRd', title='Supplier risk by region')
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Forecast Accuracy (sample KPIs)")
    kpi_df = pd.DataFrame({"metric": ["MAPE","Bias","FVA"], "value": [np.round(np.random.uniform(0,0.3),2), np.round(np.random.uniform(-0.1,0.1),2), np.round(np.random.uniform(0,0.2),2)]})
    st.table(kpi_df)


def procurement_planner(data):
    st.header("**Procurement Planner Workbench**")
    st.markdown("Planner-friendly editable grid and controls for overrides")
    skus = data['skus'].copy()
    inventory = data['inventory'].copy()
    forecast = data['forecast'][data['forecast']['period']==data['periods'][0]]

    left, right = st.columns([2,1])
    with left:
        st.subheader("SKU Master (editable)")
        edited = st.data_editor(skus, num_rows="dynamic")
        st.subheader("Forecast (sample period)")
        st.dataframe(forecast.sample(10))

    with right:
        st.subheader("Overrides")
        sku_override = st.selectbox("Select SKU to override demand", skus['sku'].tolist())
        override_qty = st.number_input("Override quantity", min_value=0, value=0)
        if st.button("Apply Override"):
            st.success(f"Override applied for {sku_override}: {override_qty}")

    st.subheader("Supplier Offers")
    st.dataframe(data['supplier_price'].sample(20))

    st.markdown("### Run Optimization")
    columns = st.columns(3)
    with columns[0]:
        periods = st.multiselect("Select periods", options=data['periods'], default=data['periods'][:3])
    with columns[1]:
        opt_mode = st.selectbox("Optimization objective", ["Minimize Cost", "Maximize Service Level"] )
    with columns[2]:
        run = st.button("RUN OPTIMIZATION")
    if run:
        st.info("Optimization complete — showing a preview of results in Optimization Results tab")


def optimization_results(data):
    st.markdown("""
    <style>
    .opt-header { font-size: 32px; font-weight: bold; color: #1f77b4; margin-bottom: 10px; }
    .opt-subheader { font-size: 18px; font-weight: 600; color: #2c3e50; margin-top: 20px; margin-bottom: 10px; }
    .metric-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; color: white; text-align: center; }
    .constraint-good { background-color: #d4edda; border-left: 4px solid #28a745; padding: 12px; margin: 8px 0; border-radius: 4px; }
    .constraint-warning { background-color: #fff3cd; border-left: 4px solid #ffc107; padding: 12px; margin: 8px 0; border-radius: 4px; }
    .constraint-error { background-color: #f8d7da; border-left: 4px solid #dc3545; padding: 12px; margin: 8px 0; border-radius: 4px; }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="opt-header">🎯 Optimization Results</div>', unsafe_allow_html=True)
    st.markdown("Material-wise procurement quantities, supplier allocations, and cost optimization summary")
    
    # Simulated results - select lowest cost supplier per SKU
    results = data['supplier_price'].sort_values('unit_price').drop_duplicates(subset=['sku'], keep='first').reset_index(drop=True)
    results = results.merge(data['skus'][['sku','description']], on='sku', how='left')
    results['procure_qty'] = np.random.randint(100,1000,size=len(results))
    results['total_cost'] = results['procure_qty'] * results['unit_price']
    
    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    total_procurement = results['total_cost'].sum()
    avg_cost_per_sku = results['total_cost'].mean()
    supplier_count = results['supplier'].nunique()
    savings_pct = np.round(np.random.uniform(3, 15), 1)
    
    with col1:
        st.metric("Total Procurement Cost", f"${total_procurement:,.0f}", f"+{np.random.randint(2,8)}%")
    with col2:
        st.metric("Avg Cost per SKU", f"${avg_cost_per_sku:,.0f}", f"-{savings_pct}%", delta_color="inverse")
    with col3:
        st.metric("Suppliers Engaged", supplier_count)
    with col4:
        st.metric("Cost Savings", f"{savings_pct}%", "vs Baseline")
    
    # Display results table with better formatting
    st.markdown('<div class="opt-subheader">📋 Procurement Plan Details</div>', unsafe_allow_html=True)
    display_cols = ['sku','description','supplier','procure_qty','unit_price','total_cost']
    sample_size = min(10, len(results))
    display_df = results[display_cols].sample(n=sample_size).sort_values('total_cost', ascending=False).reset_index(drop=True)
    display_df = display_df.rename(columns={
        'sku': 'SKU',
        'description': 'Description',
        'supplier': 'Supplier',
        'procure_qty': 'Qty',
        'unit_price': 'Unit Price',
        'total_cost': 'Total Cost'
    })
    display_df['Unit Price'] = display_df['Unit Price'].apply(lambda x: f"${x:.2f}")
    display_df['Total Cost'] = display_df['Total Cost'].apply(lambda x: f"${x:,.0f}")
    st.dataframe(display_df, use_container_width=True)
    
    # Constraint Alerts with better styling
    st.markdown('<div class="opt-subheader">⚠️ Constraint Analysis</div>', unsafe_allow_html=True)
    col_alert1, col_alert2, col_alert3 = st.columns(3)
    with col_alert1:
        st.markdown('<div class="constraint-error"><strong>❌ MOQ Violation</strong><br/>SKU-005 at Supplier-A<br/><small>Suggested: 500 EA</small></div>', unsafe_allow_html=True)
    with col_alert2:
        st.markdown('<div class="constraint-warning"><strong>⏱️ Lead-time Risk</strong><br/>Supplier-C (25 days)<br/><small>Safety buffer: 2 days</small></div>', unsafe_allow_html=True)
    with col_alert3:
        st.markdown('<div class="constraint-error"><strong>📦 Capacity Exceeded</strong><br/>Supplier-F 92% utilized<br/><small>Max: 5000 units/period</small></div>', unsafe_allow_html=True)
    
    # Feasibility Summary
    st.markdown('<div class="opt-subheader">✅ Feasibility Summary</div>', unsafe_allow_html=True)
    feasible_count = np.random.randint(17, 20)
    col_feas1, col_feas2 = st.columns(2)
    with col_feas1:
        st.markdown(f'<div class="constraint-good"><strong>Feasible Plans: {feasible_count}/20 SKUs</strong><br/>Plan can be executed without major constraint violations</div>', unsafe_allow_html=True)
    with col_feas2:
        fig_feas = px.pie(values=[feasible_count, 20-feasible_count], names=['Feasible', 'At Risk'], color_discrete_sequence=['#28a745', '#ffc107'])
        fig_feas.update_layout(height=200, margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig_feas, use_container_width=True)
    
    # Export button
    st.markdown('<div class="opt-subheader">📥 Export & Actions</div>', unsafe_allow_html=True)
    col_export1, col_export2 = st.columns(2)
    with col_export1:
        csv = results.to_csv(index=False)
        st.download_button(
            label="📊 Export to CSV",
            data=csv,
            file_name='procurement_plan.csv',
            mime='text/csv',
            use_container_width=True
        )
    with col_export2:
        if st.button("✅ Approve & Create POs", use_container_width=True):
            st.success("Purchase Orders created successfully! 🎉")


def scenario_simulation(data):
    st.header("**Scenario Simulation**")
    st.markdown("Predefined scenarios and ad-hoc scenario builder")
    col1, col2 = st.columns(2)
    with col1:
        scenario = st.selectbox("Choose scenario", ["Base","S1: Demand Surge","S2: Supplier Delay","S3: Price Increase","S4: Supply Shortage"])
        st.write("Scenario quick controls")
        demand_pct = st.slider("Demand % change", -50, 200, 0)
        lead_time_pct = st.slider("Lead time % change", -50, 200, 0)
        price_pct = st.slider("Price % change", -50, 200, 0)
        if st.button("Apply Scenario"):
            st.success(f"Applied {scenario}")
    with col2:
        st.subheader("Scenario impact (sample)")
        kpis = pd.DataFrame({"kpi":["Total Cost","Service Level","Stock-out Probability"], "base":[1000,95,2], "scenario":[1000*(1+price_pct/100), 95*(1-demand_pct/100), 2*(1+demand_pct/100)]})
        st.table(kpis)

    st.subheader("Visualization")
    fig = px.bar(kpis, x='kpi', y=['base','scenario'], barmode='group')
    st.plotly_chart(fig, use_container_width=True)


def bom_view(data):
    st.header("**BOM & Material Dependency View**")
    st.markdown("Multi-level BOM visualization (flat sample)")
    st.dataframe(data['bom'].sample(20))
    st.subheader("Shortage propagation (sample)")
    st.info("If a component is short, all SKUs using it may be impacted — sample view below")
    comp = data['bom']['component'].value_counts().reset_index()
    comp.columns = ['component','used_by']
    st.bar_chart(comp.set_index('component'))


def constraints_dashboard(data):
    st.header("**Constraints Dashboard**")
    st.subheader("MOQ vs Suggested Qty")
    moq = data['supplier_price'].groupby('supplier').moq.mean().reset_index()
    st.table(moq)
    st.subheader("Supplier capacity utilization (sample)")
    caps = data['suppliers'][['supplier','capacity_per_period','lead_time_days']]
    st.dataframe(caps)


def po_creation(data):
    st.header("**PO Creation & Execution View**")
    st.markdown("Create and preview Purchase Orders — sample interactive form")
    sku = st.selectbox("SKU", data['skus']['sku'].tolist())
    supplier_options = data['supplier_price'][data['supplier_price']['sku']==sku]['supplier'].unique().tolist()
    supplier = st.selectbox("Select supplier", supplier_options)
    qty = st.number_input("Quantity", min_value=0, value=500)
    eta = st.date_input("Planned delivery date")
    if st.button("Create PO"):
        st.success(f"PO created: {sku} x {qty} from {supplier}, ETA {eta}")


def mobile_ops(data):
    st.header("**Mobile UI (Preview)**")
    st.markdown("Compact view for daily ops: arrivals, acknowledgements, quick approvals")
    st.subheader("Inbound arrivals (today)")
    arrivals = pd.DataFrame({"po_id":[f"PO-{i:04d}" for i in range(1,6)], "sku": np.random.choice(data['skus']['sku'],5), "qty": np.random.randint(10,500,5), "status": np.random.choice(["Expected","Arrived","Delayed"],5)})
    st.table(arrivals)


def business_landscape(data):
    st.header("**Business Landscape — Beverage Industry**")
    st.markdown("Overview of SKUs, supplier network, lead times, BOM connections and risk hotspots")
    st.subheader("SKUs & Categories")
    st.dataframe(data['skus'])
    st.subheader("Procurement cycle swimlane (sample)")
    st.markdown("Order -> Supplier Ack -> Production -> Delivery -> Receiving")


def input_data_spec(data):
    st.header("**Input Data Specification**")
    st.markdown("Upload / validate all input masters and pipelines")
    st.subheader("SKU Master")
    st.dataframe(data['skus'])
    st.subheader("Supplier Master")
    st.dataframe(data['suppliers'])
    st.subheader("Upload area")
    upl = st.file_uploader("Upload CSV (SKU / Supplier / Forecast)")
    if upl is not None:
        df = pd.read_csv(upl)
        st.success(f"Uploaded {df.shape[0]} rows")


def output_data_spec(data):
    st.markdown("""
    <style>
    .output-header { font-size: 32px; font-weight: bold; color: #27ae60; margin-bottom: 10px; }
    .output-subheader { font-size: 18px; font-weight: 600; color: #2c3e50; margin-top: 20px; margin-bottom: 10px; }
    .status-badge { display: inline-block; padding: 6px 12px; border-radius: 20px; font-size: 12px; font-weight: 600; }
    .status-feasible { background-color: #d4edda; color: #155724; }
    .status-risk { background-color: #fff3cd; color: #856404; }
    .output-card { background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 10px 0; border-left: 4px solid #27ae60; }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="output-header">📊 Output Data Specification</div>', unsafe_allow_html=True)
    st.markdown("Complete procurement plan with material-wise quantities, supplier allocations, period-wise costs, inventory projections, and feasibility assessment")
    
    # Generate output spec data - select lowest cost supplier per SKU
    sample_out = data['supplier_price'].sort_values('unit_price').drop_duplicates(subset=['sku'], keep='first').reset_index(drop=True)
    sample_out['suggested_qty'] = np.random.randint(100, 1000, size=len(sample_out))
    sample_out['period_1_cost'] = sample_out['unit_price'] * sample_out['suggested_qty'] * 1.1
    sample_out['period_2_cost'] = sample_out['unit_price'] * sample_out['suggested_qty'] * 0.95
    sample_out['inventory_proj'] = np.random.randint(50, 500, size=len(sample_out))
    sample_out['feasible'] = np.random.choice([True,False], size=len(sample_out), p=[0.85,0.15])
    
    # Summary metrics
    st.markdown('<div class="output-subheader">📈 Output Summary</div>', unsafe_allow_html=True)
    col_summary1, col_summary2, col_summary3, col_summary4 = st.columns(4)
    total_output_cost = sample_out['period_1_cost'].sum() + sample_out['period_2_cost'].sum()
    feasible_count = sample_out['feasible'].sum()
    
    with col_summary1:
        st.metric("Total Output Cost (2 Periods)", f"${total_output_cost:,.0f}")
    with col_summary2:
        st.metric("Feasible SKUs", f"{feasible_count}/{len(sample_out)}")
    with col_summary3:
        st.metric("Total Inventory Projected", f"{sample_out['inventory_proj'].sum():,} units")
    with col_summary4:
        st.metric("Avg Qty per SKU", f"{sample_out['suggested_qty'].mean():.0f}")
    
    # Detailed output table
    st.markdown('<div class="output-subheader">📋 Material-wise Procurement Plan</div>', unsafe_allow_html=True)
    display_size = min(15, len(sample_out))
    output_display = sample_out[['sku','supplier','period_1_cost','period_2_cost','suggested_qty','inventory_proj','feasible']].sample(n=display_size).sort_values('period_1_cost', ascending=False).reset_index(drop=True)
    output_display = output_display.rename(columns={
        'sku': 'SKU',
        'supplier': 'Supplier',
        'period_1_cost': 'Period 1 Cost',
        'period_2_cost': 'Period 2 Cost',
        'suggested_qty': 'Qty',
        'inventory_proj': 'Inventory Proj',
        'feasible': 'Feasible'
    })
    output_display['Period 1 Cost'] = output_display['Period 1 Cost'].apply(lambda x: f"${x:,.0f}")
    output_display['Period 2 Cost'] = output_display['Period 2 Cost'].apply(lambda x: f"${x:,.0f}")
    output_display['Feasible'] = output_display['Feasible'].apply(lambda x: '✅ Yes' if x else '⚠️ At Risk')
    st.dataframe(output_display, use_container_width=True)
    
    # Period-wise cost breakdown
    st.markdown('<div class="output-subheader">💰 Period-wise Cost Projection</div>', unsafe_allow_html=True)
    period_costs = pd.DataFrame({
        'Period': ['Period 1', 'Period 2'],
        'Cost': [sample_out['period_1_cost'].sum(), sample_out['period_2_cost'].sum()]
    })
    fig_periods = px.bar(period_costs, x='Period', y='Cost', color='Period',
                        color_discrete_sequence=['#3498db', '#2ecc71'],
                        title='Projected Cost by Period')
    fig_periods.update_layout(height=300, showlegend=False, hovermode='x unified')
    fig_periods.update_traces(hovertemplate='<b>%{x}</b><br>Cost: $%{y:,.0f}<extra></extra>')
    st.plotly_chart(fig_periods, use_container_width=True)
    
    # Feasibility summary with warnings
    st.markdown('<div class="output-subheader">⚠️ Feasibility & Risk Assessment</div>', unsafe_allow_html=True)
    col_risk1, col_risk2 = st.columns(2)
    with col_risk1:
        st.markdown(f'''<div class="output-card">
        <strong>✅ Feasible Records: {feasible_count}</strong><br/>
        {(feasible_count/len(sample_out)*100):.1f}% of SKUs can be procured as planned
        </div>''', unsafe_allow_html=True)
    with col_risk2:
        st.markdown(f'''<div class="output-card">
        <strong>⚠️ At-Risk Records: {len(sample_out)-feasible_count}</strong><br/>
        May require supplier negotiation or modified quantities
        </div>''', unsafe_allow_html=True)
    
    # Shortage analysis
    st.markdown('<div class="output-subheader">📉 Shortage & Service Level Analysis</div>', unsafe_allow_html=True)
    shortage_scenarios = pd.DataFrame({
        'Scenario': ['Current Plan', 'Conservative (-10%)', 'Aggressive (+15%)'],
        'Shortage Risk': [2.5, 5.8, 1.2],
        'Service Level': [97.5, 94.2, 98.8]
    })
    fig_shortage = px.line(shortage_scenarios, x='Scenario', y=['Shortage Risk', 'Service Level'],
                           markers=True, title='Service Level vs Shortage Risk')
    fig_shortage.update_layout(height=300, hovermode='x unified')
    st.plotly_chart(fig_shortage, use_container_width=True)
    
    # Export options
    st.markdown('<div class="output-subheader">📥 Export & Download Options</div>', unsafe_allow_html=True)
    col_exp1, col_exp2, col_exp3 = st.columns(3)
    
    with col_exp1:
        csv_export = sample_out.to_csv(index=False)
        st.download_button(
            label="📊 Export to CSV",
            data=csv_export,
            file_name='output_specification.csv',
            mime='text/csv',
            use_container_width=True
        )
    
    with col_exp2:
        # Create Excel-like format for better reporting
        excel_data = sample_out[['sku','supplier','suggested_qty','period_1_cost','period_2_cost','inventory_proj','feasible']].to_csv(index=False)
        st.download_button(
            label="📈 Export Summary Report",
            data=excel_data,
            file_name='output_summary.csv',
            mime='text/csv',
            use_container_width=True
        )
    
    with col_exp3:
        if st.button("✨ Generate Executive Summary", use_container_width=True):
            st.info("📄 Executive Summary generated and sent to stakeholders!")


PAGE_FUNCS = {
    "Executive Overview": exec_overview,
    "Procurement Planner": procurement_planner,
    "Optimization Results": optimization_results,
    "Scenario Simulation": scenario_simulation,
    "BOM & Dependencies": bom_view,
    "Constraints Dashboard": constraints_dashboard,
    "PO Creation": po_creation,
    "Mobile Ops (Preview)": mobile_ops,
    "Business Landscape": business_landscape,
    "Input Data Spec": input_data_spec,
    "Output Data Spec": output_data_spec,
}


def main():
    page = sidebar_nav()
    func = PAGE_FUNCS.get(page)
    if func:
        func(data)


if __name__ == '__main__':
    main()

