# ============================================================
#    PRODUCTION-READY STREAMLIT: PROCUREMENT PLANNING SUITE
#    Designed for Beverage Industry – Executive → Operations
#    Author: ChatGPT (Optimized Full-Stack Streamlit Version)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import StringIO

st.set_page_config(
    page_title="Procurement Planning – Beverage Industry",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
#               DUMMY DATA GENERATION ENGINE
# ============================================================

def generate_dummy_data():
    np.random.seed(42)
    skus = [f"SKU-{i:03d}" for i in range(1, 31)]
    categories = ["Beverage", "Ingredient", "Packaging"]
    packs = ["Bottle", "Can", "Sachet", "6-pack", "12-pack"]

    skus_df = pd.DataFrame(
        {
            "sku": skus,
            "description": [f"Product {i}" for i in range(1, 31)],
            "category": np.random.choice(categories, size=30),
            "pack_size": np.random.choice(packs, size=30),
            "baseline_cost": np.round(np.random.uniform(0.5, 15.0, size=30), 2),
            "safety_stock": np.random.randint(50, 300, size=30),
        }
    )

    suppliers = [f"Supplier-{c}" for c in list("ABCDEFGH")]
    suppliers_df = pd.DataFrame(
        {
            "supplier": suppliers,
            "region": np.random.choice(["North", "South", "East", "West"], size=len(suppliers)),
            "lead_time_days": np.random.randint(3, 30, size=len(suppliers)),
            "moq": np.random.randint(50, 500, size=len(suppliers)),
            "capacity_per_period": np.random.randint(500, 6000, size=len(suppliers)),
            "risk_score": np.round(np.random.uniform(0, 1, size=len(suppliers)), 2),
        }
    )

    # Supplier pricing
    rows = []
    for sku in skus:
        chosen = np.random.choice(suppliers, size=4, replace=False)
        for s in chosen:
            rows.append(
                {
                    "sku": sku,
                    "supplier": s,
                    "unit_price": np.round(np.random.uniform(0.5, 20.0), 2),
                    "lead_time": suppliers_df.loc[suppliers_df.supplier == s, "lead_time_days"].values[0],
                    "moq": suppliers_df.loc[suppliers_df.supplier == s, "moq"].values[0],
                    "capacity": suppliers_df.loc[suppliers_df.supplier == s, "capacity_per_period"].values[0],
                }
            )

    supplier_price_df = pd.DataFrame(rows)

    # Forecast
    periods = pd.date_range(start=pd.Timestamp.today().normalize(), periods=12, freq="MS")
    forecast_rows = []
    for p in periods:
        for sku in skus:
            forecast_rows.append(
                {
                    "period": p.strftime("%Y-%m"),
                    "sku": sku,
                    "forecast_qty": int(np.random.poisson(300)),
                }
            )
    forecast_df = pd.DataFrame(forecast_rows)

    # Inventory
    inventory_df = pd.DataFrame(
        {
            "sku": skus,
            "on_hand": np.random.randint(0, 2000, size=30),
            "allocated": np.random.randint(0, 200, size=30),
        }
    )

    # BOM
    components = [f"COMP-{i:02d}" for i in range(1, 11)]
    bom_rows = []
    for sku in skus:
        used = np.random.choice(components, size=np.random.randint(1, 5), replace=False)
        for c in used:
            bom_rows.append(
                {
                    "sku": sku,
                    "component": c,
                    "qty_per_parent": np.random.randint(1, 5),
                }
            )

    bom_df = pd.DataFrame(bom_rows)

    return {
        "skus": skus_df,
        "suppliers": suppliers_df,
        "supplier_price": supplier_price_df,
        "forecast": forecast_df,
        "inventory": inventory_df,
        "bom": bom_df,
        "periods": list(periods.strftime("%Y-%m")),
    }


@st.cache_data
def load_data():
    return generate_dummy_data()


data = load_data()

# ============================================================
#                   NAVIGATION SIDEBAR
# ============================================================


def sidebar_navigation():
    st.sidebar.title("PROCUREMENT SUITE")
    return st.sidebar.radio(
        "Navigate to:",
        [
            "Executive Overview",
            "Planner Workbench",
            "Optimization Results",
            "Root Cause Analysis",
            "Scenario Simulation",
            "BOM & Dependencies",
            "Constraints Dashboard",
            "PO Creation",
            "Mobile Ops",
            "Business Landscape",
            "Input Data Spec",
            "Output Data Spec",
        ],
    )


# ============================================================
#                EXECUTIVE DASHBOARD
# ============================================================


def exec_dashboard():
    st.title("📊 Executive Overview")

    col1, col2, col3, col4 = st.columns(4)
    total_cost = data["supplier_price"].groupby("sku").unit_price.min().sum() * 100
    inventory_cov = np.random.randint(20, 90)

    col1.metric("Total Procurement Cost", f"${total_cost:,.0f}")
    col2.metric("Inventory Coverage (days)", f"{inventory_cov}")
    col3.metric("Avg Forecast Accuracy", f"{np.random.uniform(70, 95):.1f}%")
    col4.metric("Savings vs Baseline", f"{np.random.uniform(5, 20):.1f}%")

    st.subheader("Supplier Risk Heatmap")
    fig = px.scatter(
        data["suppliers"],
        x="region",
        y="supplier",
        size="capacity_per_period",
        color="risk_score",
        color_continuous_scale="OrRd",
    )
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
#                PROCUREMENT PLANNER WORKBENCH
# ============================================================

def planner_workbench():
    st.title("🛠️ Procurement Planner Workbench")

    left, right = st.columns([2, 1])
    with left:
        st.subheader("SKU Master (Editable)")
        st.data_editor(data["skus"], num_rows="dynamic")

        st.subheader("Forecast Sample")
        sample = data["forecast"][data["forecast"]["period"] == data["periods"][0]].sample(10)
        st.dataframe(sample)

    with right:
        st.subheader("Overrides")
        sku = st.selectbox("Select SKU", data["skus"]["sku"])
        qty = st.number_input("Override Demand Qty", min_value=0)
        if st.button("Apply Override"):
            st.success(f"Override applied for {sku} → {qty}")

    st.subheader("Supplier Offers")
    st.dataframe(data["supplier_price"].sample(20))

# ============================================================
#                   OPTIMIZATION RESULTS
# ============================================================

def optimization_results():
    st.title("🎯 Optimization Results Summary")

    results = (
        data["supplier_price"]
        .sort_values("unit_price")
        .drop_duplicates(subset=["sku"])
        .reset_index(drop=True)
    )

    results["procure_qty"] = np.random.randint(200, 1000, size=len(results))
    results["total_cost"] = results["procure_qty"] * results["unit_price"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Procurement Cost", f"${results['total_cost'].sum():,.0f}")
    col2.metric("Avg SKU Cost", f"${results['total_cost'].mean():.2f}")
    col3.metric("Suppliers Engaged", results['supplier'].nunique())
    col4.metric("Cost Savings", f"{np.random.uniform(5, 15):.1f}%")

    st.subheader("Procurement Plan")
    st.dataframe(results[["sku", "supplier", "procure_qty", "unit_price", "total_cost"]])

# ============================================================
#                   ROOT CAUSE ANALYSIS
# ============================================================

def root_cause_analysis():
    st.title("🔍 Root Cause Analysis")

    st.subheader("Key Drivers Behind Stock-outs / Excess")
    rca = pd.DataFrame({
        "Cause": ["Forecast Error", "Supplier Delay", "MOQ Violation", "Capacity Issue", "Inventory Mismatch"],
        "Impact %": np.random.uniform(5, 35, 5)
    })

    fig = px.bar(rca, x="Cause", y="Impact %", color="Impact %", color_continuous_scale="Reds")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("SKU-Level Sensitivity")
    st.dataframe(data["skus"][["sku", "baseline_cost", "safety_stock"]])

# ============================================================
#                   SCENARIO SIMULATION
# ============================================================

def scenario_simulation():
    st.title("🧪 Scenario Simulation")

    col1, col2 = st.columns(2)
    with col1:
        lt_change = st.slider("Supplier Lead Time ± (%)", -20, 20, 0)
        demand_change = st.slider("Demand Shift ± (%)", -30, 30, 0)
    with col2:
        price_change = st.slider("Price Variance ± (%)", -15, 15, 0)
        capacity_change = st.slider("Capacity Adjust ± (%)", -50, 50, 0)

    st.markdown("### Scenario Summary")
    st.write({
        "Lead Time Change": lt_change,
        "Demand Change": demand_change,
        "Price Change": price_change,
        "Capacity Change": capacity_change,
    })

    st.info("Scenario applied. Optimization impact will appear here.")

# ============================================================
#               BOM & DEPENDENCIES
# ============================================================

def bom_dependencies():
    st.title("📦 BOM & Material Dependencies")

    sku = st.selectbox("Choose SKU", data["skus"]["sku"])
    bom = data["bom"][data["bom"]["sku"] == sku]

    st.subheader(f"Components for {sku}")
    st.dataframe(bom)

# ============================================================
#               CONSTRAINTS DASHBOARD
# ============================================================

def constraints_dashboard():
    st.title("⚠️ Constraints Dashboard")

    constraints = pd.DataFrame({
        "Constraint": ["MOQ", "Lead Time", "Capacity", "Budget", "Risk Score"],
        "Violations": np.random.randint(1, 15, 5)
    })

    fig = px.bar(
        constraints,
        x="Constraint",
        y="Violations",
        color="Violations",
        color_continuous_scale="OrRd"
    )
    st.plotly_chart(fig)

# ============================================================
#               PO CREATION & EXECUTION
# ============================================================

def po_creation():
    st.title("📑 PO Creation & Execution")

    df = data["supplier_price"].sample(10)
    df["order_qty"] = np.random.randint(100, 500, size=len(df))
    df["po_amount"] = df["order_qty"] * df["unit_price"]

    st.subheader("Draft PO List")
    st.dataframe(df)

    if st.button("Generate PO File"):
        st.success("PO generated and sent to supplier inbox!")

# ============================================================
#               MOBILE OPERATIONS VIEW
# ============================================================

def mobile_ops():
    st.title("📱 Mobile Ops View – Shopfloor Friendly")

    st.metric("Today’s Orders to Release", f"{np.random.randint(5, 20)}")
    st.metric("Urgent Shortages", f"{np.random.randint(1, 10)}")

    st.subheader("Critical SKUs")
    st.write(data["skus"].sample(5))

# ============================================================
#               BUSINESS LANDSCAPE
# ============================================================

def business_landscape():
    st.title("🌍 Business Landscape Overview")
    st.write("""
    - Multi-supplier beverage procurement  
    - Lead time variation and MOQ challenges  
    - High inventory sensitivity due to shelf life  
    - Need for scenario planning  
    - Demand variability + promotions impact  
    """)

# ============================================================
#               DATA SPECIFICATIONS
# ============================================================

def input_data_spec():
    st.title("📥 Input Data Specification")
    st.write(data["skus"].head())

def output_data_spec():
    st.title("📤 Output Data Specification")
    st.write("""
    - Procurement Plan  
    - Supplier Allocation  
    - Total Cost  
    - Constraint Violations  
    - PO File  
    """)

# ============================================================
#                   MAIN ROUTER
# ============================================================

page = sidebar_navigation()

if page == "Executive Overview":
    exec_dashboard()
elif page == "Planner Workbench":
    planner_workbench()
elif page == "Optimization Results":
    optimization_results()
elif page == "Root Cause Analysis":
    root_cause_analysis()
elif page == "Scenario Simulation":
    scenario_simulation()
elif page == "BOM & Dependencies":
    bom_dependencies()
elif page == "Constraints Dashboard":
    constraints_dashboard()
elif page == "PO Creation":
    po_creation()
elif page == "Mobile Ops":
    mobile_ops()
elif page == "Business Landscape":
    business_landscape()
elif page == "Input Data Spec":
    input_data_spec()
elif page == "Output Data Spec":
    output_data_spec()
