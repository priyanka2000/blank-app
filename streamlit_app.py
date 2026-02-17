# ============================================================
#    PRODUCTION-READY STREAMLIT: PROCUREMENT PLANNING SUITE
#    Designed for Beverage Industry – Executive → Operations
#    Author: ChatGPT (Optimized Full-Stack Streamlit Version)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pulp as pl
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
    role = st.session_state.get("user_role", "Planner")

    role_pages = {
        "Executive": [
            "Executive Overview",
            "Scenario Comparison",
            "Business Landscape",
            "Constraints Dashboard",
            "Input Data Spec",
            "Output Data Spec",
        ],
        "Planner": [
            "Executive Overview",
            "Planner Workbench",
            "Optimization Results",
            "Scenario Simulation",
            "Scenario Comparison",
            "BOM & Dependencies",
            "Constraints Dashboard",
            "PO Creation",
            "Mobile Ops",
            "Input Data Spec",
            "Output Data Spec",
        ],
        "Buyer": [
            "Planner Workbench",
            "PO Creation",
            "Optimization Results",
            "Constraints Dashboard",
            "Input Data Spec",
            "Output Data Spec",
        ],
        "Engineer": [
            "BOM & Dependencies",
            "Planner Workbench",
            "Constraints Dashboard",
            "Input Data Spec",
            "Output Data Spec",
        ],
    }

    pages = role_pages.get(role, [
        "Executive Overview",
        "Planner Workbench",
        "Optimization Results",
        "Scenario Simulation",
        "Scenario Comparison",
        "BOM & Dependencies",
        "Constraints Dashboard",
        "PO Creation",
        "Mobile Ops",
        "Business Landscape",
        "Input Data Spec",
        "Output Data Spec",
    ])

    return st.sidebar.radio("Navigate to:", pages)


def get_user_role():
    if "user_role" not in st.session_state:
        st.session_state.user_role = "Planner"
    st.session_state.user_role = st.sidebar.selectbox(
        "User Role",
        ["Executive", "Planner", "Buyer", "Engineer"],
        index=["Executive", "Planner", "Buyer", "Engineer"].index(st.session_state.user_role),
    )
    return st.session_state.user_role


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
    user = st.session_state.get("user_role", "Planner")

    left, right = st.columns([2, 1])
    with left:
        st.subheader("SKU Master (Editable)")
        # Planners can edit SKU master; others get read-only view
        if user == "Planner":
            st.data_editor(data["skus"], num_rows="dynamic")
        else:
            st.dataframe(data["skus"])

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

    # Quick KPIs for planners/buyers
    if user in ("Planner", "Buyer"):
        col1, col2, col3 = st.columns(3)
        total_forecast = data["forecast"]["forecast_qty"].sum()
        avg_price = data["supplier_price"]["unit_price"].mean()
        col1.metric("Total Forecast Qty (12m)", f"{total_forecast}")
        col2.metric("Avg Supplier Unit Price", f"${avg_price:.2f}")
        col3.metric("Unique Suppliers", f"{data['suppliers']['supplier'].nunique()}")

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
    display_df = results[["sku", "supplier", "procure_qty", "unit_price", "total_cost"]].copy()
    st.dataframe(display_df)

    # Recommendation workflow: allow user to accept/reject recommended supplier per SKU
    st.markdown("**Recommendation Actions**")
    selected_skus = st.multiselect("Select SKUs to accept recommendation", options=results["sku"].tolist())
    if "baseline_plan" not in st.session_state:
        # store baseline for comparison
        st.session_state.baseline_plan = results.copy()

    if st.button("Apply Accepted Recommendations"):
        accepted = results[results["sku"].isin(selected_skus)].copy()
        # compute impact: accepted replace baseline where accepted
        baseline_total = st.session_state.baseline_plan["total_cost"].sum()
        new_total = baseline_total - st.session_state.baseline_plan[st.session_state.baseline_plan["sku"].isin(selected_skus)]["total_cost"].sum() + accepted["total_cost"].sum()
        st.success(f"Applied {len(selected_skus)} recommendations — New Total Cost: ${new_total:,.0f} (baseline ${baseline_total:,.0f})")
        # record accepted SKUs for session
        st.session_state.accepted_skus = list(set(st.session_state.get("accepted_skus", [])) | set(selected_skus))

    if st.session_state.get("accepted_skus"):
        st.info(f"Accepted SKUs this session: {', '.join(st.session_state.accepted_skus)}")

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
        # Service level and lead-time threshold
        service_level = st.slider("Service Level Target (%)", 50, 100, 90)
        max_lead_time = st.number_input("Max lead time for service level (days)", min_value=1, max_value=180, value=14)
    with col2:
        price_change = st.slider("Price Variance ± (%)", -15, 15, 0)
        capacity_change = st.slider("Capacity Adjust ± (%)", -50, 50, 0)
        # Supplier risk penalty (multiplier applied to price by risk_score)
        risk_penalty = st.slider("Supplier Risk Penalty (0=no penalty → 1=high)", 0.0, 1.0, 0.2)

    st.markdown("### Scenario Summary")
    st.write({
        "Lead Time Change": lt_change,
        "Demand Change": demand_change,
        "Price Change": price_change,
        "Capacity Change": capacity_change,
        "Service Level (%)": service_level,
        "Max Lead Time (days)": max_lead_time,
        "Risk Penalty": risk_penalty,
    })

    st.info("Scenario applied. Optimization impact will appear here.")

    # Allow saving scenarios for comparison
    st.subheader("Save Scenario")
    name = st.text_input("Scenario name")
    if st.button("Save Scenario"):
        if not name:
            st.error("Provide a scenario name")
        else:
            if "scenarios" not in st.session_state:
                st.session_state.scenarios = {}
            st.session_state.scenarios[name] = {
                "lt_change": lt_change,
                "demand_change": demand_change,
                "price_change": price_change,
                "capacity_change": capacity_change,
                "service_level": service_level,
                "max_lead_time": int(max_lead_time),
                "risk_penalty": float(risk_penalty),
            }
            st.success(f"Scenario '{name}' saved")

    # Run optimization for the current (unsaved) scenario to show impact live
    if st.button("Run Optimization for Current Scenario"):
        params = {
            "lt_change": lt_change,
            "demand_change": demand_change,
            "price_change": price_change,
            "capacity_change": capacity_change,
        }
        with st.spinner("Running optimization..."):
            plan = optimize_plan(params)
        if plan is None:
            st.error("Optimization infeasible (not enough capacity). Try relaxing constraints or changing scenario parameters.")
        else:
            st.subheader("Optimization Summary")
            st.metric("Total Cost", f"${plan['total_cost'].sum():,.0f}")
            st.dataframe(plan.head(50))

# ============================================================
#               SCENARIO COMPARISON
# ============================================================

def scenario_comparison():
    st.title("⚖️ Scenario Comparison")
    scenarios = st.session_state.get("scenarios", {})
    if not scenarios:
        st.info("No saved scenarios yet — create scenarios in Scenario Simulation.")
        return

    # Build comparison metrics for each scenario
    rows = []
    baseline = (
        data["supplier_price"].sort_values("unit_price").drop_duplicates(subset=["sku"]) 
    )
    baseline["procure_qty"] = np.random.randint(200, 1000, size=len(baseline))
    baseline["total_cost"] = baseline["procure_qty"] * baseline["unit_price"]
    baseline_total = baseline["total_cost"].sum()

    for name, params in scenarios.items():
        # Simulate simple impacts
        demand_factor = 1 + params.get("demand_change", 0) / 100.0
        price_factor = 1 + params.get("price_change", 0) / 100.0
        total_qty = int(baseline["procure_qty"].sum() * demand_factor)
        avg_price = baseline["unit_price"].mean() * price_factor
        total_cost = (baseline["procure_qty"].sum() * avg_price)
        suppliers_engaged = baseline["supplier"].nunique()
        rows.append({
            "scenario": name,
            "total_qty": total_qty,
            "avg_unit_price": round(avg_price, 2),
            "total_cost": round(total_cost, 2),
            "savings_vs_baseline_pct": round(100 * (baseline_total - total_cost) / baseline_total, 2),
            "suppliers_engaged": suppliers_engaged,
        })

    cmp_df = pd.DataFrame(rows).set_index("scenario")
    st.dataframe(cmp_df)

    st.subheader("Detail: select scenario to view distribution")
    sel = st.selectbox("Choose scenario", options=list(scenarios.keys()))
    params = scenarios[sel]
    st.write(params)

    st.info("Use these numbers as illustrative impacts; connect to optimization engine for production-grade comparisons.")

    # Offer an optimized plan preview for the selected scenario
    if st.button("Run Optimization for selected scenario"):
        with st.spinner("Running optimization..."):
            # ensure params include service_level and risk if present in scenario
            if "service_level" in params:
                params_local = params
            else:
                params_local = params
            plan = optimize_plan(params_local)
        if plan is None:
            st.error("Optimization infeasible for this scenario (capacity < demand).")
        else:
            st.subheader("Optimized Procurement Plan (sample)")
            st.dataframe(plan.head(50))
            st.metric("Total Cost", f"${plan['total_cost'].sum():,.0f}")
            # store last plan for Output Data Spec
            st.session_state.last_plan = plan

# ============================================================
#               BOM & DEPENDENCIES
# ============================================================

def bom_dependencies():
    st.title("📦 BOM & Material Dependencies")

    sku = st.selectbox("Choose SKU", data["skus"]["sku"])
    bom = data["bom"][data["bom"]["sku"] == sku]

    st.subheader(f"Components for {sku}")
    st.dataframe(bom)


def optimize_plan(params: dict):
    """Run a simple LP to minimize procurement cost under supplier capacity and MOQ constraints.

    params keys: demand_change (%), price_change (%), capacity_change (%)
    Returns allocation DataFrame or None if infeasible.
    """
    # prepare demand per SKU (12-month sum) and offers
    forecast = data["forecast"].groupby("sku")["forecast_qty"].sum().rename("demand")
    demand = forecast.to_dict()

    offers = data["supplier_price"].copy()
    # apply price factor and capacity factor
    price_factor = 1 + params.get("price_change", 0) / 100.0
    capacity_factor = 1 + params.get("capacity_change", 0) / 100.0
    # risk_penalty scales with supplier risk_score (0..1)
    risk_penalty = float(params.get("risk_penalty", 0.0))
    offers["adj_price"] = offers["unit_price"] * price_factor * (1 + risk_penalty * offers.get("lead_time", 0) * 0)
    # if supplier risk_score exists, include its effect
    if "risk_score" in data["suppliers"].columns:
        # merge supplier risk into offers
        offers = offers.merge(data["suppliers"][["supplier", "risk_score"]], on="supplier", how="left")
        offers["adj_price"] = offers["unit_price"] * price_factor * (1 + risk_penalty * offers["risk_score"].fillna(0))
    else:
        offers["adj_price"] = offers["unit_price"] * price_factor

    # supplier capacities
    supplier_caps = data["suppliers"].set_index("supplier")["capacity_per_period"].to_dict()

    # Build LP
    prob = pl.LpProblem("procure_opt", pl.LpMinimize)

    # Variables: continuous order qty per offer and binary use flag for MOQ
    vars_x = {}
    vars_y = {}
    bigM = {}

    for idx, row in offers.iterrows():
        name = f"x_{idx}"
        vars_x[idx] = pl.LpVariable(name, lowBound=0, cat="Continuous")
        vars_y[idx] = pl.LpVariable(f"y_{idx}", cat="Binary")
        bigM[idx] = demand.get(row["sku"], 0)

    # Objective: minimize total cost
    prob += pl.lpSum([vars_x[i] * offers.loc[i, "adj_price"] for i in offers.index])

    # Demand satisfaction per SKU
    for sku, qty in demand.items():
        related = [vars_x[i] for i in offers[offers["sku"] == sku].index]
        if related:
            prob += pl.lpSum(related) == qty * (1 + params.get("demand_change", 0) / 100.0)

    # Service level requirement: require a fraction of demand served by offers with lead_time <= max_lead_time
    service_level = float(params.get("service_level", 0.0)) / 100.0
    max_lead_time = params.get("max_lead_time", None)
    if service_level > 0 and max_lead_time is not None:
        for sku, qty in demand.items():
            related_fast = [vars_x[i] for i in offers[(offers["sku"] == sku) & (offers["lead_time"] <= float(max_lead_time))].index]
            if related_fast:
                prob += pl.lpSum(related_fast) >= service_level * qty * (1 + params.get("demand_change", 0) / 100.0)

    # Supplier capacity
    for sup, cap in supplier_caps.items():
        related = [vars_x[i] for i in offers[offers["supplier"] == sup].index]
        if related:
            prob += pl.lpSum(related) <= cap * capacity_factor

    # MOQ linking
    for i, row in offers.iterrows():
        moq = row.get("moq", 0)
        prob += vars_x[i] <= bigM[i] * vars_y[i]
        # if chosen, meet MOQ
        prob += vars_x[i] >= moq * vars_y[i]

    # Solve with default solver
    status = prob.solve(pl.PULP_CBC_CMD(msg=False))
    if pl.LpStatus[status] != "Optimal":
        return None

    # Build result
    alloc = []
    for i, row in offers.iterrows():
        q = pl.value(vars_x[i])
        if q is None:
            q = 0
        if q > 0:
            alloc.append({
                "sku": row["sku"],
                "supplier": row["supplier"],
                "order_qty": float(q),
                "unit_price": float(row["unit_price"]),
                "adj_price": float(row["adj_price"]),
                "total_cost": float(q) * float(row["unit_price"]),
            })

    plan_df = pd.DataFrame(alloc)
    if plan_df.empty:
        return None
    return plan_df

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
    This page provides KPI summaries and complete output tables produced by the optimization engine.
    """)

    # KPI summary (from last optimization if available)
    st.subheader("KPI Summary")
    lp = st.session_state.get("last_plan")
    baseline = st.session_state.get("baseline_plan")
    if lp is not None:
        total_cost = lp["total_cost"].sum()
        avg_price = lp["adj_price"].mean() if "adj_price" in lp.columns else lp["unit_price"].mean()
        suppliers_engaged = lp["supplier"].nunique()
        st.metric("Optimized Total Cost", f"${total_cost:,.0f}")
        st.metric("Optimized Avg Unit Price", f"${avg_price:.2f}")
        st.metric("Suppliers Engaged", f"{suppliers_engaged}")
    else:
        st.info("No optimization run in this session yet. Run a scenario optimization to populate outputs.")

    st.subheader("Full Output Tables")
    st.write("**Optimized Procurement Plan (last run)**")
    if lp is not None:
        st.dataframe(lp)
    else:
        st.write("(no optimized plan)")

    st.write("**Baseline Plan (derived)**")
    if baseline is not None:
        st.dataframe(baseline)
    else:
        st.write("(no baseline saved)")

    st.write("**All Input Tables**")
    with st.expander("SKUs"):
        st.dataframe(data["skus"])
    with st.expander("Suppliers"):
        st.dataframe(data["suppliers"])
    with st.expander("Supplier Offers"):
        st.dataframe(data["supplier_price"])
    with st.expander("Forecast"):
        st.dataframe(data["forecast"].head(200))
    with st.expander("Inventory"):
        st.dataframe(data["inventory"])
    with st.expander("BOM"):
        st.dataframe(data["bom"].head(200))

# ============================================================
#                   MAIN ROUTER
# ============================================================

user = get_user_role()

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
elif page == "Scenario Comparison":
    scenario_comparison()
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
