import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import json
import os
import textwrap
from datetime import datetime

# --- Page Config ---
st.set_page_config(
    page_title="NeuralPulse | Industrial AI",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown('<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined" rel="stylesheet" />', unsafe_allow_html=True)

# --- Constants & Config ---
USERS_FILE = 'users.json'
MODEL_FILE = 'best_model.pkl'
SCALER_FILE = 'scaler.pkl'

# --- Theme & Styling ---
# Deep Navy (#050A14) / Charcoal (#111827) / Accent (#3B82F6)
STYLING_CSS_GLOBAL = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@24,400,0,0');
    
    /* GLOBAL THEME */
    .stApp {
        background-color: #0F172A; /* Deep Navy as Use requested */
        background-image: 
            linear-gradient(rgba(255, 255, 255, 0.05) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255, 255, 255, 0.05) 1px, transparent 1px);
        background-size: 50px 50px;
        font-family: 'Inter', sans-serif;
        color: #F8FAFC;
    }
    
    /* SIDEBAR STYLING */
    section[data-testid="stSidebar"] {
        background-color: #0D1B2A;
    }
    section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] span, section[data-testid="stSidebar"] div {
        color: white !important;
    }

    /* WIDGET STYLING (Global) */
    .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] > div {
        background-color: #1E293B !important;
        color: white !important;
        border: 1px solid #334155 !important;
    }

    /* 2. Fallback for Shadow DOM / Form Hints */
    .stTextInput > div > div > div > small {
        display: none !important;
    }
    
    /* 3. Ensure the password toggle icon background blends in */
    [data-testid="stVisibilityToggle"] {
        background-color: transparent !important;
    }
    
    /* 4. Fix Input Padding & Color (Ensure text doesn't hit the icon) */
    .stTextInput input {
        color: #ffffff !important;
        padding-right: 50px !important; /* Make space for the icon */
        caret-color: #ffffff !important;
    }
    
    /* 2. Force +/- Buttons to be Visible & Styled */
    [data-testid="stNumberInputStepDown"], [data-testid="stNumberInputStepUp"] {
        color: #ffffff !important; /* Icon Color */
        background-color: #334155 !important; /* Visible Background */
        border-left: 1px solid #475569 !important;
        opacity: 1 !important; /* Stop hiding them */
    }

    /* 3. Ensure Input Text is White */
    .stNumberInput input {
        color: #ffffff !important;
        background-color: #1e293b !important; /* Dark Blue-Grey Input Background */
    }
    
    /* Fix Selectbox dropdown text */
    ul[data-baseweb="menu"] {
        background-color: #1E293B !important;
    }
    
    /* BUTTONS */
    .stButton button {
        background-color: #3B82F6 !important;
        color: white !important;
        border: none !important;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        background-color: #2563EB !important;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4) !important;
    }
    
    /* HIDE DEFAULT STREAMLIT ELEMENTS */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
</style>
"""

STYLING_CSS_LOGIN = """
<style>
    /* LOGIN CARD - Dark Industrial */
    .login-card {
        background-color: #1E293B;
        padding: 40px;
        border-radius: 16px;
        border: 1px solid #334155;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        text-align: center;
        margin-bottom: 20px;
    }

    /* TYPOGRAPHY */
    h1 {
        color: #F8FAFC !important;
        font-weight: 700 !important;
        font-size: 28px !important;
        margin-top: 10px !important;
        margin-bottom: 5px !important;
    }
    
    .subtitle {
        color: #94A3B8;
        font-size: 12px;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        font-weight: 600;
        margin-bottom: 20px;
    }
    
    label {
        color: #CBD5E1 !important;
        font-size: 12px !important;
        font-weight: 500 !important;
    }

    /* INPUT FIELDS */
    .stTextInput input {
        background-color: #0F172A !important; 
        border: 1px solid #334155 !important;
        color: #E2E8F0 !important;
        border-radius: 8px !important;
        padding: 10px 12px !important;
    }
    
    .stTextInput input:focus {
        border-color: #3B82F6 !important;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2) !important;
    }
    
    /* BUTTON */
    .stButton button {
        background-color: #2563EB !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 12px 0 !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        width: 100% !important;
        margin-top: 20px !important;
        transition: all 0.2s ease;
    }
    
    .stButton button:hover {
        background-color: #1D4ED8 !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3) !important;
    }

    /* FOOTER */
    .footer-text {
        color: #64748B;
        font-size: 11px;
        margin-top: 30px;
    }
</style>
"""

STYLING_CSS_DASHBOARD = """
<style>
    /* GLASS CARDS (DASHBOARD) */
    .glass-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 20px;
    }
</style>
"""

st.markdown(STYLING_CSS_GLOBAL, unsafe_allow_html=True)

# --- Logic: Auth & Data ---

def load_users():
    if not os.path.exists(USERS_FILE):
        return {"users": []}
    with open(USERS_FILE, 'r') as f:
        return json.load(f)

def save_users(users_data):
    with open(USERS_FILE, 'w') as f:
        json.dump(users_data, f, indent=4)

def check_login(username, password):
    data = load_users()
    for user in data['users']:
        if user['username'] == username and user['password'] == password:
            return user
    return None

def register_user(fullname, operator_id, department, password):
    data = load_users()
    for user in data['users']:
        if user['username'] == operator_id:
            return False, "Operator ID already exists."
    
    new_user = {
        "name": fullname,
        "username": operator_id,
        "department": department,
        "password": password,
        "access_key": "generated-key"
    }
    data['users'].append(new_user)
    save_users(data)
    return True, "Account created. Please login."

@st.cache_resource
def load_resources():
    try:
        model = joblib.load(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
        return model, scaler
    except Exception as e:
        return None, None

# --- Application State ---
if 'user' not in st.session_state:
    st.session_state['user'] = None
if 'page' not in st.session_state:
    st.session_state['page'] = 'login' 

def logout():
    st.session_state['user'] = None
    st.rerun()

# --- Main Render Logic ---

def login_page():
    st.markdown(STYLING_CSS_LOGIN, unsafe_allow_html=True)
    # 1. Start of Page Layout
    left_col, main_col, right_col = st.columns([1, 2, 1])
    
    with main_col:
        # 2. Card Start & Header (Sandwich Method)
        st.markdown(textwrap.dedent("""
            <div class="login-card">
                <div style="margin-bottom: 20px;">
                    <span class="material-symbols-outlined" style="font-size: 48px; color: #3B82F6; background: rgba(59, 130, 246, 0.1); padding: 15px; border-radius: 50%;">precision_manufacturing</span>
                </div>
                <h1>NeuralPulse</h1>
                <div class="subtitle">Machine Failure Prediction System</div>
                <br>
        """), unsafe_allow_html=True)
        
        # 3. Inputs (Inside the visual card)
        username = st.text_input("Operator ID", placeholder="Enter your ID")
        password = st.text_input("Access Key", type="password", placeholder="Enter your key")
        
        # 6. Button
        if st.button("INITIALIZE SESSION", use_container_width=True):
            user = check_login(username, password)
            if user:
                st.session_state['user'] = user
                st.toast("Access Granted", icon="🔓")
                time.sleep(0.5)
                st.rerun()
            else:
                st.error("Access Denied: Invalid Credentials")
        
        # 5. Helper Links (Nested Columns)
        st.markdown("<br>", unsafe_allow_html=True)
        col_rem, col_forgot = st.columns(2)
        with col_rem:
            st.checkbox("Remember device")
        with col_forgot:
            st.markdown('<div style="text-align: right;"><a href="#" style="color: #60A5FA; text-decoration: none;">Forget ID?</a></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Create Account", type="tertiary", use_container_width=True):
            st.session_state['page'] = 'register'
            st.rerun()

        # 6. Card End & Footer
        st.markdown(textwrap.dedent("""
                <div class="footer-text">
                    Authorized access only. System status: <span style="color:#10B981;">Operational</span>
                </div>
            </div>
        """), unsafe_allow_html=True)

def register_page():
    st.markdown(STYLING_CSS_LOGIN, unsafe_allow_html=True)
    # 1. Start of Page Layout
    left_col, main_col, right_col = st.columns([1, 2, 1])
    
    with main_col:
        # 2. Card Start & Header (Sandwich Method)
        st.markdown(textwrap.dedent("""
            <div class="login-card">
                <div style="margin-bottom: 20px;">
                    <span class="material-symbols-outlined" style="font-size: 48px; color: #3B82F6; background: rgba(59, 130, 246, 0.1); padding: 15px; border-radius: 50%;">person_add</span>
                </div>
                <h1>New Operator Registration</h1>
                <div class="subtitle">Machine Failure Prediction System</div>
                <br>
        """), unsafe_allow_html=True)
        
        # 3. Inputs
        new_fullname = st.text_input("Full Name", placeholder="e.g. John Doe")
        new_emp_id = st.text_input("Employee ID", placeholder="e.g. OP-4500")
        new_dept = st.selectbox("Department", ["Production", "Maintenance", "Quality Control"])
        new_password = st.text_input("Password", type="password", placeholder="Choose a secure password")
        
        # 4. Action Button
        if st.button("REGISTER", use_container_width=True):
            if new_fullname and new_emp_id and new_password:
                success, msg = register_user(new_fullname, new_emp_id, new_dept, new_password)
                if success:
                    st.success(msg)
                    st.session_state['page'] = 'login'
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(msg)
            else:
                st.warning("Please fill all fields.")
        
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Already have an account? Login here", type="tertiary", use_container_width=True):
            st.session_state['page'] = 'login'
            st.rerun()

        # 5. Card End & Footer
        st.markdown(textwrap.dedent("""
                <div class="footer-text">
                    System requires approval for high-level access.
                </div>
            </div>
        """), unsafe_allow_html=True)

if st.session_state['user'] is None:
    if st.session_state['page'] == 'register':
        register_page()
    else:
        login_page()

else:
    # LOGGED IN DASHBOARD
    st.markdown(STYLING_CSS_DASHBOARD, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown(f"""
        <div style="padding: 1rem 0; text-align: center;">
            <div style="margin-bottom: 0px;"><span class="material-symbols-outlined" style="font-size: 40px; color: #3B82F6;">precision_manufacturing</span></div>
            <h3 style="margin:0; font-size: 1.2rem;">NeuralPulse</h3>
            <p style="font-size: 0.8rem; color: #9CA3AF !important;">Status: <span style="color: #10B981;">Online</span></p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")
        
        page = st.radio("MENU", ["📊 System Dashboard", "🧠 Prediction Engine", "📜 Alert Logs"], label_visibility="collapsed")
        
        st.markdown("---")
        st.markdown(f"<div style='padding: 10px; background: rgba(255,255,255,0.05); border-radius: 8px;'>User: <b>{st.session_state['user']['name']}</b></div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("LOGOUT", use_container_width=True):
            logout()

    # --- MAIN CONTENT AREA ---
    
    # 1. DASHBOARD
    if "Dashboard" in page:
        # Breadcrumbs & Header
        c_head1, c_head2, c_head3 = st.columns([2, 4, 1])
        with c_head1:
            st.markdown("<div style='color: #9CA3AF; font-size: 0.8rem; margin-bottom: 5px;'>Operations  ›  Unit A4 Monitor <span style='background:#10B981; color:black; padding: 2px 6px; border-radius: 4px; font-weight:bold; font-size: 0.6rem; margin-left: 8px;'>ONLINE</span></div>", unsafe_allow_html=True)
            st.title("System Dashboard")
        with c_head3:
            st.markdown("<div style='text-align: right; padding-top: 20px;'><span class='material-symbols-outlined' style='color: #9CA3AF;'>notifications</span> <span class='material-symbols-outlined' style='color: #9CA3AF; margin-left: 10px;'>help</span></div>", unsafe_allow_html=True)
            
        st.markdown("---")
        
        # ROW 1: KPI CARDS (Restored 3-Column Layout)
        kpi1, kpi2, kpi3 = st.columns(3)
        
        # Card 1: Failure Risk (HTML/CSS)
        with kpi1:
            st.markdown("""
            <div class="glass-card" style="height: 280px; display: flex; flex-direction: column; justify-content: space-between;">
                <div>
                    <div style="color: #9CA3AF; font-size: 0.9rem; margin-bottom: 10px;">● Current Failure Risk</div>
                    <div style="font-size: 3.5rem; font-weight: 700; color: #F3F4F6;">12<span style="font-size: 1.5rem; color: #9CA3AF;">%</span></div>
                </div>
                <div>
                    <div style="background: rgba(16, 185, 129, 0.2); color: #10B981; padding: 4px 8px; border-radius: 4px; display: inline-block; font-size: 0.8rem; font-weight: 600;">↘ -5% from avg</div>
                    <div style="height: 4px; background: #374151; border-radius: 2px; margin-top: 10px; width: 100%;">
                        <div style="height: 100%; width: 12%; background: #F59E0B; border-radius: 2px;"></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Card 2: System Health (Plotly Gauge)
        with kpi2:
            import plotly.graph_objects as go
            
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = 88,
                title = {'text': "System Health Status", 'font': {'size': 14, 'color': "#9CA3AF"}},
                number = {'suffix': "/100", 'font': {'size': 24, 'color': "white"}},
                gauge = {
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#374151"},
                    'bar': {'color': "#3B82F6"},
                    'bgcolor': "rgba(0,0,0,0)",
                    'borderwidth': 0,
                    'bordercolor': "rgba(0,0,0,0)",
                    'steps': [
                        {'range': [0, 60], 'color': "rgba(239, 68, 68, 0.2)"},
                        {'range': [60, 85], 'color': "rgba(245, 158, 11, 0.2)"},
                        {'range': [85, 100], 'color': "rgba(16, 185, 129, 0.2)"}
                    ],
                    'threshold': {
                        'line': {'color': "white", 'width': 2},
                        'thickness': 0.75,
                        'value': 88
                    }
                }
            ))
            fig_gauge.update_layout(
                paper_bgcolor='rgba(31, 41, 55, 0.4)',
                font={'color': "white", 'family': "Inter"},
                height=280,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            # Custom container to match glass card
            st.markdown('<div class="glass-card" style="padding: 0;">', unsafe_allow_html=True)
            st.plotly_chart(fig_gauge, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Card 3: Predicted Status (HTML/CSS)
        with kpi3:
            st.markdown("""
            <div class="glass-card" style="height: 280px; display: flex; flex-direction: column; justify-content: space-between;">
                <div>
                    <div style="color: #9CA3AF; font-size: 0.9rem; margin-bottom: 10px;">✨ Predicted Status</div>
                    <div style="font-size: 2.2rem; font-weight: 800; color: #F3F4F6; letter-spacing: 1px;">● NORMAL</div>
                </div>
                <div>
                    <div style="display: flex; justify-content: space-between; color: #9CA3AF; font-size: 0.8rem; margin-bottom: 5px;">
                        <span>Confidence Level</span>
                        <span style="color: white; font-weight: 600;">94%</span>
                    </div>
                    <div style="height: 6px; background: #374151; border-radius: 3px; width: 100%;">
                        <div style="height: 100%; width: 94%; background: #3B82F6; border-radius: 3px; box-shadow: 0 0 10px rgba(59, 130, 246, 0.5);"></div>
                    </div>
                    <div style="margin-top: 15px; font-size: 0.75rem; color: #6B7280;">
                        Next predicted maintenance: <br><span style="color: #9CA3AF;">14 Days</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ROW 2: CHARTS (Full Width)
        st.subheader("Performance Analytics")
        c_chart1, c_chart2 = st.columns(2)
        
        # Vibration Analysis (Spline Area Chart)
        with c_chart1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("Vibration Analysis")
            st.markdown("<div style='color: #9CA3AF; font-size: 0.8rem; margin-bottom: 10px;'>Last 24 Hours • Sensor V-04</div>", unsafe_allow_html=True)
            
            # Mock Data
            x = np.linspace(0, 24, 100)
            y = 30 + 10 * np.sin(x/2) + np.random.normal(0, 2, 100)
            df_vib = pd.DataFrame({'Time': x, 'Vibration': y})
            
            try:
                import plotly.express as px
                fig_vib = px.area(df_vib, x='Time', y='Vibration', template='plotly_dark')
                fig_vib.update_traces(line_color='#3B82F6', fillcolor='rgba(59, 130, 246, 0.1)')
                fig_vib.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    height=220,
                    margin=dict(l=0, r=0, t=0, b=0),
                    xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
                )
                st.plotly_chart(fig_vib, use_container_width=True)
            except Exception as e:
                st.error(f"Error loading charts: {e}")
            st.markdown('</div>', unsafe_allow_html=True)

        # Temperature Loads (Bar Chart)
        with c_chart2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("Temperature Loads")
            st.markdown("<div style='color: #9CA3AF; font-size: 0.8rem; margin-bottom: 10px;'>Current Distribution • Avg 55°C</div>", unsafe_allow_html=True)
            
            df_temp = pd.DataFrame({
                'Unit': ['Unit 1', 'Unit 2', 'Unit 3', 'Unit 4', 'Unit 5'],
                'Temp': [65, 45, 55, 80, 60]
            })
            
            try:
                fig_temp = px.bar(df_temp, x='Unit', y='Temp', template='plotly_dark')
                fig_temp.update_traces(marker_color='#374151', marker_line_width=0)
                # Highlight max
                colors = ['#374151'] * 5
                colors[3] = '#3B82F6' # Unit 4
                fig_temp.update_traces(marker_color=colors)
                
                fig_temp.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    height=220,
                    margin=dict(l=0, r=0, t=0, b=0),
                    xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=False)
                )
                st.plotly_chart(fig_temp, use_container_width=True)
            except:
                st.error("Chart Error")
            st.markdown('</div>', unsafe_allow_html=True)

        # ROW 3: MOCK TABLE
        st.subheader("Active Machines Status")
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        
        machines = pd.DataFrame({
            "ID": ["M-001", "M-002", "M-003", "M-004", "M-005"],
            "Type": ["Hydraulic Press A", "CNC Lathe", "Milling Unit", "Conveyor Belt", "Assembly Arm"],
            "Temperature": [45, 62, 55, 40, 58],
            "Vibration Load": [86, 24, 65, 12, 92],
            "Last Maint.": ["2 days ago", "1 week ago", "3 days ago", "Yesterday", "4 hours ago"],
            "Status": ["Optimal", "Warning", "Optimal", "Optimal", "Critical"]
        })
        
        st.dataframe(
            machines,
            column_config={
                "Temperature": st.column_config.NumberColumn(
                    "Temperature",
                    format="%d°C",
                ),
                "Vibration Load": st.column_config.ProgressColumn(
                    "Vibration Load",
                    format="%d%%",
                    min_value=0,
                    max_value=100,
                ),
                "Status": st.column_config.TextColumn(
                    "Status",
                ),
            },
            hide_index=True,
            use_container_width=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # 2. PREDICTION ENGINE
    elif "Prediction" in page:
        st.title("Neural Prediction Engine")
        st.markdown("---")
        
        model, scaler = load_resources()
        
        if model:
            c1, c2 = st.columns([1, 1.5])
            
            with c1:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.subheader("Sensor Inputs")
                
                # 5 Specific Inputs Request
                temp_input = st.number_input("Temperature [K]", 290.0, 310.0, 300.0)
                rpm_input = st.number_input("RPM (Rotational Speed)", 1000, 3000, 1500)
                tool_wear_input = st.number_input("Tool Wear [min]", 0, 300, 0)
                pressure_input = st.number_input("Pressure [Bar] (Mapped to Torque)", 0.0, 100.0, 40.0)
                vib_input = st.number_input("Vibration [Hz]", 0.0, 100.0, 50.0)
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            with c2:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.subheader("Analysis Results")
                
                if st.button("ANALYZE SENSOR DATA", use_container_width=True):
                    # Mapping Logic
                    # 1. Temp -> Air_temperature_K
                    # 2. Process Temp -> Air Temp + 10 (Derived)
                    # 3. RPM -> Rotational_speed_rpm
                    # 4. Tool Wear -> Tool_wear_min
                    # 5. Pressure -> Torque_Nm (Proxy)
                    # 6. Type -> 'M' (Medium, Default encoded: H=0, L=0, M=1)
                    # Vibration -> Ignored by model
                    
                    input_df = pd.DataFrame([{
                        'UDI': 0,
                        'Air_temperature_K': temp_input,
                        'Process_temperature_K': temp_input + 10,
                        'Rotational_speed_rpm': rpm_input,
                        'Torque_Nm': pressure_input, # Mapping Pressure to Torque
                        'Tool_wear_min': tool_wear_input,
                        'Type_H': 0,
                        'Type_L': 0,
                        'Type_M': 1 # Defaulting to Medium
                    }])
                    
                    try:
                        scaled = scaler.transform(input_df)
                        prob = model.predict_proba(scaled)[0][1]
                        
                        # Display
                        st.markdown("---")
                        
                        # Custom Gauge / Progress
                        st.write(f"**Failure Probability: {prob:.1%}**")
                        st.progress(int(prob * 100))
                        
                        if prob > 0.5:
                            st.markdown("""
                            <div style="background: rgba(239, 68, 68, 0.2); border-left: 5px solid #EF4444; padding: 20px; border-radius: 8px;">
                                <h2 style="color: #EF4444 !important; margin:0;">CRITICAL RISK DETECTED</h2>
                                <p>Immediate maintenance recommended.</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown("""
                            <div style="background: rgba(16, 185, 129, 0.2); border-left: 5px solid #10B981; padding: 20px; border-radius: 8px;">
                                <h2 style="color: #10B981 !important; margin:0;">SYSTEM OPERATIONAL</h2>
                                <p>No immediate anomalies detected.</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                    except Exception as e:
                        st.error(f"Prediction Error: {e}")
                

                st.markdown("</div>", unsafe_allow_html=True)

    # 3. ALERT LOGS
    elif "Logs" in page:
        st.title("Alert History Log")
        st.markdown("<div style='color: #9CA3AF; margin-bottom: 20px;'>Monitor predicted failures and review machine health anomalies.</div>", unsafe_allow_html=True)
        
        # Controls
        c_filter, c_export = st.columns([6, 2])
        with c_filter:
            st.text_input("Search Machine ID, Error Code...", placeholder="Search logs...", label_visibility="collapsed")
        with c_export:
            # Placeholder to be swapped with actual download button below logic
            pass

        st.markdown("---")
        
        # Generate Mock Data
        # timestamp_range = pd.date_range(end=datetime.now(), periods=20, freq='H')
        data = []
        for i in range(20):
            is_fail = np.random.choice([0, 1], p=[0.8, 0.2])
            row = {
                "Timestamp": (datetime.now() - pd.Timedelta(hours=i)).strftime("%Y-%m-%d %H:%M:%S"),
                "Machine ID": f"MACH-{np.random.choice(['X99', 'A02', 'B15', 'C33', 'D41', 'Z12'])}",
                "Temperature": np.random.randint(40, 95),
                "Vibration": np.round(np.random.uniform(0.5, 5.0), 2),
                "Predicted Failure": is_fail,
                "Probability": np.random.randint(85, 99) if is_fail else np.random.randint(5, 40)
            }
            data.append(row)
            
        df_logs = pd.DataFrame(data)
        
        # Styling Function
        def highlight_row(row):
            if row['Predicted Failure'] == 1:
                return ['background-color: #3b1c1c; color: #ffcccc'] * len(row)
            else:
                return ['background-color: rgba(16, 185, 129, 0.05); color: #d1fae5'] * len(row)

        # Apply Styling
        styled_df = df_logs.style.apply(highlight_row, axis=1)
        
        # Export Button (Placed here to have data ready)
        with c_export:
            st.download_button(
                label="📥 Export Report (CSV)",
                data=df_logs.to_csv(index=False).encode('utf-8'),
                file_name=f"alert_history_{datetime.now().strftime('%Y%m%d')}.csv",
                mime='text/csv',
                use_container_width=True
            )

        # Render Table
        st.dataframe(
            styled_df,
            column_config={
                "Predicted Failure": st.column_config.CheckboxColumn(
                    "Failure Predicted",
                ),
                "Probability": st.column_config.ProgressColumn(
                    "Confidence Level",
                    format="%d%%",
                    min_value=0,
                    max_value=100,
                ),
            },
            use_container_width=True,
            height=600
        )
