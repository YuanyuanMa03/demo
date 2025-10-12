import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import solve_ivp
# Import models
from carbon_models import D1Model, D2Model, D3Model, D4Model, L1aModel, L2bModel, C1Model, RothCModel, DSSATCenturyModel

# Set Chinese fonts optimized for Streamlit Cloud
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Bitstream Vera Sans', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# Set page configuration
st.set_page_config(
    page_title="Carbon Dynamics Model Visualization Demo",
    page_icon="🌱",
    layout="wide"
)

# Page title
st.title("🌱 Carbon Dynamics Model Visualization Demo")
st.markdown("---")

# Sidebar parameter settings
st.sidebar.header("Model Parameter Settings")

# Select model
model_options = {
    "D1 Model (Single Pool Model)": "D1",
    "D2 Model (Serial Two-Pool Model)": "D2", 
    "D3 Model (Parallel Two-Pool Model)": "D3",
    "D4 Model (Feedback Model)": "D4",
    "L1a Model (Time-Varying Parameter Model)": "L1a",
    "L2b Model (Environmental Factor Correction Model)": "L2b",
    "C1 Model (Gamma Distribution Model)": "C1",
    "RothC Model (Multi-Pool Model)": "RothC",
    "DSSAT-Century Model": "DSSAT"
}

selected_model_name = st.sidebar.selectbox(
    "Select Carbon Decomposition Model",
    list(model_options.keys())
)

selected_model_key = model_options[selected_model_name]

# General parameters
st.sidebar.subheader("General Parameters")
initial_carbon = st.sidebar.slider("Initial carbon content (g/m²)", 100.0, 5000.0, 1000.0, 50.0)
simulation_time = st.sidebar.slider("Simulation time (years)", 1, 100, 50)

# Simulation control
st.sidebar.subheader("Simulation Control")
time_steps = st.sidebar.slider("Simulation time steps", 10, 1000, 100, 10)
time_span = st.sidebar.slider("Simulation time span (years)", 1, 100, 10, 1)

# Model-specific parameters
model = None
extra_params = {}

if selected_model_key == "D1":
    st.sidebar.subheader("D1 Model Parameters")
    k = st.sidebar.slider("Decay rate constant k (y⁻¹)", 0.001, 0.5, 0.0231, 0.001)
    model = D1Model(k=k)
    
elif selected_model_key == "D2":
    st.sidebar.subheader("D2 Model Parameters")
    r = st.sidebar.slider("Proportion parameter r", 0.0, 1.0, 0.870, 0.01)
    k1 = st.sidebar.slider("Pool 1 decay rate k1 (y⁻¹)", 0.001, 1.0, 0.221, 0.001)
    k2 = st.sidebar.slider("Pool 2 decay rate k2 (y⁻¹)", 0.001, 0.1, 0.0125, 0.001)
    carbon_input = st.sidebar.slider("Carbon input rate A (g/m²/y)", 0.0, 100.0, 50.0, 1.0)
    model = D2Model(r=r, k1=k1, k2=k2)
    extra_params['A'] = carbon_input
    
elif selected_model_key == "D3":
    st.sidebar.subheader("D3 Model Parameters")
    alpha = st.sidebar.slider("Allocation proportion α", 0.0, 1.0, 0.863, 0.01)
    k1 = st.sidebar.slider("Pool 1 decay rate k1 (y⁻¹)", 0.001, 1.0, 0.221, 0.001)
    k2 = st.sidebar.slider("Pool 2 decay rate k2 (y⁻¹)", 0.001, 0.1, 0.0125, 0.001)
    carbon_input = st.sidebar.slider("Carbon input rate A (g/m²/y)", 0.0, 100.0, 50.0, 1.0)
    model = D3Model(alpha=alpha, k1=k1, k2=k2)
    extra_params['A'] = carbon_input
    
elif selected_model_key == "D4":
    st.sidebar.subheader("D4 Model Parameters")
    r = st.sidebar.slider("Proportion parameter r", 0.0, 1.0, 0.879, 0.01)
    k1 = st.sidebar.slider("Pool 1 decay rate k1 (y⁻¹)", 0.001, 1.0, 0.220, 0.001)
    k2 = st.sidebar.slider("Pool 2 decay rate k2 (y⁻¹)", 0.001, 0.1, 0.0143, 0.001)
    carbon_input = st.sidebar.slider("Carbon input rate A (g/m²/y)", 0.0, 100.0, 50.0, 1.0)
    model = D4Model(r=r, k1=k1, k2=k2)
    extra_params['A'] = carbon_input
    
elif selected_model_key == "L1a":
    st.sidebar.subheader("L1a Model Parameters")
    a = st.sidebar.slider("Parameter a", 0.0, 1.0, 0.236, 0.01)
    b = st.sidebar.slider("Parameter b (y⁻¹)", 0.0, 1.0, 0.0940, 0.01)
    m = st.sidebar.slider("Parameter m", 0.0, 5.0, 1.0, 0.1)
    model = L1aModel(a=a, b=b, m=m)
    
elif selected_model_key == "L2b":
    st.sidebar.subheader("L2b Model Parameters")
    k_base = st.sidebar.slider("Base decay rate k_base (y⁻¹)", 0.001, 0.5, 0.02, 0.001)
    t_opt = st.sidebar.slider("Optimal temperature T_opt (°C)", 0, 40, 25, 1)
    w_opt = st.sidebar.slider("Optimal moisture content w_opt", 0.0, 1.0, 0.6, 0.01)
    temperature = st.sidebar.slider("Current temperature T (°C)", 0, 40, 25, 1)
    moisture = st.sidebar.slider("Current moisture content w", 0.0, 1.0, 0.6, 0.01)
    model = L2bModel(k_base=k_base, t_opt=t_opt, w_opt=w_opt)
    extra_params['T'] = temperature
    extra_params['w'] = moisture
    
elif selected_model_key == "C1":
    st.sidebar.subheader("C1 Model Parameters")
    alpha = st.sidebar.slider("Shape parameter α", 0.1, 10.0, 2.0, 0.1)
    beta = st.sidebar.slider("Scale parameter β", 0.1, 5.0, 0.5, 0.1)
    model = C1Model(alpha=alpha, beta=beta)
    
elif selected_model_key == "RothC":
    st.sidebar.subheader("RothC Model Parameters")
    st.sidebar.info("RothC model uses preset parameters, only adjusting initial conditions here")
    dpm_init = st.sidebar.slider("Decomposable plant material initial amount (g/m²)", 0.0, 1000.0, 200.0, 10.0)
    rpm_init = st.sidebar.slider("Resistant plant material initial amount (g/m²)", 0.0, 1000.0, 300.0, 10.0)
    bio_init = st.sidebar.slider("Microbial biomass initial amount (g/m²)", 0.0, 500.0, 100.0, 5.0)
    hum_init = st.sidebar.slider("Humus initial amount (g/m²)", 0.0, 2000.0, 1000.0, 10.0)
    iom_init = st.sidebar.slider("Inert organic matter initial amount (g/m²)", 0.0, 1000.0, 200.0, 10.0)
    model = RothCModel()
    extra_params['initial_state'] = [dpm_init, rpm_init, bio_init, hum_init, iom_init]
    
elif selected_model_key == "DSSAT":
    st.sidebar.subheader("DSSAT-Century Model Parameters")
    ch_init = st.sidebar.slider("Carbohydrate initial amount (g/m²)", 0.0, 500.0, 100.0, 5.0)
    cl_init = st.sidebar.slider("Cellulose initial amount (g/m²)", 0.0, 500.0, 150.0, 5.0)
    ln_init = st.sidebar.slider("Lignin initial amount (g/m²)", 0.0, 500.0, 200.0, 5.0)
    som_init = st.sidebar.slider("Soil organic carbon initial amount (g/m²)", 0.0, 2000.0, 1000.0, 10.0)
    temperature = st.sidebar.slider("Temperature T (°C)", 0, 40, 25, 1)
    moisture = st.sidebar.slider("Moisture content w", 0.0, 1.0, 0.6, 0.01)
    field_capacity = st.sidebar.slider("Field capacity wfc", 0.1, 1.0, 0.7, 0.01)
    cn_ratio = st.sidebar.slider("Carbon-nitrogen ratio C:N", 10, 200, 50, 1)
    model = DSSATCenturyModel()
    extra_params['initial_state'] = [ch_init, cl_init, ln_init]
    extra_params['som_init'] = som_init
    extra_params['T'] = temperature
    extra_params['w'] = moisture
    extra_params['wfc'] = field_capacity
    extra_params['cn_ratio'] = cn_ratio

# Run simulation button
if st.sidebar.button("Run Simulation", type="primary"):
    # Create time axis
    t_span = (0, simulation_time)
    t_eval = np.linspace(0, simulation_time, 200)
    
    try:
        # Run simulation based on model type
        if selected_model_key == "D1":
            solution = solve_ivp(model.dynamics, t_span, [initial_carbon], t_eval=t_eval)
            carbon_data = solution.y[0]
            df = pd.DataFrame({
                'Time (years)': solution.t,
                'Carbon content (g/m²)': carbon_data
            })
            
        elif selected_model_key in ["D2", "D3", "D4"]:
            initial_state = [initial_carbon * 0.7, initial_carbon * 0.3]  # Allocate initial carbon to two pools
            solution = solve_ivp(
                lambda t, y: model.dynamics(t, y, extra_params['A']), 
                t_span, 
                initial_state, 
                t_eval=t_eval
            )
            total_carbon = solution.y[0] + solution.y[1]
            df = pd.DataFrame({
                'Time (years)': solution.t,
                'Total carbon content (g/m²)': total_carbon,
                'Pool 1 carbon content (g/m²)': solution.y[0],
                'Pool 2 carbon content (g/m²)': solution.y[1]
            })
            
        elif selected_model_key == "L1a":
            solution = solve_ivp(model.dynamics, t_span, [initial_carbon], t_eval=t_eval)
            carbon_data = solution.y[0]
            df = pd.DataFrame({
                'Time (years)': solution.t,
                'Carbon content (g/m²)': carbon_data
            })
            
        elif selected_model_key == "L2b":
            solution = solve_ivp(
                lambda t, y: model.dynamics(t, y, extra_params['T'], extra_params['w']), 
                t_span, 
                [initial_carbon], 
                t_eval=t_eval
            )
            carbon_data = solution.y[0]
            df = pd.DataFrame({
                'Time (years)': solution.t,
                'Carbon content (g/m²)': carbon_data
            })
            
        elif selected_model_key == "C1":
            solution = solve_ivp(model.dynamics, t_span, [initial_carbon], t_eval=t_eval)
            carbon_data = solution.y[0]
            df = pd.DataFrame({
                'Time (years)': solution.t,
                'Carbon content (g/m²)': carbon_data
            })
            
        elif selected_model_key == "RothC":
            solution = solve_ivp(
                model.dynamics, 
                t_span, 
                extra_params['initial_state'], 
                t_eval=t_eval
            )
            total_carbon = np.sum(solution.y, axis=0)
            df = pd.DataFrame({
                'Time (years)': solution.t,
                'Total carbon content (g/m²)': total_carbon,
                'Decomposable Plant Material (g/m²)': solution.y[0],
                'Resistant Plant Material (g/m²)': solution.y[1],
                'Microbial Biomass (g/m²)': solution.y[2],
                'Humus (g/m²)': solution.y[3],
                'Inert Organic Matter (g/m²)': solution.y[4]
            })
            
        elif selected_model_key == "DSSAT":
            # Plant residue decomposition
            residue_solution = solve_ivp(
                lambda t, y: model.residue_dynamics(
                    t, y, extra_params['T'], extra_params['w'], 
                    extra_params['wfc'], extra_params['cn_ratio']
                ), 
                t_span, 
                extra_params['initial_state'], 
                t_eval=t_eval
            )
            
            # Soil organic carbon decomposition
            som_solution = solve_ivp(
                lambda t, y: model.som_dynamics(t, y, extra_params['T'], extra_params['w'], extra_params['wfc']), 
                t_span, 
                [extra_params['som_init']], 
                t_eval=t_eval
            )
            
            total_residue = np.sum(residue_solution.y, axis=0)
            total_carbon = total_residue + som_solution.y[0]
            
            df = pd.DataFrame({
                'Time (years)': residue_solution.t,
                'Total carbon content (g/m²)': total_carbon,
                'Total plant residue (g/m²)': total_residue,
                'Carbohydrates (g/m²)': residue_solution.y[0],
                'Cellulose (g/m²)': residue_solution.y[1],
                'Lignin (g/m²)': residue_solution.y[2],
                'Soil organic carbon (g/m²)': som_solution.y[0]
            })
        
        # Display results
        st.subheader(f"{selected_model_name} Simulation Results")
        
        # Key indicators
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Initial carbon content", f"{initial_carbon:.1f} g/m²")
        with col2:
            final_carbon = df.iloc[-1, 1]  # Second column is total carbon content
            st.metric("Final carbon content", f"{final_carbon:.1f} g/m²")
        with col3:
            loss_rate = (initial_carbon - final_carbon) / initial_carbon * 100
            st.metric("Carbon loss rate", f"{loss_rate:.1f}%")
        
        # Chart display
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot total carbon content curve
        if selected_model_key in ["D1", "L1a", "C1", "L2b"]:
            ax.plot(df['Time (years)'], df['Carbon content (g/m²)'], 'b-', linewidth=2, label='Total carbon content')
        else:
            ax.plot(df['Time (years)'], df.iloc[:, 1], 'b-', linewidth=2, label='Total carbon content')
            
            # If there is pool data, also plot pool curves
            if df.shape[1] > 2:
                for i in range(2, min(6, df.shape[1])):  # Plot at most 5 pool curves
                    ax.plot(df['Time (years)'], df.iloc[:, i], '--', linewidth=1.5, alpha=0.7, label=df.columns[i])
        
        ax.set_xlabel('Time (years)')
        ax.set_ylabel('Carbon content (g/m²)')
        ax.set_title(f'{selected_model_name} Carbon dynamics')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        
        # Display data table
        with st.expander("View detailed data"):
            st.dataframe(df.style.format("{:.2f}"), height=300)
            
        # Model equations display
        st.subheader("Model Equations")
        
        # Using markdown and latex combined to display equations for better compatibility
        equation_markdown = {
            "D1": r"**D1 Model Equation:**  $\frac{dx}{dt} = -k \cdot x$",
            "D2": r"**D2 Model Equation:**  $\frac{dx_1}{dt} = A - k_1 \cdot x_1$; $\frac{dx_2}{dt} = (1-r) \cdot k_1 \cdot x_1 - k_2 \cdot x_2$",
            "D3": r"**D3 Model Equation:**  $\frac{dx_1}{dt} = \alpha \cdot A - k_1 \cdot x_1$; $\frac{dx_2}{dt} = (1-\alpha) \cdot A - k_2 \cdot x_2$",
            "D4": r"**D4 Model Equation:**  $\frac{dx_1}{dt} = A - k_1 \cdot x_1$; $\frac{dx_2}{dt} = (1-r) \cdot k_1 \cdot x_1 - k_2 \cdot x_2$",
            "L1a": r"**L1a Model Equation:**  $\frac{dx}{dt} = -(a + b \cdot e^{-m \cdot t}) \cdot x$",
            "L2b": r"**L2b Model Equation:**  $\frac{dx}{dt} = -k_{base} \cdot f(T) \cdot f(W) \cdot x$",
            "C1": r"**C1 Model Equation:**  $\frac{dx}{dt} = -\bar{k} \cdot x$ (where $\bar{k}$ is the mean of the Gamma distribution)",
            "RothC": r"**RothC Model Equation:**  $\frac{dDPM}{dt} = -k_{dpm} \cdot DPM$; $\frac{dRPM}{dt} = -k_{rpm} \cdot RPM$; $\frac{dBIO}{dt} = -k_{bio} \cdot BIO$; $\frac{dHUM}{dt} = -k_{hum} \cdot HUM$; $\frac{dIOM}{dt} = 0$",
            "DSSAT": r"**DSSAT-Century Model Equation:**  $\frac{dCH}{dt} = -k_{ch} \cdot CH \cdot f(T) \cdot f(W) \cdot f(C:N)$; $\frac{dSOM}{dt} = -k_{som} \cdot SOM \cdot f(T) \cdot f(W)$"
        }
        
        # Display equations
        st.markdown(equation_markdown[selected_model_key])
        
        # Model descriptions
        descriptions = {
            "D1": "The D1 model is the simplest single-pool model, assuming all organic carbon decomposes at the same rate.",
            "D2": "The D2 model is a series double-pool model, where carbon flows from one pool to another, simulating the transfer of carbon between different stability pools.",
            "D3": "The D3 model is a parallel double-pool model, where carbon is simultaneously allocated to two independent pools, each decomposing at different rates.",
            "D4": "The D4 model is a feedback model, similar to D2 but with different parameter settings.",
            "L1a": "The L1a model has a time-varying decomposition rate, with parameters a, b, and m controlling the time dependence.",
            "L2b": "The L2b model considers the effects of temperature and moisture on decomposition rate, corrected by environmental factors.",
            "C1": "The C1 model uses a Gamma distribution to describe the heterogeneity of decomposition rates, which better fits the complexity of carbon decomposition in real soils.",
            "RothC": "The RothC model is a classic multi-pool model that divides soil organic carbon into five different pools, each with different decomposition characteristics.",
            "DSSAT": "The DSSAT-Century model distinguishes between different components of plant residues (carbohydrates, cellulose, lignin) and soil organic carbon, simulating their decomposition processes separately."
        }
        
        st.info(descriptions[selected_model_key])
        
    except Exception as e:
        st.error(f"Error occurred during simulation: {str(e)}")
        st.info("Please check if the parameter settings are reasonable, or try adjusting the simulation time.")

else:
    # Default display instructions
    st.info("Please set model parameters in the sidebar, then click the 'Run Simulation' button to start the demonstration.")
    
    # Display brief introductions of all models
    st.subheader("Model Introduction")
    
    model_info = {
        "D1 Model (Single Pool Model)": "The simplest carbon decomposition model, assuming all organic carbon decomposes at the same rate.",
        "D2 Model (Series Double Pool Model)": "Carbon flows from one pool to another, simulating the transfer of carbon between different stability pools.",
        "D3 Model (Parallel Double Pool Model)": "Carbon is simultaneously allocated to two independent pools, each decomposing at different rates.",
        "D4 Model (Feedback Model)": "Similar to the D2 model but with different parameter settings.",
        "L1a Model (Time-Varying Parameter Model)": "A model with decomposition rate varying over time.",
        "L2b Model (Environmental Factor Correction Model)": "A model considering the effects of temperature and moisture on decomposition rate.",
        "C1 Model (Gamma Distribution Model)": "A model using probability distribution to describe the heterogeneity of decomposition rates.",
        "RothC Model (Multi-Pool Model)": "A classic five-pool soil organic carbon model.",
        "DSSAT-Century Model": "A detailed model distinguishing between plant residue components and soil organic carbon."
    }
    
    for model_name, description in model_info.items():
        with st.expander(model_name):
            st.write(description)

# Footer
st.markdown("---")
st.caption("Carbon Dynamics Model Visualization Tool | Different Structure Decomposition Models Based on First-Order Kinetics Equations")