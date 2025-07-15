import streamlit as st
import pandas as pd

st.set_page_config(layout="wide")

st.title("Experiment Results Dashboard")

# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("compiled_results.csv")
        return df
    except FileNotFoundError:
        st.error("The file `utils_analysis/compiled_results.csv` was not found. Please make sure the file exists.")
        return pd.DataFrame()


df = load_data()

if not df.empty:
    # Sidebar for filters
    st.sidebar.header("Filters")

    # Get unique values for filters
    # The first column is unnamed index, so we drop it.
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)

    categorical_columns = ['stopping_criteria', 'answer_how', 'sampling_strategy', 'aggregation_strategy', 'model']
    
    filters = {}
    for col in categorical_columns:
        if col in df.columns:
            unique_vals = sorted(df[col].unique())
            filters[col] = st.sidebar.multiselect(f"{col.replace('_', ' ').title()}", unique_vals, default=unique_vals)

    if 'k' in df.columns:
        k_values = sorted(df['k'].unique())
        selected_k = st.sidebar.multiselect("K", k_values, default=k_values)
    else:
        selected_k = []

    # Filter dataframe
    filtered_df = df.copy()
    for col, selected_values in filters.items():
        filtered_df = filtered_df[filtered_df[col].isin(selected_values)]
    
    if 'k' in df.columns and selected_k:
        filtered_df = filtered_df[filtered_df['k'].isin(selected_k)]


    st.header("Filtered Data")
    st.dataframe(filtered_df)

    st.header("Accuracy Plots")

    # Plotting
    if not filtered_df.empty:
        plot_x_axis = st.selectbox("Select X-axis for plotting", [col for col in categorical_columns + ['k'] if col in df.columns])

        st.subheader(f"Accuracy vs {plot_x_axis}")
        
        # Check if the selected x-axis has more than one unique value
        if len(filtered_df[plot_x_axis].unique()) > 1:
            # Group by the selected x-axis and calculate mean accuracy
            accuracy_by_xaxis = filtered_df.groupby(plot_x_axis)['accuracy'].mean().reset_index()
            accuracy_by_xaxis = accuracy_by_xaxis.sort_values(by=plot_x_axis)

            # Bar chart
            st.bar_chart(accuracy_by_xaxis.set_index(plot_x_axis))
        else:
            st.warning(f"Not enough data to plot. Select more than one value for '{plot_x_axis}' in the filters or the current selection only has one value.")
    else:
        st.warning("No data matches the selected filters.") 