import pandas as pd
import plotly.express as px
import streamlit as st

# Function to preprocess data
def preprocess_data(df, x_col, y_col):
    if x_col and x_col != "None":
        if 'Date' in x_col or pd.api.types.is_datetime64_any_dtype(df[x_col]):
            df[x_col] = pd.to_datetime(df[x_col], errors='coerce')
        else:
            df[x_col] = df[x_col].astype(str).replace('[^-?\d.]', '', regex=True)
            df[x_col] = pd.to_numeric(df[x_col], errors='coerce')

    df[y_col] = df[y_col].astype(str).replace('[^-?\d.]', '', regex=True)
    df[y_col] = pd.to_numeric(df[y_col], errors='coerce')

    return df

# Function to calculate CPK
def calculate_cpk(mean, std, upper_limit, lower_limit):
    if upper_limit is not None and lower_limit is not None and std > 0:
        return min((upper_limit - mean) / (3 * std), (mean - lower_limit) / (3 * std))
    return None

# Function to create scatter plot
def create_scatter_plot(df, x_col, y_col, title, x_min, x_max, y_min, y_max, x_upper_limit, x_lower_limit, y_upper_limit, y_lower_limit, color_col):
    if color_col == "None":
        fig = px.scatter(df, x=x_col if x_col != "None" else None, y=y_col, title=title, color_discrete_sequence=['blue'])
    else:
        fig = px.scatter(df, x=x_col if x_col != "None" else None, y=y_col, color=df[color_col], title=title, color_discrete_sequence=px.colors.qualitative.Set1)

    fig.update_traces(marker=dict(size=8))
    fig.update_layout(
        title={'text': title, 'x': 0.5, 'xanchor': 'center'},
        xaxis=dict(range=[x_min, x_max] if x_min is not None and x_max is not None else None),
        yaxis=dict(range=[y_min, y_max] if y_min is not None and y_max is not None else None),
        legend_title=color_col if color_col != "None" else None
    )

    # Add limit lines
    if y_upper_limit is not None:
        fig.add_hline(y=y_upper_limit, line=dict(color="red", dash="dash"), annotation_text=f'Upper Limit = {y_upper_limit}')
    if y_lower_limit is not None:
        fig.add_hline(y=y_lower_limit, line=dict(color="black", dash="dash"), annotation_text=f'Lower Limit = {y_lower_limit}')
    
    if x_col != "None" and not pd.api.types.is_datetime64_any_dtype(df[x_col]):
        if x_upper_limit is not None:
            fig.add_vline(x=x_upper_limit, line=dict(color="red", dash="dash"), annotation_text=f'X Upper Limit = {x_upper_limit}')
        if x_lower_limit is not None:
            fig.add_vline(x=x_lower_limit, line=dict(color="black", dash="dash"), annotation_text=f'X Lower Limit = {x_lower_limit}')

    return fig

# Streamlit app
def main():
    st.title('📊 Scatter Plot Visualization Tool')

    uploaded_file = st.file_uploader("📂 Upload a File", type=["xlsx", "xls", "csv"])
    if uploaded_file:
        try:
            xlsx = pd.ExcelFile(uploaded_file)
            sheet = st.selectbox("📄 Select Sheet", xlsx.sheet_names)
            df = pd.read_excel(uploaded_file, sheet_name=sheet)

            columns = df.columns.tolist()
            col1, col2 = st.columns(2)
            with col1:
                x_col = st.selectbox("📍 Select X-Axis", ["None"] + columns)
            with col2:
                y_col = st.selectbox("📍 Select Y-Axis", columns)

            df = preprocess_data(df, x_col if x_col != "None" else None, y_col)

            # Filter options
            filter_col = st.selectbox("🎨 Select Filter Column (for color grouping)", ["None"] + columns)
            selected_values = []
            if filter_col != "None":
                selected_values = st.multiselect("🎯 Select Filter Value(s)", df[filter_col].dropna().unique())
                df = df[df[filter_col].isin(selected_values)]

            # X/Y axis limits
            col1, col2 = st.columns(2)
            with col1:
                if x_col != "None" and pd.api.types.is_datetime64_any_dtype(df[x_col]):
                    x_min = st.date_input("📅 X-Axis Min", value=df[x_col].min().date() if pd.notna(df[x_col].min()) else None)
                    x_max = st.date_input("📅 X-Axis Max", value=df[x_col].max().date() if pd.notna(df[x_col].max()) else None)
                    x_upper_limit, x_lower_limit = None, None
                else:
                    x_min = st.number_input("📈 X-Axis Min", value=float(df[x_col].min()) if x_col != "None" and pd.notna(df[x_col].min()) else None)
                    x_max = st.number_input("📉 X-Axis Max", value=float(df[x_col].max()) if x_col != "None" and pd.notna(df[x_col].max()) else None)
                    x_upper_limit = st.number_input("🚀 X Upper Limit", value=None)
                    x_lower_limit = st.number_input("📏 X Lower Limit", value=None)
            with col2:
                y_min = st.number_input("📈 Y-Axis Min", value=float(df[y_col].min()) if not df[y_col].isnull().all() else None)
                y_max = st.number_input("📉 Y-Axis Max", value=float(df[y_col].max()) if not df[y_col].isnull().all() else None)
                y_upper_limit = st.number_input("🚀 Y Upper Limit", value=None)
                y_lower_limit = st.number_input("📏 Y Lower Limit", value=None)

            # Generate plot
            title = f"{x_col if x_col != 'None' else 'Index'} VS {y_col}"
            fig = create_scatter_plot(df, x_col, y_col, title, x_min, x_max, y_min, y_max, x_upper_limit, x_lower_limit, y_upper_limit, y_lower_limit, filter_col)
            st.plotly_chart(fig)
            
            # Compute statistics
            sample_size = df[y_col].dropna().count()
            mean_value = df[y_col].mean()
            std_value = df[y_col].std()
            cpk = calculate_cpk(mean_value, std_value, y_upper_limit, y_lower_limit)

            # Display statistics
            st.subheader("📊 Data Statistics")
            stats_data = {
                "Metric": ["Sample Size", "Mean", "Std Dev", "CPK"],
                "Value": [sample_size, f"{mean_value:.2f}", f"{std_value:.2f}", f"{cpk:.2f}" if cpk is not None else "N/A"]
            }
            st.table(pd.DataFrame(stats_data))
            
        except Exception as e:
            st.error(f"🚨 Error processing file: {e}")
    else:
        st.info("📌 Please upload an Excel file to continue.")

if __name__ == "__main__":
    main()
