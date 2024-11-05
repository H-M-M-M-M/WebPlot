import pandas as pd
import plotly.express as px
import streamlit as st

# Function to preprocess data
def preprocess_data(df, x_col, y_col):
    if x_col:
        if 'Date' in x_col:
            df[x_col] = pd.to_datetime(df[x_col], errors='coerce')  # Convert to datetime
        else:
            df[x_col] = df[x_col].astype(str).replace('[^-?\d.]', '', regex=True)  # Retain negative sign and numeric characters
            df[x_col] = pd.to_numeric(df[x_col], errors='coerce')  # Convert to numeric, forcing errors to NaN
    
    df[y_col] = df[y_col].astype(str).replace('[^-?\d.]', '', regex=True)  # Retain negative sign and numeric characters
    df[y_col] = pd.to_numeric(df[y_col], errors='coerce')  # Convert to numeric, forcing errors to NaN

    return df

# Function to calculate CPK
def calculate_cpk(mean, std, upper_limit, lower_limit):
    if upper_limit is not None and lower_limit is not None:
        if upper_limit > mean and lower_limit < mean:
            return min((upper_limit - mean) / (3 * std), (mean - lower_limit) / (3 * std))
    return None

# Function to create scatter plot
def create_scatter_plot(df, x_col, y_col, title, subtitle, x_min, x_max, y_min, y_max, y_upper_limit, y_lower_limit):
    if x_col == "None":
        fig = px.scatter(df, y=y_col, title=title, color_discrete_sequence=['blue'])
    else:
        fig = px.scatter(df, x=x_col, y=y_col, title=title, color_discrete_sequence=['blue'])

    fig.update_traces(marker=dict(size=8), selector=dict(mode='markers'))
    fig.update_layout(
        title={
            'text': title,
            'x': 0.5,  # Center the title
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title=x_col if x_col != 'None' else None,
        yaxis_title=y_col,
        xaxis=dict(range=[x_min, x_max] if x_col != "None" and (x_min is not None or x_max is not None) else None),
        yaxis=dict(range=[y_min, y_max] if y_min is not None and y_max is not None else None),
        annotations=[
            dict(
                x=0.5,
                y=-0.2,  # Adjusted to avoid overlap with the plot
                xref='paper',
                yref='paper',
                text=subtitle,
                showarrow=False,
                font=dict(size=10, color="gray"),  # Reduced font size for the subtitle
                xanchor='center',
                yanchor='top'
            )
        ]
    )
    
    # Draw horizontal lines if limits are provided
    if y_upper_limit is not None:
        fig.add_hline(y=y_upper_limit, line=dict(color="red", dash="dash"), annotation_text=f'Upper Limit = {y_upper_limit}')

    if y_lower_limit is not None:
        fig.add_hline(y=y_lower_limit, line=dict(color="black", dash="dash"), annotation_text=f'Lower Limit = {y_lower_limit}')

    return fig

# Streamlit app
def main():
    st.title('Scatter Plot Drawing Application')

    # File upload with better error handling
    uploaded_file = st.file_uploader("选择一个Excel文件", type="xlsx")
    if uploaded_file:
        try:
            xlsx = pd.ExcelFile(uploaded_file)
            sheet_names = xlsx.sheet_names
            sheet = st.selectbox("选择工作表", sheet_names)
            df = pd.read_excel(uploaded_file, sheet_name=sheet)

            # Select columns for X and Y axes
            columns = df.columns.tolist()
            x_col = st.selectbox("选择X轴列（可选）", ["None"] + columns)
            y_col = st.selectbox("选择Y轴列", columns)

            # Preprocess data
            df = preprocess_data(df, x_col if x_col != "None" else None, y_col)

            # Filter options
            filter_col = st.selectbox("选择筛选列（可选）", ["None"] + columns)
            if filter_col != "None":
                filter_values = df[filter_col].dropna().unique()
                selected_values = st.multiselect("选择筛选值", filter_values, default=filter_values.tolist())
                df = df[df[filter_col].isin(selected_values)]

            # X and Y axis limits
            if x_col != "None" and pd.api.types.is_datetime64_any_dtype(df[x_col]):
                x_min = st.date_input("X轴最小值", value=df[x_col].min().to_pydatetime())
                x_max = st.date_input("X轴最大值", value=df[x_col].max().to_pydatetime())
            else:
                x_min = st.number_input("X轴最小值", value=float(df[x_col].min()) if x_col != "None" and not df[x_col].isnull().all() else None, help="设置X轴最小值")
                x_max = st.number_input("X轴最大值", value=float(df[x_col].max()) if x_col != "None" and not df[x_col].isnull().all() else None, help="设置X轴最大值")
            
            y_min = st.number_input("Y轴最小值", value=float(df[y_col].min()) if not df[y_col].isnull().all() else None, help="设置Y轴最小值")
            y_max = st.number_input("Y轴最大值", value=float(df[y_col].max()) if not df[y_col].isnull().all() else None, help="设置Y轴最大值")

            # Input lines to draw
            y_upper_limit = st.number_input("Y轴上限（Upper Limit）", value=None, help="设置Y轴的上规格限")
            y_lower_limit = st.number_input("Y轴下限（Lower Limit）", value=None, help="设置Y轴的下规格限")

            # Calculate statistics
            sample_size = df[y_col].dropna().count()
            mean_value = df[y_col].mean()
            std_value = df[y_col].std()

            # Calculate CPK
            cpk = calculate_cpk(mean_value, std_value, y_upper_limit, y_lower_limit)

            # Generate custom title and subtitle
            title = f"{x_col if x_col != 'None' else 'Index'} VS {y_col}"
            subtitle = f"Sample Size: {sample_size}, Mean: {mean_value:.2f}, Std: {std_value:.2f}"
            if cpk is not None:
                subtitle += f", CPK: {cpk:.2f}"

            # Create scatter plot
            fig = create_scatter_plot(df, x_col, y_col, title, subtitle, x_min, x_max, y_min, y_max, y_upper_limit, y_lower_limit)

            # Show plot
            st.plotly_chart(fig)

        except Exception as e:
            st.error(f"处理文件时出错: {e}")
    else:
        st.info("请上传一个Excel文件以继续")

if __name__ == "__main__":
    main()
