import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import numpy as np
from scipy.stats import norm

# Function to preprocess data
def preprocess_data(df, x_col, y_col, filter_col_1, filter_col_2):
    # Ensure filter columns are treated as text
    if filter_col_1 != "None":
        df[filter_col_1] = df[filter_col_1].astype(str)
    if filter_col_2 != "None":
        df[filter_col_2] = df[filter_col_2].astype(str)

    # Process x_col and y_col
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
    hover_data = [df.columns[0], df.columns[1]]  # Display the first two columns in the hover data

    if color_col == "None":
        fig = px.scatter(df, x=x_col if x_col != "None" else None, y=y_col, title=title, color_discrete_sequence=['blue'], hover_data=hover_data)
    else:
        fig = px.scatter(df, x=x_col if x_col != "None" else None, y=y_col, color=df[color_col], title=title, color_discrete_sequence=px.colors.qualitative.Set1, hover_data=hover_data)

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

# Function to create histogram based on Y-axis data and filter color grouping
def create_histogram(df, y_col, y_min, y_max, y_upper_limit, y_lower_limit, filter_col_1, filter_col_2):
    # Histogram plot
    if filter_col_1 != "None" and filter_col_2 != "None":
        fig = px.histogram(df, x=y_col, color=[df[filter_col_1], df[filter_col_2]], nbins=20, histnorm='probability', opacity=0.75)
    elif filter_col_1 != "None":
        fig = px.histogram(df, x=y_col, color=filter_col_1, nbins=20, histnorm='probability', opacity=0.75)
    elif filter_col_2 != "None":
        fig = px.histogram(df, x=y_col, color=filter_col_2, nbins=20, histnorm='probability', opacity=0.75)
    else:
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=df[y_col],
            nbinsx=20,  # You can adjust the number of bins here
            histnorm='probability',
            name=f'{y_col} Distribution',
            opacity=0.75
        ))

    # Only apply Y-axis limits if they are set (not None)
    if y_min is not None and y_max is not None:
        fig.update_layout(
            title=f'{y_col} Histogram',
            xaxis_title=y_col,
            yaxis_title='Probability Density',
            bargap=0.2,
            xaxis=dict(range=[y_min, y_max])  # Matching the Y-axis range of scatter plot
        )
        
        # Add limit lines if defined
        if y_upper_limit is not None:
            fig.add_vline(x=y_upper_limit, line=dict(color="red", dash="dash"), annotation_text=f'Upper Limit = {y_upper_limit}')
        if y_lower_limit is not None:
            fig.add_vline(x=y_lower_limit, line=dict(color="black", dash="dash"), annotation_text=f'Lower Limit = {y_lower_limit}')

    # Fit a normal distribution to the data and add a fit line for each filter group
    if filter_col_1 != "None" or filter_col_2 != "None":
        if filter_col_1 != "None":
            groups = df[filter_col_1].dropna().unique()
        elif filter_col_2 != "None":
            groups = df[filter_col_2].dropna().unique()
        for group in groups:
            group_data = df[df[filter_col_1] == group] if filter_col_1 != "None" else df[df[filter_col_2] == group]
            mu, std = norm.fit(group_data[y_col].dropna())
            xmin, xmax = group_data[y_col].min(), group_data[y_col].max()
            x = np.linspace(xmin, xmax, 100)
            p = norm.pdf(x, mu, std)
            fig.add_trace(go.Scatter(
                x=x, y=p, mode='lines', name=f'{group} Fit Line', line=dict(width=2)
            ))
    else:
        mu, std = norm.fit(df[y_col].dropna())
        xmin, xmax = df[y_col].min(), df[y_col].max()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)
        fig.add_trace(go.Scatter(
            x=x, y=p, mode='lines', name='Fit Line', line=dict(color='orange', width=2)
        ))

    return fig

# Streamlit app
def main():
    st.title('📊 Scatter Plot and Histogram Visualization Tool')

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

            filter_col_1 = st.selectbox("🎨 Select Filter Column 1 (for color grouping)", ["None"] + columns)
            filter_col_2 = st.selectbox("🎨 Select Filter Column 2 (for color grouping)", ["None"] + columns)

            # Preprocess data and ensure filter columns are treated as text
            df = preprocess_data(df, x_col if x_col != "None" else None, y_col, filter_col_1, filter_col_2)

            # Filter options for both filter columns
            selected_values_1 = []
            selected_values_2 = []
            if filter_col_1 != "None":
                selected_values_1 = st.multiselect("🎯 Select Filter Value(s) for Filter Column 1", df[filter_col_1].dropna().unique())
                df = df[df[filter_col_1].isin(selected_values_1)]
            if filter_col_2 != "None":
                selected_values_2 = st.multiselect("🎯 Select Filter Value(s) for Filter Column 2", df[filter_col_2].dropna().unique())
                df = df[df[filter_col_2].isin(selected_values_2)]

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

            # Generate scatter plot
            title = f"{x_col if x_col != 'None' else 'Index'} VS {y_col}"
            fig = create_scatter_plot(df, x_col, y_col, title, x_min, x_max, y_min, y_max, x_upper_limit, x_lower_limit, y_upper_limit, y_lower_limit, filter_col_1)
            st.plotly_chart(fig)

            # User option to display histogram
            show_histogram = st.checkbox("Show Histogram", value=False)

            if show_histogram:
                # Generate histogram for Y-axis data with color grouping (if filter columns are selected)
                hist_fig = create_histogram(df, y_col, y_min, y_max, y_upper_limit, y_lower_limit, filter_col_1, filter_col_2)
                st.plotly_chart(hist_fig)

            # Compute statistics for each group if filter columns are selected
            stats_data = []
            if filter_col_1 != "None" or filter_col_2 != "None":
                group_filter = filter_col_1 if filter_col_1 != "None" else filter_col_2
                for group in df[group_filter].dropna().unique():
                    group_data = df[df[group_filter] == group]
                    sample_size = group_data[y_col].dropna().count()
                    mean_value = group_data[y_col].mean()
                    std_value = group_data[y_col].std()
                    cpk = calculate_cpk(mean_value, std_value, y_upper_limit, y_lower_limit)
                    stats_data.append([group, sample_size, f"{mean_value:.2f}", f"{std_value:.2f}", f"{cpk:.2f}" if cpk is not None else "N/A"])

            # Overall statistics
            overall_sample_size = df[y_col].dropna().count()
            overall_mean = df[y_col].mean()
            overall_std = df[y_col].std()
            overall_cpk = calculate_cpk(overall_mean, overall_std, y_upper_limit, y_lower_limit)

            # Add overall stats to the table
            stats_data.insert(0, ['Overall', overall_sample_size, f"{overall_mean:.2f}", f"{overall_std:.2f}", f"{overall_cpk:.2f}" if overall_cpk is not None else "N/A"])

            # 创建 DataFrame，并去除空行
            stats_df = pd.DataFrame(stats_data, columns=["Group", "Sample Size", "Mean", "Std Dev", "CPK"])
            stats_df = stats_df.dropna(how="all").reset_index(drop=True)  # 移除完全为空的行

            # Streamlit 显示交互式表格
            st.subheader("📊 Statistics by Group")
            if not stats_df.empty:
                st.dataframe(stats_df, use_container_width=True)  # 交互式表格
            else:
                st.write("No statistics available for the selected filters.")

        except Exception as e:
            st.error(f"🚨 Error processing file: {e}")
    else:
        st.info("📌 Please upload an Excel file to continue.")

if __name__ == "__main__":
    main()
