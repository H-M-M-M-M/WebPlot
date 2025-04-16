import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import numpy as np
import scipy.stats as stats
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
def create_scatter_plot(df, x_col, y_col, title, x_min, x_max, y_min, y_max,
                        x_upper_limit, x_lower_limit, y_upper_limit, y_lower_limit,
                        filter_col_1, filter_col_2):

    # 创建组合颜色列（如有）
    if filter_col_1 != "None" and filter_col_2 != "None":
        df["__ColorGroup__"] = df[filter_col_1].astype(str) + " | " + df[filter_col_2].astype(str)
        color_col = "__ColorGroup__"
    elif filter_col_1 != "None":
        color_col = filter_col_1
    elif filter_col_2 != "None":
        color_col = filter_col_2
    else:
        color_col = None

    fig = px.scatter(
        df,
        x=x_col if x_col != "None" else df.index,
        y=y_col,
        color=color_col,
        title=title,
        height=500
    )

    fig.update_layout(title=title, xaxis_title=x_col, yaxis_title=y_col)
    fig.update_traces(marker=dict(size=6), selector=dict(mode='markers'))

    fig.update_layout(
        xaxis=dict(range=[x_min, x_max] if x_min is not None and x_max is not None else None),
        yaxis=dict(range=[y_min, y_max] if y_min is not None and y_max is not None else None)
    )

    if x_upper_limit is not None:
        fig.add_vline(x=x_upper_limit, line=dict(color='red', dash='dash'), annotation_text="X Upper", annotation_position="top right")
    if x_lower_limit is not None:
        fig.add_vline(x=x_lower_limit, line=dict(color='green', dash='dash'), annotation_text="X Lower", annotation_position="bottom right")
    if y_upper_limit is not None:
        fig.add_hline(y=y_upper_limit, line=dict(color='red', dash='dash'), annotation_text="Y Upper", annotation_position="top left")
    if y_lower_limit is not None:
        fig.add_hline(y=y_lower_limit, line=dict(color='green', dash='dash'), annotation_text="Y Lower", annotation_position="bottom left")

    return fig

# Function to create histogram based on Y-axis data and filter color grouping
def create_histogram(df, y_col, y_min, y_max, y_upper_limit, y_lower_limit, filter_col_1, filter_col_2):
    if filter_col_1 != "None" and filter_col_2 != "None":
        df["__ColorGroup__"] = df[filter_col_1].astype(str) + " | " + df[filter_col_2].astype(str)
        color_col = "__ColorGroup__"
    elif filter_col_1 != "None":
        color_col = filter_col_1
    elif filter_col_2 != "None":
        color_col = filter_col_2
    else:
        color_col = None

    # Create histogram
    fig = px.histogram(
        df,
        x=y_col,
        color=color_col,
        barmode="overlay",
        histnorm="percent",
        nbins=50,
        height=400
    )

    # Update layout for axis labels
    fig.update_layout(xaxis_title=y_col, yaxis_title="Percent")
    fig.update_layout(
        xaxis=dict(range=[y_min, y_max] if y_min is not None and y_max is not None else None)
    )

    # Add vertical lines for upper and lower limits
    if y_upper_limit is not None:
        fig.add_vline(x=y_upper_limit, line=dict(color='red', dash='dash'), annotation_text="Y Upper", annotation_position="top right")
    if y_lower_limit is not None:
        fig.add_vline(x=y_lower_limit, line=dict(color='green', dash='dash'), annotation_text="Y Lower", annotation_position="bottom right")

    # Add fit line (Normal Distribution Fit) for each group
    if color_col:
        for group in df[color_col].dropna().unique():
            group_data = df[df[color_col] == group][y_col].dropna()
            
            if len(group_data) > 0:
                # Fit data to a normal distribution
                mu, std = stats.norm.fit(group_data)
                
                # Generate the fitted line (probability density function)
                x_fit = np.linspace(y_min, y_max, 100)
                y_fit = stats.norm.pdf(x_fit, mu, std)
                
                # Add the fit line to the plot for this group
                fig.add_trace(go.Scatter(
                    x=x_fit, 
                    y=y_fit * np.max(np.histogram(group_data, bins=50)[0]) * (y_max - y_min) / 50,  # Scaling to match histogram height
                    mode='lines', 
                    name=f"Fit Line ({group})",
                    line=dict(width=2)
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
            fig = create_scatter_plot(df, x_col, y_col, title, x_min, x_max, y_min, y_max, x_upper_limit, x_lower_limit, y_upper_limit, y_lower_limit, filter_col_1, filter_col_2)
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
                    stats_data.append([group, sample_size, f"{mean_value:.4f}", f"{std_value:.4f}", f"{cpk:.2f}" if cpk is not None else "N/A"])

            # Overall statistics
            overall_sample_size = df[y_col].dropna().count()
            overall_mean = df[y_col].mean()
            overall_std = df[y_col].std()
            overall_cpk = calculate_cpk(overall_mean, overall_std, y_upper_limit, y_lower_limit)

            # Add overall stats to the table
            stats_data.insert(0, ['Overall', overall_sample_size, f"{overall_mean:.4f}", f"{overall_std:.4f}", f"{overall_cpk:.2f}" if overall_cpk is not None else "N/A"])

            # Display statistics
            #st.markdown("### 📊 Selected Data Statistics")
            #stats_df = pd.DataFrame(stats_data, columns=[group_filter if filter_col_1 != "None" or filter_col_2 != "None" else "Group", "Sample Size", "Mean", "Std Dev", "CPK"])
            #st.dataframe(stats_df)

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
