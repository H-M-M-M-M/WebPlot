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

#绘制散点图
def create_scatter_plot(df, x_col, y_col, title, x_min, x_max, y_min, y_max,
                        x_upper_limit, x_lower_limit, y_upper_limit, y_lower_limit,
                        filter_col_1, filter_col_2, add_trendline=False):
    if y_col not in df.columns:
        raise ValueError(f"❌ Y轴字段 `{y_col}` 不存在于数据中！")
    if x_col != "None" and x_col not in df.columns:
        raise ValueError(f"❌ X轴字段 `{x_col}` 不存在于数据中！")

    # 分组用颜色列
    if filter_col_1 != "None" and filter_col_2 != "None":
        df["__ColorGroup__"] = df[filter_col_1].astype(str) + " | " + df[filter_col_2].astype(str)
        color_col = "__ColorGroup__"
    elif filter_col_1 != "None":
        color_col = filter_col_1
    elif filter_col_2 != "None":
        color_col = filter_col_2
    else:
        color_col = None

    color_seq = px.colors.qualitative.Plotly

    # X轴处理逻辑
    if x_col == "None":
        df["__Index__"] = df.index
        x_plot = "__Index__"
        x_numeric_col = "__Index__"
    else:
        if pd.api.types.is_datetime64_any_dtype(df[x_col]):
            df["__XNumeric__"] = df[x_col].astype(np.int64) / 1e9
            x_plot = x_col
            x_numeric_col = "__XNumeric__"
        else:
            try:
                df["__XNumeric__"] = pd.to_numeric(df[x_col], errors="coerce")
                x_plot = x_col
                x_numeric_col = "__XNumeric__"
            except:
                df["__XNumeric__"] = range(len(df))
                x_plot = x_col
                x_numeric_col = "__XNumeric__"

    # 绘图
    if color_col is None:
        fig = px.scatter(df, x=x_plot, y=y_col, title=title, height=500)
        fig.update_traces(marker=dict(color='blue', size=6))
    else:
        fig = px.scatter(
            df, x=x_plot, y=y_col, color=color_col,
            color_discrete_sequence=color_seq,
            title=title, height=400
        )

    # 趋势线处理
    if add_trendline:
        if color_col is not None:
            unique_groups = df[color_col].unique()
            color_map = {group: color_seq[i % len(color_seq)] for i, group in enumerate(unique_groups)}

            for group in unique_groups:
                group_data = df[df[color_col] == group]
                group_data = group_data.dropna(subset=[x_numeric_col, y_col])
                if len(group_data) >= 2:
                    x_vals = group_data[x_numeric_col].to_numpy().flatten()
                    y_vals = group_data[y_col].to_numpy().flatten()

                    if len(x_vals) == len(y_vals):
                        coeffs = np.polyfit(x_vals, y_vals, deg=1)
                        trend_y = np.polyval(coeffs, x_vals)
                        fig.add_trace(go.Scatter(
                            x=group_data[x_plot],
                            y=trend_y,
                            mode='lines',
                            name=f"Trendline ({group})",
                            line=dict(dash='dash', color=color_map[group])
                        ))
        else:
            df_valid = df.dropna(subset=[x_numeric_col, y_col])
            if len(df_valid) >= 2:
                x_vals = df_valid[x_numeric_col].to_numpy().flatten()
                y_vals = df_valid[y_col].to_numpy().flatten()

                if len(x_vals) == len(y_vals):
                    coeffs = np.polyfit(x_vals, y_vals, deg=1)
                    trend_y = np.polyval(coeffs, x_vals)
                    fig.add_trace(go.Scatter(
                        x=df_valid[x_plot],
                        y=trend_y,
                        mode='lines',
                        name="Trendline",
                        line=dict(dash='dash', color='black')
                    ))

    # 设置坐标轴
    fig.update_layout(
        title=title,
        xaxis_title=x_col if x_col != "None" else "Index",
        yaxis_title=y_col,
        legend_title=color_col if color_col else None
    )
    fig.update_traces(marker=dict(size=6), selector=dict(mode='markers'))

    # 坐标轴范围
    if x_min is not None and x_max is not None:
        fig.update_layout(xaxis=dict(range=[x_min, x_max]))
    if y_min is not None and y_max is not None:
        fig.update_layout(yaxis=dict(range=[y_min, y_max]))

    # 极限线
    if x_upper_limit is not None:
        fig.add_vline(x=x_upper_limit, line=dict(color='red', dash='dash'),
                      annotation_text="X Upper", annotation_position="top right")
    if x_lower_limit is not None:
        fig.add_vline(x=x_lower_limit, line=dict(color='green', dash='dash'),
                      annotation_text="X Lower", annotation_position="bottom right")
    if y_upper_limit is not None:
        fig.add_hline(y=y_upper_limit, line=dict(color='red', dash='dash'),
                      annotation_text="Y Upper", annotation_position="top left")
    if y_lower_limit is not None:
        fig.add_hline(y=y_lower_limit, line=dict(color='green', dash='dash'),
                      annotation_text="Y Lower", annotation_position="bottom left")

    return fig


# Function to create histogram based on Y-axis data and filter color grouping
def create_histogram(df, y_col, y_min, y_max, y_upper_limit, y_lower_limit, filter_col_1, filter_col_2):
    # 创建颜色分组列（如有）
    if filter_col_1 != "None" and filter_col_2 != "None":
        df["__ColorGroup__"] = df[filter_col_1].astype(str) + " | " + df[filter_col_2].astype(str)
        color_col = "__ColorGroup__"
    elif filter_col_1 != "None":
        color_col = filter_col_1
    elif filter_col_2 != "None":
        color_col = filter_col_2
    else:
        color_col = None

    # 创建直方图
    if color_col is None or color_col == "None":
        fig = px.histogram(
            df,
            x=y_col,
            nbins=50,
            histnorm="percent",
            title="📊 Histogram",
            height=400
        )
        fig.update_traces(marker=dict(color='blue'))
    else:
        fig = px.histogram(
            df,
            x=y_col,
            color=color_col,
            color_discrete_sequence=px.colors.qualitative.Plotly,
            barmode="overlay",
            histnorm="percent",
            nbins=50,
            title="📊 Histogram",
            height=400
        )

    # 更新布局
    fig.update_layout(
        xaxis_title=y_col,
        yaxis_title="Percent",
        xaxis=dict(range=[y_min, y_max] if y_min is not None and y_max is not None else None)
    )

    # 添加上下限线
    if y_upper_limit is not None:
        fig.add_vline(x=y_upper_limit, line=dict(color='red', dash='dash'), annotation_text="Y Upper", annotation_position="top right")
    if y_lower_limit is not None:
        fig.add_vline(x=y_lower_limit, line=dict(color='green', dash='dash'), annotation_text="Y Lower", annotation_position="bottom right")

    # 添加拟合曲线
    def add_fit_line(data, label="Fit Line"):
        data = data.dropna()
        if len(data) >= 10:
            mu, std = stats.norm.fit(data)
            x_fit = np.linspace(y_min, y_max, 100)
            y_fit = stats.norm.pdf(x_fit, mu, std)
            y_fit_percent = y_fit / y_fit.sum() * 100
            fig.add_trace(go.Scatter(
                x=x_fit,
                y=y_fit_percent,
                mode='lines',
                name=label,
                line=dict(width=2)
            ))

    if color_col and color_col != "None":
        for group in df[color_col].dropna().unique():
            group_data = df[df[color_col] == group][y_col]
            add_fit_line(group_data, label=f"Fit Line ({group})")
    else:
        add_fit_line(df[y_col])

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

            add_trendline = st.checkbox("📈 Add Trendline", value=False)
            
            # Generate scatter plot
            title = f"{x_col if x_col != 'None' else 'Index'} VS {y_col}"
            fig = create_scatter_plot(df, x_col, y_col, title, x_min, x_max, y_min, y_max, x_upper_limit, x_lower_limit, y_upper_limit, y_lower_limit, filter_col_1, filter_col_2, add_trendline=add_trendline)
            st.plotly_chart(fig, use_container_width=True)

            # User option to display histogram
            show_histogram = st.checkbox("Show Histogram", value=False)

            if show_histogram:
                # Generate histogram for Y-axis data with color grouping (if filter columns are selected)
                hist_fig = create_histogram(df, y_col, y_min, y_max, y_upper_limit, y_lower_limit, filter_col_1, filter_col_2)
                st.plotly_chart(hist_fig)

            # Compute statistics for each group if filter columns are selected
            # 先计算整体统计数据
            overall_sample_size = df[y_col].dropna().count()
            overall_mean = df[y_col].mean()
            overall_std = df[y_col].std()
            overall_cpk = calculate_cpk(overall_mean, overall_std, y_upper_limit, y_lower_limit)

            # 初始化一个列表用于存储所有统计信息
            stats_data = []

            # 显示 Overall 统计信息
            stats_data.append({
                "Group": "Overall",
                "Sample Size": overall_sample_size,
                "Mean": round(overall_mean, 4),
                "Std Dev": round(overall_std, 4),
                "CPK": round(overall_cpk, 4) if overall_cpk is not None else None
            })

            # 如果有选择筛选列
            if filter_col_1 != "None" and filter_col_2 != "None":
                group_filter_1 = filter_col_1
                group_filter_2 = filter_col_2
                # 计算多层筛选后的统计
                for group_1 in df[group_filter_1].dropna().unique():
                    for group_2 in df[df[group_filter_1] == group_1][group_filter_2].dropna().unique():
                        group_data = df[(df[group_filter_1] == group_1) & (df[group_filter_2] == group_2)]
                        sample_size = group_data[y_col].dropna().count()
                        mean_value = group_data[y_col].mean()
                        std_value = group_data[y_col].std()
                        cpk = calculate_cpk(mean_value, std_value, y_upper_limit, y_lower_limit)
                        stats_data.append({
                            "Group": f"{group_1} - {group_2}",
                            "Sample Size": sample_size,
                            "Mean": round(mean_value, 4),
                            "Std Dev": round(std_value, 4),
                            "CPK": round(cpk, 4) if cpk is not None else None
                        })

            elif filter_col_1 != "None":  # 如果只有一个筛选条件
                group_filter = filter_col_1
                for group in df[group_filter].dropna().unique():
                    group_data = df[df[group_filter] == group]
                    sample_size = group_data[y_col].dropna().count()
                    mean_value = group_data[y_col].mean()
                    std_value = group_data[y_col].std()
                    cpk = calculate_cpk(mean_value, std_value, y_upper_limit, y_lower_limit)
                    stats_data.append({
                        "Group": group,
                        "Sample Size": sample_size,
                        "Mean": round(mean_value, 4),
                        "Std Dev": round(std_value, 4),
                        "CPK": round(cpk, 4) if cpk is not None else None
                    })

            # 显示统计数据表格
            if stats_data:
                st.markdown("### 📊 Selected Data Statistics")
                df_stats = pd.DataFrame(stats_data)
                st.dataframe(df_stats, width=1000)  # 设置更宽的表格

        except Exception as e:
            st.error(f"🚨 Error processing file: {e}")
    else:
        st.info("📌 Please upload an Excel file to continue.")

if __name__ == "__main__":
    main()
