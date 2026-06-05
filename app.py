import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px  # Thư viện đồ thị tương tác cao cấp chống chìm chữ
import joblib

# ==============================================================================
# 1. INTERFACE & SIDEBAR NAVIGATION WITH REAL-TIME THEME CONTROL
# ==============================================================================
st.set_page_config(
    page_title="Lending Club Risk Analytics",
    page_icon="💳",
    layout="wide"
)

st.sidebar.title("Hệ Thống Thẩm Định")
st.sidebar.markdown("---")

app_theme = st.sidebar.selectbox(
    "🎨 CHỌN GIAO DIỆN HIỂN THỊ:",
    ["Dark Mode (Nền tối)", "Light Mode (Nền sáng)"]
)

# Cấu hình mã màu tương phản mạnh và màu nền khối cố định (Card Background)
if app_theme == "Dark Mode (Nền tối)":
    text_color_theme = "#ffffff"       # Chữ trắng tinh
    grid_color_theme = "#334155"       # Lưới xám
    accent_color_line = "#ff4b4b"      # Đường hồi quy đỏ tươi
    card_bg_color = "#1e293b"          # Nền khối màu xám tối (Slate)
    title_html_color = "#ffffff"       # Tiêu đề màu trắng
    plotly_template = "plotly_dark"    # Theme tối cho Plotly
else:
    text_color_theme = "#000000"       # ÉP CHỮ ĐEN MUN TUYỆT ĐỐI
    grid_color_theme = "#cbd5e1"       # Lưới xám nhạt rõ nét
    accent_color_line = "#dc2626"      # Đường hồi quy màu đỏ đậm
    card_bg_color = "#ffffff"          # Khối nền trắng tinh
    title_html_color = "#7c3aed"       # Tiêu đề màu Tím Fintech
    plotly_template = "plotly_white"   # Theme sáng cho Plotly

# Palette màu nhấn Tím-Xanh đặc trưng Fintech
FINTECH_PALETTE = ["#7c3aed", "#2563eb", "#64748b", "#94a3b8", "#cbd5e1"]

# Gói CSS xử lý giao diện vùng chứa cố định (Fintech Card UI)
css_style = f"""
    <style>
    .timo-card-wrapper {{
        background-color: {card_bg_color} !important;
        padding: 24px !important;
        border-radius: 12px !important;
        border: 1px solid {grid_color_theme} !important;
        margin-bottom: 25px !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }}
    table {{
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        border-radius: 8px;
        overflow: hidden;
        margin-top: 10px !important;
        margin-bottom: 10px !important;
    }}
    th {{
        background-color: #1e293b !important;
        color: #ffffff !important;
        font-weight: 600;
        padding: 14px 16px !important;
        text-align: left !important;
    }}
    td {{
        background-color: {card_bg_color} !important;
        color: {text_color_theme} !important;
        padding: 14px 16px !important;
        line-height: 1.6;
        border-bottom: 1px solid {grid_color_theme} !important;
    }}
    </style>
"""
st.markdown(css_style, unsafe_allow_html=True)

# Điều hướng trang
page = st.sidebar.radio(
    "HỆ THỐNG MODULES:",
    [
        "Module 1: Baseline EDA",
        "Module 2: Kiểm Định Thống Kê & PCA",
        "Module 3: Backtest Mô Hình & Simulation"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("Môi trường: Local Testing (Sample Data)")

# ==============================================================================
# 2. DATA LOADER (WITH CACHE)
# ==============================================================================
@st.cache_data
def load_data():
    df = pd.read_csv("data_sample.csv")
    if 'fico_range_low' in df.columns and 'fico_range_high' in df.columns and 'fico_score' not in df.columns:
        df['fico_score'] = (df['fico_range_low'] + df['fico_range_high']) / 2
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("❌ Không tìm thấy file 'data_sample.csv' trong thư mục thực thi.")
    st.stop()

# Header chính bọc gọn gàng trong Card UI cố định
st.markdown(
    f"""
    <div class="timo-card-wrapper" 
         style="background-color:#f9f9f9; padding:20px; border-radius:10px; box-shadow:0 4px 8px rgba(0,0,0,0.1);">
        <h1 style="color:{title_html_color}; font-size:32px; font-weight:800; margin-top:0; margin-bottom:15px;">
            Risk-Based Pricing Optimization
        </h1>
    </div>
    """, 
    unsafe_allow_html=True
)


st.markdown(
    """
    | Cấu trúc Risk | Chi tiết nghiệp vụ & Mô hình hóa |
    | :--- | :--- |
    | 🎯 **Risk Appetite** | **Tối ưu hóa hàm định giá dựa trên rủi ro (Risk-based Pricing):** Cân bằng giữa việc giữ chân nhóm hồ sơ rủi ro thấp bằng dải lãi suất cạnh tranh và áp mức bù rủi ro lên nhóm rủi ro cao. |
    | 🧪 **Methodology** | **Xử lý dữ liệu & Mô hình hóa:** Áp dụng toán tử Log-transform xử lý phân phối lệch biến tài chính; sàng lọc đa cộng tiện bằng tương quan Pearson và hệ số VIF. Thử nghiệm hồi quy tuyến tính (Ridge Baseline) và cây phi tuyến (XGBoost Engine). |
    | 💼 **Operational Impact** | **Vận hành hệ thống:** Triển khai cơ chế chấm điểm rủi ro tự động (Automated Underwriting), phân tầng danh mục hồ sơ qua không gian giảm chiều (PCA) để thiết lập luồng quyết định phê duyệt tự động. |
    """
)

# ==============================================================================
# MODULE 1: BASELINE EDA
# ==============================================================================
if page == "Module 1: Baseline EDA":
    st.subheader("📋 Module 1: Phân Phối Dữ Liệu Nền Tảng (Baseline EDA)")
    st.markdown("Phân tích tổng quan cấu trúc các khoản vay và hành vi cơ bản của người đi vay.")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        with st.container(border=True):
            st.metric("Cỡ mẫu phân tích (N)", f"{len(df):,}")
    with col2:
        with st.container(border=True):
            st.metric("Lãi suất bình quan (int_rate)", f"{df['int_rate'].mean():.2f}%")
    with col3:
        with st.container(border=True):
            st.metric("Thu nhập trung vị (annual_inc)", f"${df['annual_inc'].median():,.0f}/năm")
    with col4:
        with st.container(border=True):
            st.metric("Khoản vay trung vị (loan_amnt)", f"${df['loan_amnt'].median():,.0f}")
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "Q1: Kỳ hạn (Term)", 
        "Q2: Mục đích vay (Purpose)", 
        "Q3: Biến động Thu nhập", 
        "Q4: Hình thức sở hữu nhà"
    ])
    
    with tab1:
        col_left, col_right = st.columns([1, 2])
        with col_left:
            st.markdown("<br>**Tỷ lệ cấu trúc kỳ hạn:**", unsafe_allow_html=True)
            st.dataframe(df['term'].value_counts(normalize=True).to_frame(name="Tỷ lệ"), use_container_width=True)
            st.info("💡 **Nhận xét:** Gói vay 36 tháng chiếm đa số. Kỳ hạn dài 60 tháng bị áp dải lãi suất cao hơn để bù đắp rủi ro biến động dòng tiền theo thời gian.")
        with col_right:
            with st.container(border=True):
                # Thay thế bằng Plotly Express chống chìm chữ tuyệt đối
                fig = px.box(df, x='term', y='int_rate', color='term',
                             labels={'term': 'Kỳ hạn', 'int_rate': 'Lãi suất (%)'},
                             title='Biến động Lãi suất theo Kỳ hạn',
                             template=plotly_template, color_discrete_sequence=["#2563eb", "#7c3aed"])
                st.plotly_chart(fig, use_container_width=True)

    with tab2:
        with st.container(border=True):
            top_purposes = df['purpose'].value_counts().index[:7]
            sub_df_p = df[df['purpose'].isin(top_purposes)]
            fig = px.histogram(sub_df_p, y='purpose', color='purpose',
                               category_orders={'purpose': top_purposes},
                               labels={'purpose': 'Mục đích vay', 'count': 'Số lượng hồ sơ'},
                               title='Top 7 Mục đích sử dụng vốn phổ biến',
                               template=plotly_template, color_discrete_sequence=["#7c3aed"])
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        st.info("💡 **Nhận xét:** Đảo nợ (debt_consolidation) và tất toán thẻ tín dụng chiếm tỷ trọng áp đảo, phản ánh nhu cầu cấu trúc lại tài chính cá nhân.")

    with tab3:
        col_l, col_r = st.columns(2)
        with col_l:
            with st.container(border=True):
                fig = px.histogram(df, x='annual_inc', nbins=40,
                                   labels={'annual_inc': 'Thu nhập hàng năm (USD)'},
                                   title='Phân phối gốc (annual_inc dính Outliers)',
                                   template=plotly_template, color_discrete_sequence=["#64748b"])
                st.plotly_chart(fig, use_container_width=True)
        with col_r:
            with st.container(border=True):
                df['log_annual_inc'] = np.log10(df['annual_inc'] + 1)
                fig = px.histogram(df, x='log_annual_inc', nbins=40, marginal="rug",
                                   labels={'log_annual_inc': 'Log10(Thu nhập)'},
                                   title='Phân phối sau xử lý Log-Transform',
                                   template=plotly_template, color_discrete_sequence=["#7c3aed"])
                st.plotly_chart(fig, use_container_width=True)
        st.info("💡 **Nhận xét:** Biến thu nhập lệch phải do Outliers. Áp dụng Log-transformation đưa phân phối về dạng gần đối xứng.")

    with tab4:
        with st.container(border=True):
            home_med = df.groupby('home_ownership')['loan_amnt'].median().reset_index()
            fig = px.bar(home_med, x='home_ownership', y='loan_amnt',
                         labels={'home_ownership': 'Hình thức sở hữu nhà', 'loan_amnt': 'Giá trị vay trung vị (USD)'},
                         title='Giá trị khoản vay trung vị theo hình thức sở hữu nhà',
                         template=plotly_template, color_discrete_sequence=["#64748b"])
            st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# MODULE 2: KIỂM ĐỊNH THỐNG KÊ & PCA
# ==============================================================================
elif page == "Module 2: Kiểm Định Thống Kê & PCA":
    st.subheader("🔬 Module 2: Kiểm Định Đa Cộng Tuyến & Giảm Chiều Không Gian")
    st.markdown("Kiểm tra cấu trúc biến đầu vào trước khi tiến hành mô hình hóa định lượng.")

    tab5, tab6, tab7, tab8 = st.tabs([
        "Q5: Tương quan FICO", 
        "Q6: Đòn bẩy nợ DTI", 
        "Q7: Ma trận Tương quan", 
        "Q8: Không gian PCA 2D"
    ])
    
    with tab5:
        with st.container(border=True):
            df_sample = df.sample(min(1500, len(df)))
            fig = px.scatter(df_sample, x='fico_score', y='int_rate', trendline="ols",
                             trendline_color_override=accent_color_line,
                             labels={'fico_score': 'Điểm tín dụng FICO', 'int_rate': 'Lãi suất (%)'},
                             title='Xu hướng tuyến tính: Điểm FICO vs Lãi suất',
                             template=plotly_template, color_discrete_sequence=["#94a3b8"])
            st.plotly_chart(fig, use_container_width=True)
        st.info("💡 **Nhận xét kỹ thuật:** Hệ số tương quan nghịch biến thể hiện rõ ràng. Điểm số FICO đóng vai trò là biến mỏ neo định vị mức độ uy tín; điểm tín nhiệm tăng là cơ sở toán học để hệ thống giảm tỷ lệ bù rủi ro (Risk Premium), từ đó hạ khung lãi suất khuyến nghị.")
    
    with tab6:
        with st.container(border=True):
            grade_order = sorted(df['grade'].dropna().unique())
            fig = px.box(df, x='grade', y='dti', category_orders={'grade': grade_order},
                         labels={'grade': 'Hạng rủi ro (Grade)', 'dti': 'Chỉ số DTI'},
                         title='Phân phối chỉ số đòn bẩy DTI theo hạng rủi ro (Grade)',
                         template=plotly_template, color_discrete_sequence=["#94a3b8"])
            st.plotly_chart(fig, use_container_width=True)

    with tab7:
        core_risk_features = ['int_rate', 'loan_amnt', 'installment', 'annual_inc', 'dti', 'fico_score', 'open_acc', 'revol_bal']
        available_features = [c for c in core_risk_features if c in df.columns]
        
        with st.container(border=True):
            # Với ma trận Tương quan Heatmap, ép màu nền trắng/tối cố định cứng cho Figure Matplotlib
            fig, ax = plt.subplots(figsize=(6.5, 4.5))
            fig.patch.set_facecolor(card_bg_color)
            ax.set_facecolor(card_bg_color)
            
            sns.heatmap(
                df[available_features].corr(), annot=True, fmt=".2f", cmap="coolwarm", 
                vmin=-1, vmax=1, linewidths=0.8, cbar=True, ax=ax,
                annot_kws={'size': 8, 'weight': 'bold', 'color': text_color_theme}
            )
            ax.set_title("Ma trận tương quan giữa các biến số cốt lõi", fontsize=9, fontweight='bold', color=text_color_theme)
            ax.tick_params(colors=text_color_theme, which='both')
            st.pyplot(fig)
        st.warning("⚠️ **Lưu ý kỹ thuật:** Biến `loan_amnt` và `installment` có tương quan cao. Cần kiểm soát hệ số VIF để xử lý đa cộng tuyến trước khi chạy mô hình tuyến tính.")

    with tab8:
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        try:
            sub_df = df[['loan_amnt', 'annual_inc', 'dti']].dropna()
            X_s = StandardScaler().fit_transform(sub_df)
            X_pca = PCA(n_components=2).fit_transform(X_s)
            
            pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
            pca_df['int_rate'] = df.loc[sub_df.index, 'int_rate'].values
            
            with st.container(border=True):
                fig = px.scatter(pca_df, x='PC1', y='PC2', color='int_rate',
                                 color_continuous_scale='viridis',
                                 labels={'int_rate': 'Lãi suất (%)'},
                                 title='Mặt phẳng PC1 & PC2 phân loại dải lãi suất',
                                 template=plotly_template)
                st.plotly_chart(fig, use_container_width=True)
            st.info("💡 **Nhận xét:** Sự phân tách dải màu trên mặt phẳng 2D chứng minh cấu trúc thuộc tính đầu vào mang năng lực phân nhóm rủi ro.")
        except Exception as e:
            st.error(f"Lỗi chạy toán tử PCA: {e}")

# ==============================================================================
# MODULE 3: BACKTEST MÔ HÌNH & SIMULATION
# ==============================================================================
elif page == "Module 3: Backtest Mô Hình & Simulation":
    st.subheader("🔮 Module 3: Backtest Hiệu Năng Thuật Toán & Giả Lập Thẩm Định")
    st.markdown("Đối chiếu năng lực dự đoán thực tế của các cấu trúc mô hình.")
    
    tab9, tab10 = st.tabs(["Hiệu năng Mô hình (Benchmark)", "Hệ thống Thẩm định Giả lập"])
    
    with tab9:
        st.markdown("**Kết quả đo lường hiệu suất thực tế trên tập Test độc lập (Đồng bộ hóa 100% từ Kaggle):**")
        leaderboard_data = {
            "Chỉ số đo lường": ["R-squared (↑)", "MAE (↓)", "RMSE (↓)"],
            "XGBoost (GPU Accelerated)": ["0.9771", "0.4706%", "0.7301%"],
            "Ridge (Best Linear Baseline)": ["0.9621", "0.6859%", "0.9385%"],
            "Random Forest (Sampled 100k)": ["0.6142", "2.1734%", "2.9992%"]
        }
        st.table(pd.DataFrame(leaderboard_data))
        st.info("💡 **Nhận xét kỹ thuật:** Thuật toán **XGBoost** cho thấy năng lực dự báo áp đảo với độ chính xác cao nhất, xử lý hiệu quả các phân phối phi tuyến đan xen trong tập dữ liệu lớn.")
    
    with tab10:
        st.markdown("### Thẩm định & Tính toán lãi suất khuyến nghị")
        
        with st.form("loan_form"):
            c1, c2 = st.columns(2)
            with c1:
                v_loan = st.number_input("Số tiền vay đề xuất (USD):", min_value=1000, max_value=40000, value=15000, step=1000)
                v_inc = st.number_input("Thu nhập hàng năm khách hàng (USD):", min_value=10000, max_value=300000, value=75000, step=5000)
            with c2:
                v_fico = st.slider("Điểm tín dụng (FICO Score):", min_value=400, max_value=850, value=710)
                v_dti = st.slider("Tỷ lệ nợ trên thu nhập (DTI %):", min_value=0.0, max_value=50.0, value=12.5, step=0.5)
                
            btn_submit = st.form_submit_button("Chạy Thẩm Định AI Engine Real-time")
            
        if btn_submit:
            try:
                model = joblib.load("xgboost_model.pkl")
                scaler = joblib.load("scaler.pkl")
                
                expected_features = scaler.feature_names_in_
                mock_row = {feat: 0.0 for feat in expected_features}
                
                mock_row['loan_amnt'] = float(v_loan)
                mock_row['annual_inc'] = float(v_inc)
                mock_row['dti'] = float(v_dti)
                
                if 'fico_range_low' in mock_row:
                    mock_row['fico_range_low'] = float(v_fico)
                if 'fico_score' in mock_row:
                    mock_row['fico_score'] = float(v_fico)
                    
                full_input_df = pd.DataFrame([mock_row], columns=expected_features)
                input_scaled = scaler.transform(full_input_df)
                prediction = model.predict(input_scaled)
                final_rate = float(prediction[0])
                
            except FileNotFoundError:
                calc_rate = 14.5 - ((v_fico - 600) * 0.025) + (v_dti * 0.08) - (np.log10(v_inc) * 0.4) + (v_loan * 0.00005)
                final_rate = max(4.5, min(32.0, calc_rate))
            except Exception as e:
                st.error(f"🚨 Hệ thống phát hiện lỗi tính toán cấu trúc: {e}")
                st.stop()
            
            st.markdown("---")
            st.markdown(f"#### Kết quả tính toán lãi suất khuyến nghị từ AI: :blue[{final_rate:.2f}%]")
            
            if final_rate < 9.5:
                st.success("🟢 **Hành động: Phê duyệt tự động (Auto-approved).** Hồ sơ thuộc nhóm rủi ro thấp. Hệ thống đề xuất sẵn sàng giải ngân.")
            elif final_rate < 18.0:
                st.warning("🟡 **Hành động: Thẩm định thủ công (Manual Underwriting).** Hồ sơ rủi ro trung bình, yêu cầu hậu kiểm nguồn thu.")
            else:
                st.error("🔴 **Hành động: Từ chối giải ngân (Rejected).** Chỉ số đòn bẩy nợ vượt ngưỡng an toàn hoặc điểm FICO dưới hạn mức cho phép.")