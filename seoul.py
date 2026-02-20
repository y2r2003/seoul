risk_app/
├─ app.py                  # Streamlit 앱 메인 코드
├─ train_2019_2024.csv     # 파생변수까지 포함된 train 데이터
├─ test_2025.csv            # 2025 데이터 (파생변수 포함)
├─ requirements.txt        # 필요한 라이브러리

pandas
streamlit
scikit-learn
openpyxl

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import streamlit as st

# -------------------------------
# 1️⃣ 데이터 불러오기
# -------------------------------
train_df = pd.read_csv("train_2019_2024.csv")
test_df  = pd.read_csv("test_2025.csv")

# -------------------------------
# 2️⃣ 위험지수 계산에 필요한 컬럼
# -------------------------------
to_scale = ['closure_rate', 'sales_per_store_inv', 'traffic_conversion_rate', 'store_density', 'franchise_ratio_change']

# -------------------------------
# 3️⃣ Min-Max 스케일링 (train 기준)
# -------------------------------
scaler = MinMaxScaler()
scaler.fit(train_df[to_scale])

train_scaled = pd.DataFrame(scaler.transform(train_df[to_scale]), columns=to_scale, index=train_df.index)
test_scaled  = pd.DataFrame(scaler.transform(test_df[to_scale]), columns=to_scale, index=test_df.index)

# -------------------------------
# 4️⃣ 가중치 정의 및 risk_score 계산
# -------------------------------
weights = {
    'closure_rate': 0.30,
    'sales_per_store_inv': 0.25,
    'traffic_conversion_rate': 0.20,
    'store_density': 0.15,
    'franchise_ratio_change': 0.10
}

train_df['risk_score'] = 0
test_df['risk_score'] = 0
for col, w in weights.items():
    train_df['risk_score'] += train_scaled[col] * w
    test_df['risk_score']  += test_scaled[col] * w

# -------------------------------
# 5️⃣ 사분위수 기준 risk_level 부여 (train 기준)
# -------------------------------
q1 = train_df['risk_score'].quantile(0.25)
q2 = train_df['risk_score'].quantile(0.50)
q3 = train_df['risk_score'].quantile(0.75)

def risk_level(score):
    if score < q1:
        return 'Low Risk'
    elif score < q2:
        return 'Medium Risk'
    elif score < q3:
        return 'High Risk'
    else:
        return 'Critical Risk'

test_df['risk_level'] = test_df['risk_score'].apply(risk_level)

# -------------------------------
# 6️⃣ Streamlit 앱 UI
# -------------------------------
st.title("2025 상권 위험지수 분석")

# 6-1. 전체 요약
st.subheader("전체 위험지수 요약")
risk_summary = test_df['risk_level'].value_counts()
st.bar_chart(risk_summary)

# 6-2. 구/상권별 팝업형 선택
st.subheader("구/상권별 위험지수 확인")
district_list = test_df['district'].unique()
selected_district = st.selectbox("구 선택", district_list)

sub_df = test_df[test_df['district'] == selected_district]
market_list = sub_df['market'].unique()
selected_market = st.selectbox("상권 선택", market_list)

market_row = sub_df[sub_df['market'] == selected_market].iloc[0]

st.info(
    f"구: {market_row['district']}\n"
    f"상권: {market_row['market']}\n"
    f"Risk Score: {market_row['risk_score']:.3f}\n"
    f"Risk Level: {market_row['risk_level']}"
)