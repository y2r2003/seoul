import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import streamlit as st

st.set_page_config(page_title="2025 상권 위험지수", layout="wide")

st.title("📊 2025 상권 위험지수 분석")

# -------------------------------
# 1️⃣ 데이터 불러오기
# -------------------------------
train_df = pd.read_excel("2019-2024.xlsx")
test_df = pd.read_excel("2025.xlsx")

# -------------------------------
# 2️⃣ 위험지수 계산에 사용할 컬럼
# -------------------------------
to_scale = [
    'closure_rate',
    'sales_per_store_inv',
    'traffic_conversion_rate',
    'store_density',
    'franchise_ratio_change'
]

# -------------------------------
# 3️⃣ Train 기준 Min-Max 스케일링
# -------------------------------
scaler = MinMaxScaler()
scaler.fit(train_df[to_scale])

train_scaled = pd.DataFrame(
    scaler.transform(train_df[to_scale]),
    columns=to_scale,
    index=train_df.index
)

test_scaled = pd.DataFrame(
    scaler.transform(test_df[to_scale]),
    columns=to_scale,
    index=test_df.index
)

# -------------------------------
# 4️⃣ 가중치 정의
# -------------------------------
weights = {
    'closure_rate': 0.30,
    'sales_per_store_inv': 0.25,
    'traffic_conversion_rate': 0.20,
    'store_density': 0.15,
    'franchise_ratio_change': 0.10
}

# -------------------------------
# 5️⃣ Risk Score 계산
# -------------------------------
train_df['risk_score'] = 0
test_df['risk_score'] = 0

for col, w in weights.items():
    train_df['risk_score'] += train_scaled[col] * w
    test_df['risk_score'] += test_scaled[col] * w

# -------------------------------
# 6️⃣ Train 기준 사분위수로 Risk Level 정의
# -------------------------------
q1 = train_df['risk_score'].quantile(0.25)
q2 = train_df['risk_score'].quantile(0.50)
q3 = train_df['risk_score'].quantile(0.75)

def risk_level(score):
    if score < q1:
        return "Low Risk"
    elif score < q2:
        return "Medium Risk"
    elif score < q3:
        return "High Risk"
    else:
        return "Critical Risk"

test_df['risk_level'] = test_df['risk_score'].apply(risk_level)

# -------------------------------
# 7️⃣ 전체 위험 분포 시각화
# -------------------------------
st.subheader("전체 위험 등급 분포")

risk_summary = test_df['risk_level'].value_counts()
st.bar_chart(risk_summary)

# -------------------------------
# 8️⃣ 구 / 상권 선택 팝업형 표시
# -------------------------------
st.subheader("구 / 상권별 위험지수 확인")

district_list = sorted(test_df['district'].unique())
selected_district = st.selectbox("구 선택", district_list)

filtered_df = test_df[test_df['district'] == selected_district]

market_list = sorted(filtered_df['market'].unique())
selected_market = st.selectbox("상권 선택", market_list)

market_row = filtered_df[filtered_df['market'] == selected_market].iloc[0]

# 위험 등급별 색상 설정
color_map = {
    "Low Risk": "🟢",
    "Medium Risk": "🟡",
    "High Risk": "🟠",
    "Critical Risk": "🔴"
}

st.info(
    f"""
    {color_map[market_row['risk_level']]} **위험 분석 결과**

    - 구: {market_row['district']}
    - 상권: {market_row['market']}
    - Risk Score: {market_row['risk_score']:.4f}
    - Risk Level: {market_row['risk_level']}
    """
)