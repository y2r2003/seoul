import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import streamlit as st

# -------------------------------
# 0️⃣ 페이지 설정
# -------------------------------
st.set_page_config(page_title="2025 상권 위험지수", layout="wide")
st.title("📊 2025 상권 위험지수 분석")

# -------------------------------
# 1️⃣ 데이터 불러오기
# -------------------------------
train_df = pd.read_excel("2019-2024.xlsx")
test_df = pd.read_excel("2025.xlsx")

# 컬럼 공백 제거 + 소문자 통일
train_df.columns = train_df.columns.str.strip().str.lower()
test_df.columns = test_df.columns.str.strip().str.lower()

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
# 3️⃣ Min-Max 스케일링
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
# 6️⃣ Risk Level 정의 (사분위수 기준)
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
# 8️⃣ 구 / 상권 선택
# -------------------------------
st.subheader("구 / 상권별 위험지수 확인")

district_list = sorted(test_df['district'].unique())
selected_district = st.selectbox("구 선택", district_list)

filtered_df = test_df[test_df['district'] == selected_district]

market_list = sorted(filtered_df['industry'].unique())
selected_market = st.selectbox("상권 선택", market_list)

# -------------------------------
# 9️⃣ 상권 존재 여부 체크
# -------------------------------
selected_market_df = filtered_df[filtered_df['industry'] == selected_market]

if selected_market_df.empty:
    st.warning("선택한 구/상권 데이터가 없습니다.")
else:
    market_row = selected_market_df.iloc[0]
    
    # Risk Level → 문자열로 변환
    level = str(market_row['risk_level']).strip()
    
    # -------------------------------
    # 10️⃣ 위험 분석 결과 표시
    # -------------------------------
    color_map = {
        "Low Risk": "🟢",
        "Medium Risk": "🟡",
        "High Risk": "🟠",
        "Critical Risk": "🔴"
    }

    st.markdown(f"**{color_map.get(level, '⚪')} 위험 분석 결과**")
    st.write(f"- 구: {market_row['district']}")
    st.write(f"- 상권: {market_row['industry']}")
    st.write(f"- Risk Score: {market_row['risk_score']:.4f}")
    st.write(f"- Risk Level: {level}")

    # -------------------------------
    # 11️⃣ Risk Level별 친절 멘트
    # -------------------------------
    risk_messages = {
        "Low Risk": "🎉 지금 상권은 위험이 낮습니다. 안정적으로 운영 가능합니다.",
        "Medium Risk": "⚠️ 지금 상권은 중간 정도의 위험이 있습니다. 주의가 필요합니다.",
        "High Risk": "🔶 지금 상권은 높은 위험이 있습니다. 전략적 대응을 고려하세요.",
        "Critical Risk": "🛑 지금 상권은 매우 위험합니다. 신중한 판단이 필요합니다."
    }

    message = risk_messages.get(level, "정보를 확인할 수 없습니다.")
    st.write(message)