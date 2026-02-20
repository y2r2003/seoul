import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import streamlit as st

# 페이지 설정
st.set_page_config(page_title="2025 상권 위험지수 분석", layout="wide")

# -------------------------------
# 1️⃣ 데이터 불러오기
# -------------------------------
# 파일이 같은 경로에 있어야 합니다.
try:
    train_df = pd.read_excel("2019-2024.xlsx")
    test_df = pd.read_excel("2025.xlsx")
except FileNotFoundError:
    st.error("데이터 파일을 찾을 수 없습니다. 파일명을 확인해주세요.")
    st.stop()

# -------------------------------
# 2️⃣ 위험지수 계산 및 스케일링
# -------------------------------
to_scale = [
    'closure_rate',
    'sales_per_store_inv',
    'traffic_conversion_rate',
    'store_density',
    'franchise_ratio_change'
]

scaler = MinMaxScaler()
scaler.fit(train_df[to_scale])

train_scaled = pd.DataFrame(scaler.transform(train_df[to_scale]), columns=to_scale, index=train_df.index)
test_scaled = pd.DataFrame(scaler.transform(test_df[to_scale]), columns=to_scale, index=test_df.index)

# 가중치 정의
weights = {
    'closure_rate': 0.30,
    'sales_per_store_inv': 0.25,
    'traffic_conversion_rate': 0.20,
    'store_density': 0.15,
    'franchise_ratio_change': 0.10
}

# Risk Score 계산
train_df['risk_score'] = 0
test_df['risk_score'] = 0

for col, w in weights.items():
    train_df['risk_score'] += train_scaled[col] * w
    test_df['risk_score'] += test_scaled[col] * w

# -------------------------------
# 3️⃣ 위험 등급 정의 (사분위수 기준)
# -------------------------------
q1 = train_df['risk_score'].quantile(0.25)
q2 = train_df['risk_score'].quantile(0.50)
q3 = train_df['risk_score'].quantile(0.75)

def get_risk_level(score):
    if score < q1: return "Low Risk"
    elif score < q2: return "Medium Risk"
    elif score < q3: return "High Risk"
    else: return "Critical Risk"

test_df['risk_level'] = test_df['risk_score'].apply(get_risk_level)

# -------------------------------
# 4️⃣ UI 메인 화면
# -------------------------------
st.title("📊 2025 상권 위험지수 분석 대시보드")
st.markdown("과거 데이터(2019-2024)를 기준으로 2025년 상권의 상대적 위험도를 측정합니다.")
st.divider()

# 상단 요약 통계
st.subheader("📍 전체 위험 등급 분포")
risk_summary = test_df['risk_level'].value_counts().reindex(["Low Risk", "Medium Risk", "High Risk", "Critical Risk"])
st.bar_chart(risk_summary)

# -------------------------------
# 5️⃣ 구 / 상권 선택 및 결과 출력
# -------------------------------
st.subheader("🔍 지역별 상세 분석")

col_select1, col_select2 = st.columns(2)
with col_select1:
    district_list = sorted(test_df['district'].unique())
    selected_district = st.selectbox("분석할 '구'를 선택하세요", district_list)

filtered_df = test_df[test_df['district'] == selected_district]

with col_select2:
    market_list = sorted(filtered_df['Industry'].unique())
    selected_market = st.selectbox("분석할 '상권(업종)'을 선택하세요", market_list)

# 선택된 데이터 추출
market_row = filtered_df[filtered_df['Industry'] == selected_market].iloc[0]

# 위험 등급별 설정 (아이콘 및 메시지)
risk_info = {
    "Low Risk": {"emoji": "🟢", "msg": "🎉 지금 상권은 위험이 낮습니다. 안정적으로 운영 가능합니다.", "color": "success"},
    "Medium Risk": {"emoji": "🟡", "msg": "⚠️ 지금 상권은 중간 정도의 위험이 있습니다. 주의가 필요합니다.", "color": "info"},
    "High Risk": {"emoji": "🟠", "msg": "🔶 지금 상권은 높은 위험이 있습니다. 전략적 대응을 고려하세요.", "color": "warning"},
    "Critical Risk": {"emoji": "🔴", "msg": "🛑 지금 상권은 매우 위험합니다. 신중한 판단이 필요합니다.", "color": "error"}
}

status = risk_info[market_row['risk_level']]

# 결과 리포트 출력
st.markdown(f"### {status['emoji']} {selected_district} - {selected_market} 분석 결과")

# 메트릭 카드로 시각화
m1, m2, m3 = st.columns(3)
m1.metric("Risk Score", f"{market_row['risk_score']:.4f}")
m2.metric("위험 등급", market_row['risk_level'])
m3.metric("상태", status['emoji'])

# 맞춤 메시지 박스 출력
if status['color'] == "success":
    st.success(status['msg'])
elif status['color'] == "info":
    st.info(status['msg'])
elif status['color'] == "warning":
    st.warning(status['msg'])
else:
    st.error(status['msg'])