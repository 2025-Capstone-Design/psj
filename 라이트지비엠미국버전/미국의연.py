import pandas as pd
import numpy as np
import lightgbm as lgb
import plotly.express as px
import plotly.graph_objects as go
import warnings
# 경고 메시지 무시 설정
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# 1. 환경 설정 및 데이터 로드 (1년 주기로 변경)
# ----------------------------------------------------
# 🚨 1. [필수 수정] 새로 다운로드 받은 1년치 파일 경로와 이름으로 변경하세요.
FILE_PATH = "Awt.cbp.gov_LAX_2024-11-01_to_2025-10-31.csv" 
TARGET = 'MaxWait'
PREDICTION_START_DATE = '2025-11-01 00:00:00' 
# 🚨 2. [필수 수정] 예측 종료일을 내년 10월 말까지로 확장합니다.
PREDICTION_END_DATE = '2026-10-31 23:00:00' 

try:
    df_raw = pd.read_csv(FILE_PATH)
except FileNotFoundError:
    print(f"⚠️ 에러: 파일을 찾을 수 없습니다. 경로를 확인해주세요: {FILE_PATH}")
    exit()

# 2. 데이터 전처리 및 피처 엔지니어링 
# ----------------------------------------------------
df_raw['FlightDate'] = pd.to_datetime(df_raw['FlightDate'])
df_raw['Hour'] = df_raw['HourRange'].str.split(' ').str[0].astype(int)
df_raw['FlightDateTime'] = df_raw.apply(lambda row: row['FlightDate'] + pd.Timedelta(hours=row['Hour'], minutes=30), axis=1)
df_agg = df_raw.groupby('FlightDateTime')[TARGET].max().reset_index()
df = df_agg.rename(columns={TARGET: TARGET})

# ⭐️ [최종 수정] 학습 데이터의 극단적인 이상치(상위 1%)를 제거하여 예측 안정화
train_df_for_outlier = df[df['FlightDateTime'] < PREDICTION_START_DATE].copy()
# MaxWait 값의 99% 백분위수(Percentile) 계산
threshold = train_df_for_outlier[TARGET].quantile(0.99)
print(f"💡 MaxWait 이상치 제거 임계값 (상위 1%): {threshold:.0f}분")

# 임계값보다 큰 MaxWait 값을 임계값으로 대체 (Capping)
# MaxWait = min(MaxWait, threshold)
df[TARGET] = np.where(df[TARGET] > threshold, threshold, df[TARGET])


# 시계열 피처 생성
df['Year'] = df['FlightDateTime'].dt.year
df['Month'] = df['FlightDateTime'].dt.month
df['Day'] = df['FlightDateTime'].dt.day
df['DayOfWeek'] = df['FlightDateTime'].dt.dayofweek
df['Hour'] = df['FlightDateTime'].dt.hour
df['WeekOfYear'] = df['FlightDateTime'].dt.isocalendar().week.astype(int)

# 3. 학습 데이터에서 '역사적 최대 잠재력' 피처 생성
# ----------------------------------------------------
train_df_temp = df[df['FlightDateTime'] < PREDICTION_START_DATE].copy()
max_potential = train_df_temp.groupby(['DayOfWeek', 'Hour'])[TARGET].max().reset_index()
max_potential.rename(columns={TARGET: f'{TARGET}_Historical_Max'}, inplace=True)
df = pd.merge(df, max_potential, on=['DayOfWeek', 'Hour'], how='left')

# 4. 시간 지연 변수 (Lagged Features) 추가 및 최종 피처 정의
# ----------------------------------------------------
LAGS = [24, 24*7] 
for lag in LAGS:
    df[f'{TARGET}_Lag_{lag}'] = df[TARGET].shift(lag)
df.dropna(subset=[f'{TARGET}_Lag_{lag}' for lag in LAGS], inplace=True) 
LAGGED_FEATURES = [f'{TARGET}_Lag_{lag}' for lag in LAGS]
CONTEXTUAL_FEATURE = [f'{TARGET}_Historical_Max'] 
PURE_TIME_FEATURES = ['Month', 'Day', 'DayOfWeek', 'Hour', 'WeekOfYear']
ALL_FEATURES = PURE_TIME_FEATURES + LAGGED_FEATURES + CONTEXTUAL_FEATURE
CATEGORICAL_FEATURES = ['Month', 'DayOfWeek', 'Hour']
for col in CATEGORICAL_FEATURES:
    df[col] = df[col].astype('category')

# 5. 모델 학습 (안정화 파라미터 적용)
# ----------------------------------------------------
train_df = df[df['FlightDateTime'] < PREDICTION_START_DATE].copy()
X_train = train_df[ALL_FEATURES]
y_train = train_df[TARGET]

print("🚀 LightGBM 모델 학습 시작 (이상치 제거 및 안정화 파라미터 적용)...")
lgbm = lgb.LGBMRegressor(
    objective='rmse', 
    n_estimators=1000, 
    learning_rate=0.02, # 학습률 감소
    num_leaves=31,
    random_state=42, 
    n_jobs=-1, 
    metric='rmse', 
    categorical_feature=CATEGORICAL_FEATURES,
    lambda_l1=0.5,  # 정규화 추가
    lambda_l2=0.5,  # 정규화 추가
    min_child_samples=30 # 최소 샘플 수 증가
)
lgbm.fit(X_train, y_train)
print("✅ LightGBM 모델 학습 완료.")

# 6. 재귀적 예측 및 데이터 결합
# ----------------------------------------------------
future_start_dt = pd.to_datetime(PREDICTION_START_DATE)
future_end_dt = pd.to_datetime(PREDICTION_END_DATE)
future_index = pd.date_range(start=future_start_dt, end=future_end_dt, freq='H')

future_df = pd.DataFrame(index=future_index)
future_df.index.name = 'FlightDateTime'
future_df['Month'] = future_df.index.month
future_df['Day'] = future_df.index.day
future_df['DayOfWeek'] = future_df.index.dayofweek
future_df['Hour'] = future_df.index.hour
future_df['WeekOfYear'] = future_df.index.isocalendar().week.astype(int)
future_df = pd.merge(future_df.reset_index(), max_potential, on=['DayOfWeek', 'Hour'], how='left').set_index('FlightDateTime')

all_data = pd.concat([df.set_index('FlightDateTime'), future_df])
train_df_index = df.set_index('FlightDateTime')
LAG_168H = f'{TARGET}_Lag_{LAGS[-1]}'

print("🔄 재귀적 예측 수행 중... (1년 예측)")

for i in range(len(future_df)):
    current_dt = future_df.index[i]
    
    for lag in LAGS:
        past_dt = current_dt - pd.Timedelta(hours=lag)
        if past_dt < future_start_dt:
            if past_dt in train_df_index.index:
                all_data.loc[current_dt, f'{TARGET}_Lag_{lag}'] = train_df_index.loc[past_dt, TARGET]
        elif past_dt in all_data.index:
            all_data.loc[current_dt, f'{TARGET}_Lag_{lag}'] = all_data.loc[past_dt, TARGET]
            
    X_future_row = all_data.loc[[current_dt], ALL_FEATURES]
    
    if X_future_row[LAG_168H].isna().iloc[0]:
        pred_value = X_future_row[f'{TARGET}_Historical_Max'].iloc[0]
    else:
        for col in CATEGORICAL_FEATURES:
            X_future_row[col] = X_future_row[col].astype('category')
        pred_value = lgbm.predict(X_future_row)[0]
    
    all_data.loc[current_dt, TARGET] = pred_value

# 최종 데이터프레임 정리 및 범례 수정
final_future_predictions = all_data.loc[future_index, TARGET].reset_index().rename(columns={TARGET: 'Predicted_MaxWait'})
train_data_for_plot = train_df[['FlightDateTime', TARGET]].rename(columns={TARGET: 'Actual_MaxWait'}).copy()
train_data_for_plot['Predicted_MaxWait'] = np.nan
future_data_for_plot = final_future_predictions
future_data_for_plot['Actual_MaxWait'] = np.nan
full_data = pd.concat([train_data_for_plot, future_data_for_plot], ignore_index=True)

full_data_melted = pd.melt(
    full_data, id_vars=['FlightDateTime'], value_vars=['Actual_MaxWait', 'Predicted_MaxWait'],
    var_name='Type', value_name='MaxWait'
).dropna(subset=['MaxWait'])

full_data_melted['Type'] = full_data_melted['Type'].replace({
    'Actual_MaxWait': '실제 혼잡도 (1년 학습)',
    'Predicted_MaxWait': '예측 혼잡도 (1년 예측)'
})

# 7. Plotly 대화형 그래프 시각화
# ----------------------------------------------------
full_data_melted = full_data_melted.sort_values('FlightDateTime').reset_index(drop=True)
full_data_melted['MaxWait_Smoothed'] = full_data_melted.groupby('Type')['MaxWait'].transform(
    lambda x: x.rolling(window=168, center=True, min_periods=1).median()
)

print("📊 가독성 개선된 대화형 그래프 생성 중...")
fig = go.Figure()
MAX_Y = full_data_melted['MaxWait'].max() * 1.05

# 혼잡 수준 강조 영역
fig.add_hrect(y0=60, y1=120, fillcolor="yellow", opacity=0.1, line_width=0, annotation_text="지연 경고 (60분 초과)", annotation_position="top left")
fig.add_hrect(y0=120, y1=MAX_Y, fillcolor="red", opacity=0.15, line_width=0, annotation_text="심각 혼잡 (120분 초과)", annotation_position="top left")

# 원본 데이터와 평활화 데이터 모두 표시
for name, group in full_data_melted.groupby('Type'):
    color = 'blue' if '실제' in name else 'red'
    
    # 1. 원본 데이터 라인
    fig.add_trace(go.Scatter(
        x=group['FlightDateTime'], y=group['MaxWait'], mode='lines',
        name=f'{name} (시간별 원본)', line=dict(color=color, width=0.8), opacity=0.6,
        hovertemplate='날짜: %{x}<br>최대 혼잡도: %{y:.0f}분<extra></extra>'
    ))

    # 2. 롤링 중앙값 라인
    fig.add_trace(go.Scatter(
        x=group['FlightDateTime'], y=group['MaxWait_Smoothed'], mode='lines',
        name=f'{name} (7일 중앙값)', line=dict(color=color, dash='solid', width=3),
        hovertemplate='날짜: %{x}<br>평균 혼잡도: %{y:.0f}분<extra></extra>',
        visible=True if '실제' in name else 'legendonly' 
    ))

# 예측 기간 시각적 강조
future_start_dt = pd.to_datetime(PREDICTION_START_DATE)
future_end_dt = pd.to_datetime(PREDICTION_END_DATE)
fig.add_vrect(
    x0=future_start_dt, x1=future_end_dt, 
    fillcolor="red", opacity=0.1, line_width=0, annotation_text="1년 예측 기간", annotation_position="top right"
)

# 최종 레이아웃 및 스크롤 기능 설정
fig.update_layout(
    title='✈️ LAX 최대 대기 시간 예측 및 혼잡도 패턴 분석 (이상치 제거 및 안정화)',
    yaxis_title='최대 대기 시간 (분)',
    xaxis_title='날짜', height=700, hovermode="x unified", legend_title_text='데이터 종류', template='plotly_white'
)
fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1개월", step="month", stepmode="backward"),
            dict(count=6, label="6개월", step="month", stepmode="backward"),
            dict(step="all", label="전체")
        ])
    )
)

# 그래프 출력
fig.show()