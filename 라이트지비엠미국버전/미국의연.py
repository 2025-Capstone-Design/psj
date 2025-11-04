import pandas as pd
import numpy as np
import lightgbm as lgb
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# 1. 환경 설정 및 데이터 로드 
# ----------------------------------------------------
FILE_PATH = "Awt.cbp.gov_LAX_2024-11-01_to_2025-10-31.csv" 
TARGET = 'MaxWait'
PREDICTION_START_DATE = '2025-11-01 00:00:00' 
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

df_original_for_plot = df_agg.copy().rename(columns={TARGET: 'Actual_MaxWait_Original'})
df = df_agg.rename(columns={TARGET: TARGET})

future_start_dt = pd.to_datetime(PREDICTION_START_DATE)

# 모델 학습 안정화를 위한 이상치 처리 (Capping)를 df에만 적용
train_df_for_outlier = df[df['FlightDateTime'] < future_start_dt].copy()
threshold = train_df_for_outlier[TARGET].quantile(0.99)
print(f"💡 LightGBM 학습용 MaxWait 이상치 제거 임계값 (상위 1%): {threshold:.0f}분")
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
train_df_temp = df[df['FlightDateTime'] < future_start_dt].copy()
max_potential = train_df_temp.groupby(['DayOfWeek', 'Hour'])[TARGET].max().reset_index()
max_potential.rename(columns={TARGET: f'{TARGET}_Historical_Max'}, inplace=True)
df = pd.merge(df, max_potential, on=['DayOfWeek', 'Hour'], how='left')

# 4. 시간 지연 변수 (Lagged Features) 추가 및 최종 피처 정의
# ----------------------------------------------------
LAGS = [24, 24*7] 
# Lagged Feature 생성 시, 예측 시작일 이전의 데이터만 사용하여 shift를 수행합니다.
df_train_only = df[df['FlightDateTime'] < future_start_dt].copy()

for lag in LAGS:
    df_train_only[f'{TARGET}_Lag_{lag}'] = df_train_only[TARGET].shift(lag)

# 전체 데이터프레임에 Lagged Feature를 병합합니다. (이때 NaN이 생기는 행은 모델 학습에서 제외)
df = pd.merge(df, df_train_only[[f'{TARGET}_Lag_{lag}' for lag in LAGS] + ['FlightDateTime']], 
              on='FlightDateTime', how='left')

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
train_df = df[df['FlightDateTime'] < future_start_dt].copy()
X_train = train_df[ALL_FEATURES]
y_train = train_df[TARGET]

print("🚀 LightGBM 모델 학습 시작 (최종 안정화 파라미터 적용)...")
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
    
    # ⭐️ Lagged Feature를 가져오는 로직 개선: 과거 데이터 인덱스를 사용하여 명확하게 참조
    for lag in LAGS:
        past_dt = current_dt - pd.Timedelta(hours=lag)
        if past_dt in all_data.index:
             # 예측 구간에서는 예측값을, 학습 구간에서는 학습된 값을 참조합니다.
            all_data.loc[current_dt, f'{TARGET}_Lag_{lag}'] = all_data.loc[past_dt, TARGET]
        # 만약 과거 데이터가 없으면 (가장 초반 예측), NaN이 되도록 둡니다.
        # 이 NaN은 아래 Historical Max 로직으로 처리됩니다.
    
    X_future_row = all_data.loc[[current_dt], ALL_FEATURES]
    
    # 예측 시작 후 7일 이내 (T-168 Lagged Feature가 NaN일 경우) Historical Max 사용
    if X_future_row[LAG_168H].isna().iloc[0] or X_future_row[LAGGED_FEATURES].isna().any(axis=1).iloc[0]:
        pred_value = X_future_row[f'{TARGET}_Historical_Max'].iloc[0]
    else:
        for col in CATEGORICAL_FEATURES:
            X_future_row[col] = X_future_row[col].astype('category')
        pred_value = lgbm.predict(X_future_row)[0]
    
    all_data.loc[current_dt, TARGET] = pred_value

# ⭐️ [시각화 데이터 최종 수정] 파란색 선이 11월 이후에 표시되는 오류 방지
final_future_predictions = all_data.loc[future_index, TARGET].reset_index().rename(columns={TARGET: 'Predicted_MaxWait'})

# 파란색 선은 예측 시작일 전의 원본 데이터만 포함 (2025-11-01 직전까지)
train_data_for_plot = df_original_for_plot[
    df_original_for_plot['FlightDateTime'] < future_start_dt
].rename(columns={'Actual_MaxWait_Original': 'Actual_MaxWait'}).copy()
train_data_for_plot['Predicted_MaxWait'] = np.nan

future_data_for_plot = final_future_predictions
future_data_for_plot['Actual_MaxWait'] = np.nan

full_data = pd.concat([train_data_for_plot, future_data_for_plot], ignore_index=True)

full_data_melted = pd.melt(
    full_data, id_vars=['FlightDateTime'], value_vars=['Actual_MaxWait', 'Predicted_MaxWait'],
    var_name='Type', value_name='MaxWait'
).dropna(subset=['MaxWait'])

full_data_melted['Type'] = full_data_melted['Type'].replace({
    'Actual_MaxWait': '실제 혼잡도 (원본 데이터)',
    'Predicted_MaxWait': '예측 혼잡도 (안정화 모델)'
})

# 7. Plotly 대화형 그래프 시각화
# ----------------------------------------------------
full_data_melted = full_data_melted.sort_values('FlightDateTime').reset_index(drop=True)
full_data_melted['MaxWait_Smoothed'] = full_data_melted.groupby('Type')['MaxWait'].transform(
    lambda x: x.rolling(window=168, center=True, min_periods=1).median()
)

print("📊 가독성 개선된 대화형 그래프 생성 중...")
fig = go.Figure()

max_actual = df_original_for_plot['Actual_MaxWait_Original'].max()
max_predicted = final_future_predictions['Predicted_MaxWait'].max()
MAX_Y = max(max_actual, max_predicted) * 1.05

fig.add_hrect(y0=60, y1=120, fillcolor="yellow", opacity=0.1, line_width=0, annotation_text="지연 경고 (60분 초과)", annotation_position="top left")
fig.add_hrect(y0=120, y1=MAX_Y, fillcolor="red", opacity=0.15, line_width=0, annotation_text="심각 혼잡 (120분 초과)", annotation_position="top left")

for name, group in full_data_melted.groupby('Type'):
    color = 'blue' if '실제' in name else 'red'
    
    fig.add_trace(go.Scatter(
        x=group['FlightDateTime'], y=group['MaxWait'], mode='lines',
        name=f'{name} (시간별 원본)', line=dict(color=color, width=0.8), opacity=0.6,
        hovertemplate='날짜: %{x}<br>최대 혼잡도: %{y:.0f}분<extra></extra>'
    ))

    fig.add_trace(go.Scatter(
        x=group['FlightDateTime'], y=group['MaxWait_Smoothed'], mode='lines',
        name=f'{name} (7일 중앙값)', line=dict(color=color, dash='solid', width=3),
        hovertemplate='날짜: %{x}<br>평균 혼잡도: %{y:.0f}분<extra></extra>',
        visible=True if '실제' in name else 'legendonly' 
    ))

future_end_dt = pd.to_datetime(PREDICTION_END_DATE)
fig.add_vrect(
    x0=future_start_dt, x1=future_end_dt, 
    fillcolor="red", opacity=0.1, line_width=0, annotation_text="1년 예측 기간", annotation_position="top right"
)

fig.update_layout(
    title='✈️ LAX 최대 대기 시간 예측 및 혼잡도 패턴 분석 (예측 비현실성 최종 해결)',
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