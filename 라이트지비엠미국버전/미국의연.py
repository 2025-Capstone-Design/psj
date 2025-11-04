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
df_train_only = df[df['FlightDateTime'] < future_start_dt].copy()

for lag in LAGS:
    df_train_only[f'{TARGET}_Lag_{lag}'] = df_train_only[TARGET].shift(lag)

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

# 5. 모델 학습 
# ----------------------------------------------------
train_df = df[df['FlightDateTime'] < future_start_dt].copy()
X_train = train_df[ALL_FEATURES]
y_train = train_df[TARGET]

print("🚀 LightGBM 모델 학습 시작 (최종 안정화 파라미터 적용)...")
lgbm = lgb.LGBMRegressor(
    objective='rmse', n_estimators=1000, learning_rate=0.02, num_leaves=31, random_state=42, 
    n_jobs=-1, metric='rmse', categorical_feature=CATEGORICAL_FEATURES,
    lambda_l1=0.5, lambda_l2=0.5, min_child_samples=30
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

# ⭐️ [핵심 변수] 예측값 평활화를 위한 가중치 설정 (이전 예측값 70%, 새로운 예측값 30% 반영)
SMOOTHING_WEIGHT = 0.7 
LAST_ACTUAL_VALUE = df[df['FlightDateTime'] < future_start_dt].sort_values('FlightDateTime').iloc[-1][TARGET]

print("🔄 재귀적 예측 수행 중... (예측값 평활화 적용)")

# 예측 시작 전 마지막 실제 값으로 초기 예측값을 설정 (불연속성 완화)
all_data.loc[future_index[0], TARGET] = LAST_ACTUAL_VALUE 

for i in range(len(future_df)):
    current_dt = future_df.index[i]
    
    # Lagged Feature 참조
    for lag in LAGS:
        past_dt = current_dt - pd.Timedelta(hours=lag)
        if past_dt in all_data.index:
            all_data.loc[current_dt, f'{TARGET}_Lag_{lag}'] = all_data.loc[past_dt, TARGET]
    
    X_future_row = all_data.loc[[current_dt], ALL_FEATURES]
    
    # Lagged Feature에 NaN이 있다면 Historical Max로 강제 대체
    if X_future_row[LAGGED_FEATURES].isna().any(axis=1).iloc[0]:
        for lag_col in LAGGED_FEATURES:
            if X_future_row[lag_col].isna().iloc[0]:
                X_future_row.loc[X_future_row.index, lag_col] = X_future_row[f'{TARGET}_Historical_Max'].iloc[0]
        
    for col in CATEGORICAL_FEATURES:
        X_future_row[col] = X_future_row[col].astype('category')
    
    # 모델 예측
    new_pred_value = lgbm.predict(X_future_row)[0]
    
    # ⭐️ [핵심 수정] 예측값 평활화 적용
    if i == 0:
        # 첫 번째 예측은 초기값(LAST_ACTUAL_VALUE)과 새로운 예측값의 가중평균
        smoothed_pred_value = (LAST_ACTUAL_VALUE * SMOOTHING_WEIGHT) + (new_pred_value * (1 - SMOOTHING_WEIGHT))
    else:
        # 이전 예측값과 새로운 예측값의 가중평균
        previous_pred_value = all_data.loc[future_index[i-1], TARGET]
        smoothed_pred_value = (previous_pred_value * SMOOTHING_WEIGHT) + (new_pred_value * (1 - SMOOTHING_WEIGHT))
        
    all_data.loc[current_dt, TARGET] = smoothed_pred_value

# 7. 시각화 데이터 병합 및 그래프 생성
# ----------------------------------------------------
final_future_predictions = all_data.loc[future_index, TARGET].reset_index().rename(columns={TARGET: 'Predicted_MaxWait'})

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
    title='✈️ Los Angeles 살고있는 의연이 분석',
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

fig.show()