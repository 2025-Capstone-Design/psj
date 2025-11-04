import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
from lightgbm.callback import early_stopping

# ----------------------------------------------------
# 폰트 깨짐 해결을 위한 설정 (⚠️ 윈도우 기본 폰트인 맑은 고딕 사용)
# ----------------------------------------------------
from matplotlib import font_manager, rc
try:
    # 맑은 고딕 (Malgun Gothic) 경로 설정
    font_path = 'C:/Windows/Fonts/malgun.ttf' 
    font_name = font_manager.FontProperties(fname=font_path).get_name()
    rc('font', family=font_name)
    # 마이너스 부호 깨짐 방지
    plt.rcParams['axes.unicode_minus'] = False
    print("Matplotlib 한글 폰트 설정 완료 (맑은 고딕).")
except:
    print("⚠️ 맑은 고딕 폰트를 찾을 수 없습니다. 기본 폰트로 실행됩니다. 한글이 깨질 수 있습니다.")
    pass


# ----------------------------------------------------
# 1. 데이터 파일 불러오기 및 결합
# ----------------------------------------------------
print("\n1. 데이터 파일 불러오기 및 결합 시작...")
파일_목록 = glob.glob('test (*).xls')

모든_데이터 = []
for 파일 in 파일_목록:
    # 엑셀의 17행을 헤더로 사용하기 위해 16줄 건너뛰기
    df_임시 = pd.read_excel(파일, skiprows=16)
    혼잡도_값 = df_임시['계'].values.astype('float32')
    혼잡도_값 = 혼잡도_값[~np.isnan(혼잡도_값)]

    # 총합계 행 제거
    if len(혼잡도_값) > 24:
        혼잡도_값 = 혼잡도_값[:-1]

    모든_데이터.append(pd.DataFrame(혼잡도_값, columns=['계']))

df_결합 = pd.concat(모든_데이터, axis=0, ignore_index=True)
실제_혼잡도_원본 = df_결합['계'].values.astype('float32') 
print(f"총 데이터 포인트 수: {len(실제_혼잡도_원본)}")


## ----------------------------------------------------
## 2. 특징 공학 (Feature Engineering) 및 데이터 준비
## ----------------------------------------------------
print("\n2. 특징 공학 및 데이터 준비 시작...")
df_특징 = df_결합.copy()
과거_시점_수 = 3 # 과거 3시점의 혼잡도 값을 피처로 사용

# 2-1. 시간 및 요일 특징 생성
df_특징['시간_Hour'] = df_특징.index % 24
df_특징['요일_DayOfWeek'] = (df_특징.index // 24) % 7

# 2-2. 지연값 (Lag, 과거 시점) 특징 생성
for i in range(1, 과거_시점_수 + 1):
    df_특징[f'과거_혼잡도_{i}'] = df_특징['계'].shift(i)

# Lag 특징 생성으로 인해 발생하는 상위 NaN 행 제거
df_특징.dropna(inplace=True)

# 2-3. X(입력 특징)와 Y(정답 타겟) 정의
특징_컬럼 = [f'과거_혼잡도_{i}' for i in range(1, 과거_시점_수 + 1)] + ['시간_Hour', '요일_DayOfWeek']
X_입력 = df_특징[특징_컬럼].values
Y_정답 = df_특징['계'].values

# 2-4. 학습 데이터와 테스트 데이터 분리 (50:50)
데이터_길이 = len(df_특징)
학습_크기 = int(데이터_길이 * 0.5)
X_학습 = X_입력[:학습_크기]
Y_학습 = Y_정답[:학습_크기]
X_테스트 = X_입력[학습_크기:]
Y_테스트 = Y_정답[학습_크기:]

print(f"X_학습 데이터 형태: {X_학습.shape}")


## ----------------------------------------------------
## 3. LightGBM 모델 구축 및 학습
## ----------------------------------------------------
print("\n3. LightGBM 모델 학습 시작...")
모델_lgbm = LGBMRegressor(
    objective='regression',
    metric='rmse',
    n_estimators=1000,
    learning_rate=0.05,
    min_child_samples=10, 
    random_state=42,
    n_jobs=-1
)

# 학습 시작
모델_lgbm.fit(
    X_학습,
    Y_학습,
    eval_set=[(X_테스트, Y_테스트)],
    eval_metric='rmse',
    callbacks=[
        early_stopping(stopping_rounds=100, verbose=False)
    ]
)
print("LightGBM 모델 학습 완료!")


## ----------------------------------------------------
## 4. 예측 및 결과 시각화
## ----------------------------------------------------
print("\n4. 예측 및 결과 시각화 시작...")
# 예측 수행 
학습_예측값 = 모델_lgbm.predict(X_학습).reshape(-1, 1)
테스트_예측값 = 모델_lgbm.predict(X_테스트).reshape(-1, 1)

# 시각화를 위한 플롯 데이터 준비 (원본 인덱스에 맞추기)
학습_플롯 = np.empty_like(실제_혼잡도_원본.reshape(-1, 1))
학습_플롯[:, :] = np.nan
학습_플롯[과거_시점_수 : len(학습_예측값) + 과거_시점_수, 0] = 학습_예측값.flatten()

테스트_플롯 = np.empty_like(실제_혼잡도_원본.reshape(-1, 1))
테스트_플롯[:, :] = np.nan
테스트_시작_인덱스 = len(학습_예측값) + 과거_시점_수
테스트_플롯[테스트_시작_인덱스 : 테스트_시작_인덱스 + len(테스트_예측값), 0] = 테스트_예측값.flatten()

# 예측 결과 시각화
plt.figure(figsize=(12, 6))
# 🟢 초록색: 실제 혼잡도 원본
plt.plot(실제_혼잡도_원본, label='실제 혼잡도 데이터') 
# 🔵 파란색: 학습 구간 예측 결과
plt.plot(학습_플롯, label='학습 데이터 예측 결과')
# 🟠 주황색: 테스트 구간 예측 결과
plt.plot(테스트_플롯, label='테스트 데이터 예측 결과') 

plt.title('LightGBM 혼잡도 예측 결과')
plt.xlabel('시간 스텝')
plt.ylabel('혼잡도 수준')
plt.legend()
plt.show()


# 예측 정확도 평가 (RMSE)
rmse_학습 = np.sqrt(mean_squared_error(Y_학습, 학습_예측값))
rmse_테스트 = np.sqrt(mean_squared_error(Y_테스트, 테스트_예측값))
print(f"\n--- 예측 정확도 평가 (RMSE) ---")
print(f"학습 데이터 RMSE: {rmse_학습:.4f}")
print(f"테스트 데이터 RMSE: {rmse_테스트:.4f}")