import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import pandas as pd
import numpy as np
import glob

# 1. 합칠 파일 목록 가져오기 (예: 'data_2025_10_22.xls', 'data_2025_10_23.xls' 등)
# glob는 현재 폴더 내의 모든 .xls 파일을 찾아 리스트로 만듭니다.
file_list = glob.glob('test (*).xls')

# 2. 각 파일을 읽어 데이터프레임 리스트에 저장
all_data = []
for file in file_list:
    # 엑셀의 17행을 헤더로 사용하기 위해 16줄 건너뛰기
    df = pd.read_excel(file, skiprows=16) 
    
    # '계' 컬럼의 값만 추출
    data_values = df['계'].values.astype('float32')
    
    # NaN(결측치) 제거
    data_values = data_values[~np.isnan(data_values)]
    
    # 🌟🌟🌟 가장 중요: 각 파일의 맨 아래 '총합계' 행을 제거 🌟🌟🌟
    # 유효한 시간대 데이터 24개만 남기기 위해 마지막 행 제거
    if len(data_values) > 24:
        data_values = data_values[:-1]
    
    # 데이터프레임 형태로 다시 변환하여 리스트에 추가 (concat을 위해)
    all_data.append(pd.DataFrame(data_values, columns=['계']))

# 3. 모든 데이터프레임을 행(axis=0) 방향으로 합치기
combined_df = pd.concat(all_data, axis=0, ignore_index=True)

# 4. 합쳐진 데이터를 최종 'data' 변수에 할당
data = combined_df['계'].values.astype('float32')
data = data.reshape(-1, 1)

## ----------------------------------------------------
## 2. 데이터 전처리 및 정규화
## ----------------------------------------------------
# 데이터 스케일링: LSTM 성능 향상을 위해 0과 1 사이로 정규화
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# 학습 데이터와 테스트 데이터 분리
train_size = int(len(scaled_data) * 0.5)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

# 시퀀스 데이터셋 생성 함수
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        # i 시점부터 look_back 길이의 시퀀스를 입력(X)으로 사용
        a = dataset[i:(i + look_back), 0]
        X.append(a)
        # i + look_back 시점의 값을 출력(Y)으로 사용 (다음 시점 예측)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

# look_back(과거 몇 시점을 볼지) 설정
look_back = 3 
X_train, Y_train = create_dataset(train_data, look_back)
X_test, Y_test = create_dataset(test_data, look_back)

# LSTM 입력 형태에 맞게 데이터 차원 변환 (Samples, Timesteps, Features)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

## ----------------------------------------------------
## 3. LSTM 모델 구축 및 학습
## ----------------------------------------------------
model = Sequential()
# LSTM 레이어 추가 (50은 LSTM 유닛의 개수)
model.add(LSTM(50, input_shape=(look_back, 1)))
# 출력 레이어 (회귀 예측이므로 1개의 뉴런)
model.add(Dense(1))

# 모델 컴파일 (최적화 함수: adam, 손실 함수: MSE)
model.compile(optimizer='adam', loss='mean_squared_error')

# 모델 학습
print("모델 학습 시작...")
model.fit(X_train, Y_train, epochs=100, batch_size=1, verbose=0)
print("모델 학습 완료!")

## ----------------------------------------------------
## 4. 예측 및 결과 시각화
## ----------------------------------------------------
# 예측 수행
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# 정규화된 값을 원래 스케일로 되돌리기 (Inverse Transform)
train_predict = scaler.inverse_transform(train_predict)
Y_train_original = scaler.inverse_transform(Y_train.reshape(-1, 1))
test_predict = scaler.inverse_transform(test_predict)
Y_test_original = scaler.inverse_transform(Y_test.reshape(-1, 1))
data_original = scaler.inverse_transform(scaled_data)

# 예측 결과 시각화
plt.figure(figsize=(12, 6))
plt.plot(data_original, label='Original Data (True Congestion)')

# 학습 예측 결과 플롯 (look_back만큼 밀려서 시작)
train_plot = np.empty_like(data_original)
train_plot[:, :] = np.nan
train_plot[look_back:len(train_predict) + look_back, :] = train_predict
plt.plot(train_plot, label='Train Prediction')

# 테스트 예측 결과 플롯
test_plot = np.empty_like(data_original)
test_plot[:, :] = np.nan
# **수정된 부분:** 테스트 예측 시작 위치를 '학습 데이터 길이 + look_back'으로 정확히 맞춥니다.
# look_back을 3으로 설정하셨다고 가정합니다.
# 학습 데이터가 끝나는 시점은 len(train_predict) + look_back 입니다.
# 테스트 데이터는 이 시점부터 시작합니다.
test_start_index = len(train_predict) + look_back

# 테스트 예측값을 올바른 위치에 삽입
test_plot[test_start_index:test_start_index + len(test_predict), :] = test_predict

plt.plot(test_plot, label='Test Prediction')

plt.title('LSTM Congestion Prediction Example')
plt.xlabel('Time Step')
plt.ylabel('Congestion Level')
plt.legend()
plt.show()

# 예측 정확도 평가 (RMSE)
from sklearn.metrics import mean_squared_error
train_rmse = np.sqrt(mean_squared_error(Y_train_original, train_predict))
test_rmse = np.sqrt(mean_squared_error(Y_test_original, test_predict))
print(f"Train RMSE: {train_rmse:.4f}")
print(f"Test RMSE: {test_rmse:.4f}")