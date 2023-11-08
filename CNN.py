import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd

# 데이터 불러오기
df = pd.read_csv('202309.csv', sep=',')
timeseries = df.iloc[:, 0].values.astype('float32')[:500]
#600 -> 16%, 500->8%
# 시계열 데이터를 위한 윈도우 생성 함수
def create_sequences(data, lookback):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:i + lookback])
        y.append(data[i + lookback])
    return torch.tensor(X), torch.tensor(y)

# 데이터 전처리
lookback = 10
X, y = create_sequences(timeseries, lookback)
X = X.unsqueeze(1)  # 3D 데이터로 변환 (배치 크기, 피처 수, 시퀀스 길이)
y = y.unsqueeze(1)  # 2D 데이터로 변환 (배치 크기, 피처 수)

# 데이터를 학습 세트와 테스트 세트로 나누기
train_size = int(0.8 * len(X))
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

# 1D CNN 모델 정의
class TrafficCNN(nn.Module):
    def __init__(self):
        super(TrafficCNN, self).__init__()
        self.conv1d = nn.Conv1d(1, 16, kernel_size=3)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(16 * (lookback - 2), 1)

    def forward(self, x):
        x = self.conv1d(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)  # Flatten the output
        x = self.fc(x)
        return x

# 모델 초기화
model = TrafficCNN()

# 모델 설정
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 데이터로더 설정
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 모델 학습
num_epochs = 150
for epoch in range(num_epochs):
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

# 모델 평가
model.eval()
with torch.no_grad():
    test_inputs = X_test
    predicted = model(test_inputs)
    test_loss = criterion(predicted, y_test)
    print(f'Test Loss: {test_loss.item()}')

# 예측 결과 및 실제 결과 그래프로 나타내기
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.plot(y_test, label='True')
plt.plot(predicted, label='Predicted', linestyle='--')
plt.title('Traffic Speed Prediction')
plt.xlabel('Time Step')
plt.ylabel('Speed')
plt.legend()
plt.show()
