import matplotlib.pyplot as plt
import pandas as pd
from numpy import array
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv1D, MaxPooling1D

# 데이터 불러오기
df = pd.read_csv('202309.csv', sep=',')
# print(df.head())[:10]
#    1500000100  1500000200  1500000506  ...  1550383400  1550383500  1550383600
# 0        53.0        26.0        18.0  ...        21.0        18.0        26.0
# 1        31.0        28.0        19.0  ...        21.0        18.0        26.0
# 2        21.0        31.0        28.0  ...        21.0        18.0        26.0
# 3        24.0        38.0        21.0  ...        21.0        18.0        26.0
# 4        26.0        40.0        36.0  ...        21.0        18.0        26.0
timeseries = df.iloc[:, 0].values.astype('float32')[:500]


# split a uni-variate sequence into samples
def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence) - 1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


# define input sequence
row_seq = timeseries
n_steps = 3
X, y = split_sequence(row_seq, n_steps)

# for i in range(len(X)):
#     print(X[i], y[i])

# define model
n_features = 1
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps, n_features)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# fit model
model.fit(X, y, epochs=100, verbose=0)

# demonstrate prediction and plot
x_input = X[-1]
x_input = x_input.reshape((1, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)
print("Predictive Value : ", yhat)

# 모든 데이터 포인트를 포함하여 전체 시계열 데이터를 예측한다.
predicted_values = []
for i in range(len(X)):
    x_input = X[i].reshape((1, n_steps, n_features))
    yhat = model.predict(x_input, verbose=0)
    predicted_values.append(yhat[0][0])

# 실제 값과 예측 값 그래프로 그리기
plt.plot(y, label='Real value')
plt.plot(predicted_values, label='Predictive value', linestyle='dashed')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()
