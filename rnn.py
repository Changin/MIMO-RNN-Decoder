# rnn.py
# 패키지 필요 시 추가 & 공유하기
from torch.utils.data import TensorDataset, DataLoader
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# *************** 데이터 불러오기 - 규태 ***************


# *************** 데이터 전처리 - 수현 ****************
"""
train_ReIm = pd.read_csv("train_Y_QPSK_L3_ReIm_2X2_save.csv", header=None ).to_numpy()
test_ReIm = pd.read_csv("test_Y_QPSK_L3_ReIm_2X2_save.csv", header=None ).to_numpy()

train_qpsk = pd.read_csv("train_x_bit_QPSK_L3_dec_fin_save.csv", header=None ).to_numpy()
test_qpsk = pd.read_csv("test_x_bit_QPSK_L3_dec_fin_save.csv", header=None ).to_numpy()

train_bit = pd.read_csv("train_x_bit_QPSK_L3_2X2_save.csv", header=None ).to_numpy()
test_bit = pd.read_csv("test_x_bit_QPSK_L3_2X2_save.csv", header=None ).to_numpy()
"""
BATSIZE = 800

# 학습 데이터 변환

# x
x_train = []

for i in range(train_ReIm.shape[0] - 2):
    for j in range(16):
        x_train.append(train_ReIm[i : i + 3, j * 4 : j * 4 + 4].reshape(1, -1)[0])

x_train = torch.Tensor(x_train).float()

# 추가된부분
x_train = x_train.reshape(-1, 3, 4)

# y
y_train = torch.Tensor(train_qpsk).long()
y_train = y_train.flatten()

# bit
bit_train = []

for i in range(train_bit.shape[0]):
    for j in range(16):
        bit_train.append(train_bit[i : i + 1, j * 4 : j * 4 + 4].reshape(1, -1)[0])

bit_train = torch.Tensor(bit_train).long()

# dataset / dataloader
train_dataset = TensorDataset(x_train, y_train, bit_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATSIZE, shuffle=True)

# 테스트 데이터 변환

# x
x_test = []

for i in range(test_ReIm.shape[0] - 2):
    for j in range(16):
        x_test.append(test_ReIm[i : i + 3, j * 4 : j * 4 + 4].reshape(1, -1)[0])

x_test = torch.Tensor(x_test).float()

# 추가된부분
x_test = x_test.reshape(-1, 3, 4)

# y
y_test = torch.Tensor(test_qpsk).long()
y_test = y_test.flatten()

# bit
bit_test = []

for i in range(test_bit.shape[0]):
    for j in range(16):
        bit_test.append(test_bit[i : i + 1, j * 4 : j * 4 + 4].reshape(1, -1)[0])

bit_test = torch.Tensor(bit_test).long()

# dataset / dataloader
test_dataset = TensorDataset(x_test, y_test, bit_test)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATSIZE, shuffle=False)

"""
출력 예 )
x_train[:3]
tensor([[[-0.5619, -0.7938, -0.6537,  0.5367],
         [ 0.0937, -0.2559, -0.3049,  0.1836],
         [-0.8894, -0.4642,  0.5541, -0.0245]],

        [[ 0.6489,  0.5219,  0.2804, -0.3910],
         [-0.7107, -0.7688,  0.3040, -0.3418],
         [-0.7412, -0.9563,  0.1895,  0.6651]],

        [[-0.0275,  0.0300,  0.1243, -1.1779],
         [-0.1724,  0.7663, -0.5029, -0.9555],
         [ 0.9054,  0.3648,  1.1456, -0.6589]]])
x_train[16]
tensor([[ 0.0937, -0.2559, -0.3049,  0.1836],
        [-0.8894, -0.4642,  0.5541, -0.0245],
        [-2.0945,  0.9280, -0.6222, -0.7343]])
torch.Size([16000000, 3, 4])

y_train[:3]
tensor([3, 4, 6])
torch.Size([16000000])

bit_train[:3]
tensor([[1, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 1, 0]])
torch.Size([16000000, 4])

학습/테스트하면서 x, y, bit 불러올때 인덱스 저렇게 주면 snr별로 학습/테스트 할수 있어요
dB_snr = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]

for snr_value in dB_snr:
    for batch_idx, (x,y,bit) in enumerate(test_loader):
        x, y, bit = x[int(snr_value/2)::16].to(device), y[int(snr_value/2)::16].to(device), bit[int(snr_value/2)::16].to(device)
"""

# **************** 모델 정의 - 창인 ******************
# 하이퍼 파라미터 정의
input_size = x_seq.size(2)
num_layers = 1
hidden_size = 8


class LSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, sequence_length, num_layers, device):
        super(LSTM, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size * sequence_length, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size()[0], self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size()[0], self.hidden_size).to(self.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        return out


# ************** 학습 - 채원 ***************


# ************* 그래프 출력 & 성능 평가 - 민지 *************
