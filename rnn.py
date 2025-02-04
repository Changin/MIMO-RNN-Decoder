# rnn.py
from torch.utils.data import TensorDataset, DataLoader
import torch
import pandas as pd
import matplotlib.pyplot as plt
import torch.optim as optim
import numpy as np

# *************** 데이터 불러오기 - 규태 ***************
'''
train_x_file = pd.read_csv('train_x_bit_QPSK_L3_dec_fin_save.csv').to_numpy()
test_x_file  = pd.read_csv('test_x_bit_QPSK_L3_dec_fin_save.csv').to_numpy()
train_x_r1_file = pd.read_csv('train_x_bit_QPSK_L3_2X2_save.csv').to_numpy()
test_x_r1_file = pd.read_csv('test_x_bit_QPSK_L3_2X2_save.csv').to_numpy()

train_Y_file  = pd.read_csv('train_Y_QPSK_L3_ReIm_2X2_save.csv').to_numpy()
test_Y_file  = pd.read_csv('test_Y_QPSK_L3_ReIm_2X2_save.csv').to_numpy()
'''

# *************** 데이터 전처리 - 수현 ****************
train_ReIm = pd.read_csv("train_Y_QPSK_L3_ReIm_2X2_save.csv", header=None ).to_numpy()
test_ReIm = pd.read_csv("test_Y_QPSK_L3_ReIm_2X2_save.csv", header=None ).to_numpy()

train_qpsk = pd.read_csv("train_x_bit_QPSK_L3_dec_fin_save.csv", header=None ).to_numpy()
test_qpsk = pd.read_csv("test_x_bit_QPSK_L3_dec_fin_save.csv", header=None ).to_numpy()

train_bit = pd.read_csv("train_x_bit_QPSK_L3_2X2_save.csv", header=None ).to_numpy()
test_bit = pd.read_csv("test_x_bit_QPSK_L3_2X2_save.csv", header=None ).to_numpy()
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
y_train = torch.Tensor(train_qpsk).long().flatten()

# bit
bit_train = []

for i in range(train_bit.shape[0]):
    for j in range(16):
        bit_train.append(train_bit[i : i + 1, j * 4 : j * 4 + 4].reshape(1, -1)[0])

bit_train = torch.Tensor(bit_train).long()

# Dataset 및 DataLoader 생성 (전체 DataLoader는 SNR별 슬라이싱을 위해 사용하지 않음)
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
y_test = torch.Tensor(test_qpsk).long().flatten()

# bit
bit_test = []

for i in range(test_bit.shape[0]):
    for j in range(16):
        bit_test.append(test_bit[i : i + 1, j * 4 : j * 4 + 4].reshape(1, -1)[0])

bit_test = torch.Tensor(bit_test).long()

# Dataset 및 DataLoader 생성 (전체 DataLoader는 SNR별 슬라이싱을 위해 사용하지 않음)
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
# input_size = 1
input_size = 4
num_layers = 2          # 학습 성능 개선 위해 1 -> 2로 수정
hidden_size = 64        # 학습 성능 개선 위해 8 -> 64로 수정
# sequence_length = 12
sequence_length = 3


class LSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, sequence_length, num_layers, device):
        super(LSTM, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = torch.nn.Linear(hidden_size * sequence_length, 256)
        self.fc2 = torch.nn.Linear(256, 128)
        self.fc3 = torch.nn.Linear(128, 64)
        self.fc4 = torch.nn.Linear(64,32)
        self.fc = torch.nn.Linear(32, 16) # 16개 (0~15 심볼) 최종출력 반환
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        # x = x.unsqueeze(-1)  # 학습 시에 차원 문제로 수정(unsqueeze로 차원 추가)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out.reshape(out.shape[0], -1)
        out = self.relu(self.fc1(out))
        out = self.relu(self.fc2(out))
        out = self.relu(self.fc3(out))
        out = self.relu(self.fc4(out))
        out = self.fc(out)
        return out

# ************** 학습 - 채원 ***************

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_snr_subset(x, y, bit, snr_value):
    offset = int(snr_value / 2)  # 예: SNR=0dB -> offset 0, SNR=2dB -> offset 1, 등등
    indices = list(range(offset, x.shape[0], 16))
    return x[indices], y[indices], bit[indices]

def get_snr_dataloader(x, y, bit, snr_value, batch_size, shuffle=False):
    x_subset, y_subset, bit_subset = get_snr_subset(x, y, bit, snr_value)
    dataset = TensorDataset(x_subset, y_subset, bit_subset)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader

def symbol_to_bits(symbol, bit_length=4):
    return [int(x) for x in format(symbol, f'0{bit_length}b')]

def calculate_bit_error_rate(predictions, ground_truth, bit_length=4):
    pred_symbols = predictions.cpu().numpy()
    true_symbols = ground_truth.cpu().numpy()
    total_bit_errors = 0
    total_bits = len(pred_symbols) * bit_length
    for pred_sym, true_sym in zip(pred_symbols, true_symbols):
        pred_bits = symbol_to_bits(pred_sym, bit_length)
        true_bits = symbol_to_bits(true_sym, bit_length)
        total_bit_errors += sum(p != t for p, t in zip(pred_bits, true_bits))
    return total_bit_errors / total_bits

# SNR별 학습/테스트
dB_snr = [0, 2, 4, 6, 8, 10, 12, 14, 16]
num_epochs = 10
learning_rate = 0.001
criterion = torch.nn.CrossEntropyLoss()

models = {}

print("========== SNR별 학습 시작 ==========")
for snr_value in dB_snr:
    print(f"\n--- SNR = {snr_value} dB ---")
    train_loader_snr = get_snr_dataloader(x_train, y_train, bit_train, snr_value, BATSIZE, shuffle=True)

    model_snr = LSTM(input_size, hidden_size, sequence_length, num_layers, device).to(device)
    optimizer_snr = torch.optim.Adam(model_snr.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model_snr.train()
        total_loss = 0.0
        for batch_idx, (x_batch, y_batch, bit_batch) in enumerate(train_loader_snr):
            x_batch, y_batch, bit_batch = x_batch.to(device), y_batch.to(device), bit_batch.to(device)
            optimizer_snr.zero_grad()
            output = model_snr(x_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer_snr.step()
            total_loss += loss.item()
        print(f"SNR {snr_value} dB, Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader_snr):.4f}")
    models[snr_value] = model_snr

print("\n========== SNR별 테스트 시작 ==========")
avg_ber_list=[]  # 그래프 출력을 위해 ber 값을 저장 하기 위해 추가

for snr_value in dB_snr:
    test_loader_snr = get_snr_dataloader(x_test, y_test, bit_test, snr_value, BATSIZE, shuffle=False)
    model_snr = models[snr_value]
    model_snr.eval()
    snr_ber_list = []
    with torch.no_grad():
        for batch_idx, (x_batch, y_batch, bit_batch) in enumerate(test_loader_snr):
            x_batch, y_batch, bit_batch = x_batch.to(device), y_batch.to(device), bit_batch.to(device)
            output = model_snr(x_batch)
            predictions = torch.argmax(output, dim=1)
            ber = calculate_bit_error_rate(predictions, y_batch, bit_length=4)  # BER 계산
            snr_ber_list.append(ber)
    avg_ber = sum(snr_ber_list) / len(snr_ber_list)
    avg_ber_list.append(avg_ber)    # 그래프 출력을 위해 ber 값을 저장 하기 위해 추가
    print(f"Test SNR {snr_value} dB, Average BER: {avg_ber:.6f}")

# ************* 그래프 출력 & 성능 평가 - 민지 *************

# ber 그래프
plt.semilogy(dB_snr, avg_ber_list, marker='o', color='orange', label='Simulated')
plt.xticks(dB_snr)
plt.title('Bit Error Rate Performance Over Varying SNR')
plt.xlabel('SNR (dB)')
plt.ylabel('Bit Error Rate (BER)')
plt.grid(True, which="both", linestyle='--', linewidth=0.5)
plt.legend()
plt.show()