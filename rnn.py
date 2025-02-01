# rnn.py
# 패키지 필요 시 추가 & 공유하기
from torch.utils.data import TensorDataset, DataLoader
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# *************** 데이터 불러오기 - 규태 ***************
train_x_file = 'C:/pdpd/data/train_x_bit_QPSK_L3_dec_fin_save.csv'
test_x_file = 'C:/pdpd/data/test_x_bit_QPSK_L3_dec_fin_save.csv'
train_x_r1_file = 'C:/pdpd/data/train_x_bit_QPSK_L3_2X2_save.csv'
test_x_r1_file = 'C:/pdpd/data/test_x_bit_QPSK_L3_2X2_save.csv'

train_Y_file = 'C:/pdpd/data/train_Y_QPSK_L3_ReIm_2X2_save.csv'
test_Y_file = 'C:/pdpd/data/test_Y_QPSK_L3_ReIm_2X2_save.csv'

snr_list = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]
snr_to_col = {snr: i for i, snr in enumerate(snr_list)}

# *************** 데이터 전처리 - 수현 ****************


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
