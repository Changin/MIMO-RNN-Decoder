# rnn.py
# 패키지 필요 시 추가 & 공유하기
from torch.utils.data import TensorDataset, DataLoader
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# *************** 데이터 불러오기 - 규태 ***************


# *************** 데이터 전처리 - 수현 ****************


# **************** 모델 정의 - 창인 ******************
# 하이퍼 파라미터 정의
# input_size = x_train.size(1)
input_size = 1      # 한 타임에 들어가는 입력의 갯수(변수) (1개) -> 나중에 4로 변경 필요 (x1 실수/허수, x2 실수/허수)
num_layers = 1      # LSTM 스택 갯수 (1)
hidden_size = 8     # hidden state h의 차원 수
sequence_length = 12    # 시퀀스 길이 (총 12개 (3개의 시간))   -> 나중에 3으로 변경 필요 (3개의 시간)
# 즉 수정 후에는 [배치크기, sequence_length, input_size] 크기의 텐서를 입력해야 함


class LSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, sequence_length, num_layers, device):
        super(LSTM, self).__init__()
        self.device = device  # 학습할 때 GPU 활용하기 위해 device 넣어주세요
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers)
        self.fc = torch.nn.Linear(hidden_size * sequence_length, 16)  # 16개 (0~15 심볼) 최종출력 반환

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size()[0], self.hidden_size).to(self.device)  # 초기 hidden_state 설정
        c0 = torch.zeros(self.num_layers, x.size()[0], self.hidden_size).to(self.device)  # 기억셀 cell state 초기화
        out, _ = self.lstm(x, (h0, c0))
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        return out


# ************** 학습 - 채원 ***************


# ************* 그래프 출력 & 성능 평가 - 민지 *************
