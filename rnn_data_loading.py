from torch.utils.data import TensorDataset, DataLoader
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# 데이터 불러오기 - 규태
train_x_file = 'C:/pdpd/data/train_x_bit_QPSK_L3_dec_fin_save.csv'
test_x_file  = 'C:/pdpd/data/test_x_bit_QPSK_L3_dec_fin_save.csv'
train_x_r1_file = 'C:/pdpd/data/train_x_bit_QPSK_L3_2X2_save.csv'
test_x_r1_file = 'C:/pdpd/data/test_x_bit_QPSK_L3_2X2_save.csv'

train_Y_file  = 'C:/pdpd/data/train_Y_QPSK_L3_ReIm_2X2_save.csv'
test_Y_file  = 'C:/pdpd/data/test_Y_QPSK_L3_ReIm_2X2_save.csv'

snr_list = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]
snr_to_col = {snr: i for i, snr in enumerate(snr_list)}

# 데이터 전처리 - 수현

# 모델 정의 - 창인

# 학습 - 채원

# 그래프 출력 & 성능 평가 - 민지