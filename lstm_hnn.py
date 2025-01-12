import os
import argparse
import numpy as np
import torch
import torch.nn as nn
# import matplotlib.pyplot as plt


class LSTMHNN(nn.Module):
    """
    LSTM-based Hamiltonian Neural Network:
      - LSTMCell processes each (q, p) -> hidden state
      - readout -> scalar H
      - We get dq/dt, dp/dt from partial derivatives of H wrt (q,p).
    """

    def __init__(self, input_dim=2, hidden_dim=32):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm_cell = nn.LSTMCell(input_dim, hidden_dim)
        self.readout = nn.Linear(hidden_dim, 1)

    def forward(self, z, h_prev, c_prev):
        """
        Single step forward:
          z: shape [batch_size, 2]  (q,p)
          h_prev, c_prev: LSTM hidden states
        Returns:
          H: shape [batch_size, 1]
          h, c: updated hidden states
        """
        h, c = self.lstm_cell(z, (h_prev, c_prev))
        H = self.readout(h)  # [batch_size,1]
        return H, h, c

    def init_hidden(self, batch_size=1, device='cpu'):
        h0 = torch.zeros(batch_size, self.hidden_dim, device=device)
        c0 = torch.zeros(batch_size, self.hidden_dim, device=device)
        return h0, c0