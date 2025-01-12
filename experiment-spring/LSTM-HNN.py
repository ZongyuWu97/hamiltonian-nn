#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

from lstm_hnn import LSTMHNN

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

###############################################################################
# 1) Command-Line Arguments
###############################################################################
def get_args():
    parser = argparse.ArgumentParser("LSTM-HNN training with new random trajectory each epoch")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--save_dir", type=str, default=THIS_DIR, help="Where to save models")
    parser.add_argument("--name", type=str, default="lstm_hnn", help="Base name for saved files")

    # Model hyperparams
    parser.add_argument("--input_dim", type=int, default=2, help="Dimension of (q, p)")
    parser.add_argument("--hidden_dim", type=int, default=32, help="Hidden dimension of LSTMCell")

    # Training settings
    parser.add_argument("--epochs", type=int, default=500, help="Number of epochs")
    parser.add_argument("--learn_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--print_every", type=int, default=5, help="Print interval")
    parser.add_argument("--verbose", action="store_true", help="Print training progress or not")

    args = parser.parse_args([])
    return args


###############################################################################
# 2) Data Generation
###############################################################################
def damped_harmonic_oscillator(q0=1.0, p0=0.0, m=1.0, k=1.0, gamma=0.1,
                               t_max=5.0, dt=0.01, noise_std=0.0):
    """
    Simple Euler integration of a damped harmonic oscillator:
      dq/dt = p/m
      dp/dt = -k*q - gamma*p
    Returns arrays: times (ts), qs, ps
    """
    ts = np.arange(0, t_max, dt)
    N = len(ts)

    qs = np.zeros(N)
    ps = np.zeros(N)

    q, p = q0, p0
    for i in range(N):
        qs[i] = q
        ps[i] = p
        dq = p / m
        dp = -k * q - gamma * p
        q += dq * dt
        p += dp * dt

    # optional noise
    qs += noise_std * np.random.randn(N)
    ps += noise_std * np.random.randn(N)

    return ts, qs, ps


def get_random_trajectory(m=1.0, k=1.0, gamma=0.1, noise_std=0.0,
                          t_max=5.0, dt=0.01):
    """
    Generate a single random damped oscillator trajectory with random initial (q0, p0).
    Returns (x, dx):
      x: shape [N, 2], containing (q, p)
      dx: shape [N, 2], containing (dq, dp)
    """
    # Random initial conditions in some range, e.g. [-2, 2]
    q0 = np.random.rand() * 4 - 2.0
    p0 = np.random.rand() * 4 - 2.0

    ts, qs, ps = damped_harmonic_oscillator(q0=q0, p0=p0, m=m, k=k, gamma=gamma,
                                            t_max=t_max, dt=dt, noise_std=noise_std)

    # compute derivatives
    dq = ps / m
    dp = -k * qs - gamma * ps

    x = np.stack([qs, ps], axis=1)  # [N,2]
    dx = np.stack([dq, dp], axis=1)  # [N,2]
    return x, dx


###############################################################################
# 3) LSTM-HNN Model
###############################################################################
# class LSTMHNN(nn.Module):
#     """
#     LSTM-based Hamiltonian Neural Network:
#       - LSTMCell processes each (q, p) -> hidden state
#       - readout -> scalar H
#       - We get dq/dt, dp/dt from partial derivatives of H wrt (q,p).
#     """

#     def __init__(self, input_dim=2, hidden_dim=32):
#         super().__init__()
#         self.hidden_dim = hidden_dim
#         self.lstm_cell = nn.LSTMCell(input_dim, hidden_dim)
#         self.readout = nn.Linear(hidden_dim, 1)

#     def forward(self, z, h_prev, c_prev):
#         """
#         Single step forward:
#           z: shape [batch_size, 2]  (q,p)
#           h_prev, c_prev: LSTM hidden states
#         Returns:
#           H: shape [batch_size, 1]
#           h, c: updated hidden states
#         """
#         h, c = self.lstm_cell(z, (h_prev, c_prev))
#         H = self.readout(h)  # [batch_size,1]
#         return H, h, c

#     def init_hidden(self, batch_size=1, device='cpu'):
#         h0 = torch.zeros(batch_size, self.hidden_dim, device=device)
#         c0 = torch.zeros(batch_size, self.hidden_dim, device=device)
#         return h0, c0


def lstm_hnn_time_derivative(model, z, h_prev, c_prev):
    """
    Given z=[q,p], do:
      1) forward pass -> H
      2) partial derivatives wrt z -> dq_dt, dp_dt
      3) return updated hidden states
    """
    H, h, c = model(z, h_prev, c_prev)  # H shape [B,1]

    grad = torch.autograd.grad(
        outputs=H.sum(),
        inputs=z,
        create_graph=True
    )[0]  # shape [B,2]
    dH_dq = grad[:, 0]
    dH_dp = grad[:, 1]

    dq_dt = dH_dp.unsqueeze(-1)  # [B,1]
    dp_dt = -dH_dq.unsqueeze(-1)  # [B,1]

    return dq_dt, dp_dt, h, c, H


###############################################################################
# 4) Train LSTM-HNN on new random trajectory each epoch
###############################################################################
def train_lstm_hnn(args):
    """
    We train an LSTM-HNN for 'args.epochs' epochs.
    Each epoch:
      1) Generate a new random trajectory (train).
      2) Run one pass to compute loss & do backprop.
    We keep a separate fixed test trajectory (q0=1.0, p0=0.0) to measure test loss.
    Returns the trained model and training stats.
    """
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Prepare model & optimizer
    model = LSTMHNN(input_dim=args.input_dim, hidden_dim=args.hidden_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learn_rate)

    # Prepare a fixed test trajectory for consistent evaluation
    # e.g. q0=1, p0=0
    x_test_np, dx_test_np = get_random_trajectory(
        m=1.0, k=1.0, gamma=0.1, noise_std=0.1,
        t_max=5.0, dt=0.01
    )  # This call uses random q0/p0, but you can fix if you prefer
    # If you want a truly fixed test, define q0=1.0, p0=0.0 manually or remove noise.
    # For example:
    # ts_test, qs_test, ps_test = damped_harmonic_oscillator(q0=1.0, p0=0.0, noise_std=0.1)
    # Then compute dx, etc.

    x_test = torch.tensor(x_test_np, dtype=torch.float32, device=device)
    dx_test = torch.tensor(dx_test_np, dtype=torch.float32, device=device)

    # We'll store losses each epoch
    stats = {"train_loss": [], "test_loss": []}

    for epoch in range(1, args.epochs + 1):
        # ------------------------------------------------------------------
        # 1) Generate a new random training trajectory for this epoch
        # ------------------------------------------------------------------
        x_np, dx_np = get_random_trajectory(
            m=1.0, k=1.0, gamma=0.1, noise_std=0.1,
            t_max=5.0, dt=0.01
        )
        x_train = torch.tensor(x_np, dtype=torch.float32, device=device)
        dx_train = torch.tensor(dx_np, dtype=torch.float32, device=device)

        # ------------------------------------------------------------------
        # 2) Forward/backward on the newly generated training data
        # ------------------------------------------------------------------
        optimizer.zero_grad()
        train_loss = 0.0

        # We'll do a simple "single-step" approach over each sample
        h, c = model.init_hidden(batch_size=1, device=device)
        for i in range(x_train.shape[0]):
            z_t = x_train[i].unsqueeze(0).clone().detach().requires_grad_(True)
            dq_pred, dp_pred, h, c, _ = lstm_hnn_time_derivative(model, z_t, h, c)

            dq_true = dx_train[i, 0]
            dp_true = dx_train[i, 1]
            loss_i = (dq_true - dq_pred.squeeze()) ** 2 + (dp_true - dp_pred.squeeze()) ** 2
            train_loss += loss_i

        train_loss = train_loss / x_train.shape[0]
        train_loss.backward()
        optimizer.step()

        # ------------------------------------------------------------------
        # 3) Evaluate on the fixed test trajectory
        # ------------------------------------------------------------------
        test_loss = 0.0
        h_test, c_test = model.init_hidden(batch_size=1, device=device)
        for j in range(x_test.shape[0]):
            z_tst = x_test[j].unsqueeze(0).clone().detach().requires_grad_(True)
            dq_pred_t, dp_pred_t, h_test, c_test, _ = lstm_hnn_time_derivative(model, z_tst, h_test, c_test)

            dq_t = dx_test[j, 0]
            dp_t = dx_test[j, 1]
            test_loss_j = (dq_t - dq_pred_t.squeeze()) ** 2 + (dp_t - dp_pred_t.squeeze()) ** 2
            test_loss += test_loss_j

        test_loss = test_loss / x_test.shape[0]

        # ------------------------------------------------------------------
        # 4) Log & print
        # ------------------------------------------------------------------
        stats["train_loss"].append(train_loss.item())
        stats["test_loss"].append(test_loss.item())

        if args.verbose and (epoch % args.print_every == 0):
            print(f"Epoch {epoch}/{args.epochs} | train_loss={train_loss.item():.4e}, test_loss={test_loss.item():.4e}")

    # final losses
    print(f"\nFinal train loss: {stats['train_loss'][-1]:.4e}, final test loss: {stats['test_loss'][-1]:.4e}")
    return model, stats


###############################################################################
# 5) Main
###############################################################################
if __name__ == "__main__":
    args = get_args()
    args.verbose = True
    args.print_every = 10

    # Train model
    model, stats = train_lstm_hnn(args)

    # Save model
    os.makedirs(args.save_dir) if not os.path.exists(args.save_dir) else None
    save_path = os.path.join(args.save_dir, f"{args.name}.tar")
    torch.save(model.state_dict(), save_path)

    # Print final result
    final_train_loss = stats["train_loss"][-1]
    final_test_loss = stats["test_loss"][-1]
    print(f"Saved model to {save_path}")
    print(f"Final train loss = {final_train_loss:.4e}, final test loss = {final_test_loss:.4e}")

    # Optional: plot training curves
    fig = plt.figure(figsize=(3, 3), facecolor='white', dpi=DPI)
    epochs = np.arange(1, args.epochs + 1)
    plt.plot(epochs, stats["train_loss"], label="Train")
    plt.plot(epochs, stats["test_loss"], label="Test")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("LSTM-HNN: Training Over Random Trajectories Each Epoch")
    plt.tight_layout()
    plt.show()
    fig.savefig('{}/spring_train_curve.png'.format(args.save_dir))
