# 1D Steady Heat Conduction using PINNs - with multiple activation functions
# -------------------------------------------------------------------------
# This script trains a Physics-Informed Neural Network (PINN) to solve the 1D steady
# heat conduction equation (Laplace's equation) T''(x) = 0 with Dirichlet boundary
# conditions T(0)=T_left and T(1)=T_right. The exact solution for this problem is
# a linear function T(x) = T_left + (T_right - T_left)*x (here we take T_left=0, T_right=1, so T(x)=x).
# We compare the performance of PINNs using five different activation functions:
#    tanh, ReLU, SiLU (Sigmoid Linear Unit), linear (identity), and sigmoid.
# Each network is trained separately with the specified activation function.
# After training, we plot the predictions of each network along with the exact solution
# and a reference finite difference (FD) solution for comparison.

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Ensure reproducibility
torch.manual_seed(0)
np.random.seed(0)


# Define a subclass of MLP that supports multiple activation functions
class SiLu(nn.Module):
    def __init__(self, num_layers: int, input_dim: int, hidden_dim: int, output_dim: int, activation='silu'):
        super(SiLu, self).__init__()
        self.layers = nn.ModuleList()
        self.set_activation(activation)

        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        if num_layers == 1:
            self.layers.append(nn.Linear(input_dim, output_dim))
        else:
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            for _ in range(num_layers - 2):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.Linear(hidden_dim, output_dim))

    def set_activation(self, activation):
        activation = activation.lower()
        if activation == 'tanh':
            self.activation_fn = torch.tanh
        elif activation == 'relu':
            self.activation_fn = torch.relu
        elif activation == 'silu':
            self.activation_fn = torch.nn.functional.silu
        elif activation == 'sigmoid':
            self.activation_fn = torch.sigmoid
        elif activation == 'linear':
            self.activation_fn = lambda x: x
        else:
            raise ValueError(f"Unsupported activation type: {activation}")
        self.activation = activation

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation_fn(layer(x))
        return self.layers[-1](x)


# Define training and plotting logic
if __name__ == "__main__":
    T_left = 0.0
    T_right = 1.0
    N_int = 50

    x_int = torch.linspace(0, 1, N_int + 2, dtype=torch.float32)[1:-1].unsqueeze(1)
    x_int.requires_grad_(True)
    x_bc0 = torch.tensor([[0.0]], dtype=torch.float32)
    x_bc1 = torch.tensor([[1.0]], dtype=torch.float32)

    num_layers = 4
    input_dim = 1
    hidden_dim = 20
    output_dim = 1
    n_epochs = 10000

    activations = ['tanh', 'relu', 'silu', 'linear', 'sigmoid']
    lr_dict = {
        'tanh': 1e-4,
        'relu': 5e-4,
        'silu': 5e-4,
        'linear': 5e-4,
        'sigmoid': 5e-4
    }

    predictions = {}

    for act in activations:
        print(f"\nTraining PINN with {act.upper()} activation...")
        model = SiLu(num_layers, input_dim, hidden_dim, output_dim, activation=act)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr_dict[act])

        for epoch in range(n_epochs):
            optimizer.zero_grad()
            T_int = model(x_int)

            if act == 'linear':
                loss_pde = torch.tensor([0.0], dtype = torch.float32).flatten()
            else:
                T_x = torch.autograd.grad(T_int, x_int, grad_outputs=torch.ones_like(T_int), create_graph=True)[0]
                T_xx = torch.autograd.grad(T_x, x_int, grad_outputs=torch.ones_like(T_x), create_graph=True)[0]
                loss_pde = torch.mean((T_xx + 1) ** 2)

            T0 = model(x_bc0)
            T1 = model(x_bc1)
            loss_bc = (T0 - T_left) ** 2 + (T1 - T_right) ** 2
            loss = loss_pde + loss_bc

            loss.backward()
            optimizer.step()

            if epoch % 1000 == 0 or epoch == n_epochs - 1:
                print(f"  Epoch {epoch:5d} - Loss: {loss.item():.6f}")

        x_plot = torch.linspace(0, 1, 101, dtype=torch.float32).unsqueeze(1)
        T_pred = model(x_plot).detach().numpy().flatten()
        predictions[act] = T_pred
        print(f"Finished training with {act}. Final loss: {loss.item():.6f}")

    x_plot_np = x_plot.numpy().flatten()
    T_exact = x_plot_np

    N_fd = 6
    x_fd = np.linspace(0, 1, N_fd)
    T_fd = x_fd

    plt.figure(figsize=(8, 6))
    plot_styles = {
        'tanh': {'color': 'blue', 'linestyle': '-'},
        'relu': {'color': 'red', 'linestyle': '--'},
        'silu': {'color': 'green', 'linestyle': '-.'},
        'linear': {'color': 'orange', 'linestyle': ':'},
        'sigmoid': {'color': 'purple', 'linestyle': '-'}
    }
    label_names = {
        'tanh': 'Tanh',
        'relu': 'ReLU',
        'silu': 'SiLU',
        'linear': 'Linear',
        'sigmoid': 'Sigmoid'
    }
    for act, T_pred in predictions.items():
        style = plot_styles[act]
        plt.plot(x_plot_np, T_pred, label=f"PINN ({label_names[act]})",
                 color=style['color'], linestyle=style['linestyle'])

    plt.plot(x_plot_np, T_exact, 'k-', linewidth=2, label='Exact solution T(x)=x')
    plt.plot(x_fd, T_fd, 'ko--', label='FD solution')

    plt.xlabel('x', fontsize=12)
    plt.ylabel('Temperature T(x)', fontsize=12)
    plt.title('PINN Solutions for 1D Steady Heat Conduction (Different Activations)', fontsize=14)
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()
