import os

os.environ["DDE_BACKEND"] = "tensorflow.compat.v1"
import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt


def analytical_solution(x_coords, M_terms=20, N_terms=20):
    x = x_coords[:, 0:1]
    y = x_coords[:, 1:2]
    T = np.zeros_like(x)
    for m in range(1, 2 * M_terms, 2):
        for n in range(1, 2 * N_terms, 2):
            coeff = 16.0 / (np.pi ** 4 * m * n * (m ** 2 + n ** 2))
            T += coeff * np.sin(m * np.pi * x) * np.sin(n * np.pi * y)
    return T


def pde(x, T):
    T_xx = dde.grad.hessian(T, x, i=0, j=0)
    T_yy = dde.grad.hessian(T, x, i=1, j=1)
    return T_xx + T_yy + 1.0


geo = dde.geometry.Rectangle([0, 0], [1, 1])
bc = dde.icbc.DirichletBC(geo, lambda x: 0, lambda _, on_b: on_b)
data = dde.data.PDE(geo, pde, bc, num_domain=2500, num_boundary=400, solution=analytical_solution, num_test=10000)
activations = ["tanh", "relu", "silu", "sigmoid"]
results = {}
for act in activations:
    net = dde.nn.FNN([2] + [50] * 3 + [1], act, "Glorot normal")
    model = dde.Model(data, net)
    model.compile("adam", lr=1e-3, metrics=[dde.metrics.mean_squared_error])
    losshistory, train_state = model.train(iterations=20000, display_every=1000)
    y_true = analytical_solution(data.test_x)
    y_pred = model.predict(data.test_x)
    mse = dde.metrics.mean_squared_error(y_true, y_pred)
    abs_err = np.abs(y_true - y_pred)
    pct_err = abs_err / (np.abs(y_true) + 1e-8)
    results[act] = {
        "MSE": mse,
        "abs_err": abs_err,
        "pct_err": pct_err,
        "loss_train": np.array(losshistory.loss_train),
    }
plt.figure()
for act in activations:
    iters = np.arange(len(results[act]["loss_train"]))
    plt.plot(iters, results[act]["loss_train"], label=act)
plt.yscale("log")
plt.legend()
plt.title("Training Loss vs Iterations")
plt.figure()
for act in activations:
    err = results[act]["MSE"]
    plt.bar(act, err)
plt.ylabel("Mean Squared Error")
plt.title("Mean Squared Error per Activation")
plt.figure()
for act in activations:
    plt.hist(results[act]["abs_err"].ravel(), bins=50, alpha=0.5, label=act)
plt.legend()
plt.title("Absolute Error Distribution")
plt.show()
