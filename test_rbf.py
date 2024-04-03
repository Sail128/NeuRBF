import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
from util_network import * #rbf_gauss_a_fb, rbf_ivq_a_fb, rbf_qd_a_fb, rbf_mqd_a_fb,rbf_nlin_f_fb, rbf_ivmq_a_fb

import matplotlib.pyplot as plt

def plot_rbf(name,rbf):
    # Define the centers and scaling factors
    kc = torch.tensor([[0.0], [0.0]])  # centers at 0 and 1
    ks = torch.eye(1).repeat(2, 1, 1)  # identity matrix as scaling factors

    # Generate a range of input values
    x_values = torch.linspace(-4, 4, 1000)

    # Compute the RBF output for each input value
    y_values = torch.zeros_like(x_values)
    for i, x in enumerate(x_values):
        y, _ = rbf(x.view(1), kc, ks)
        y_values[i] = y.sum()  # sum the outputs from all centers

    # Plot the input values against the RBF output
    plt.plot(x_values.numpy(), y_values.numpy())
    plt.xlabel('x')
    plt.ylabel('RBF output')
    plt.title(f'1D {name} RBF')
    plt.show()

# plot_rbf("gaussian",rbf_gauss_a_fb)
# plot_rbf("inverse quadratic",rbf_ivq_a_fb)
# plot_rbf("quadratic",rbf_qd_a_fb)
# plot_rbf("nlin",rbf_nlin_f_fb)
# plot_rbf("inverse multiquadratic",rbf_ivmq_a_fb)
# plot_rbf("multiquadratic",rbf_mqd_a_fb)
plot_rbf("expsin",rbf_expsin_a_fb)
# plot_rbf("phs1",rbf_phs1_a_fb)
