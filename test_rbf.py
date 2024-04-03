import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
from mpl_toolkits.mplot3d import Axes3D
from util_network import * #rbf_gauss_a_fb, rbf_ivq_a_fb, rbf_qd_a_fb, rbf_mqd_a_fb,rbf_nlin_f_fb, rbf_ivmq_a_fb

import matplotlib.pyplot as plt




def plot_rbf(name,rbf):
    # Define the centers and scaling factors
    kc = torch.tensor([[0.0], [0.0]])  # centers at 0 and 0
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

def calc_rbf_2d(rbf, kc, ks, n_points=1000):
    # Generate a grid of input values
    x = torch.linspace(-4, 4, n_points) 
    y = torch.linspace(-4, 4, n_points)
    X, Y = torch.meshgrid(x, y)
    grid = torch.stack((X.flatten(), Y.flatten()), dim=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Compute the RBF output for each point in the grid
    with torch.no_grad():
        grid = grid.to(device)  # Assuming you have a GPU device
        Z, _ = rbf(grid, kc, ks)
        Z = Z.sum(dim=1).view(X.shape)  # sum the outputs from all centers

    return Z

def plot_rbf_2d(name, rbf):
    # Define the centers and scaling factors
    kc = torch.tensor([[0.0, 0.0]])  # centers at (0, 0), (1, 1), and (-1, -1)
    # ks = torch.eye(2).repeat(3, 1, 1)  # identity matrix as scaling factors
    ks = torch.tensor([[1.0, 0.0], [0.0, 1.0]]).repeat(1, 1, 1)  # identity matrix as scaling factors

    # Compute the RBF output for each input value
    Z = calc_rbf_2d(rbf, kc, ks)

    # Plot the RBF output as a 2D surface
    fig = plt.figure()
    plt.imshow(Z.cpu().numpy(), extent=[-4, 4, -4, 4], origin='lower', cmap='viridis')
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'2D {name} RBF')
    plt.show()


kc = torch.tensor([[0.0, 0.0]])  # centers at (0, 0), (1, 1), and (-1, -1)
# ks = torch.eye(2).repeat(3, 1, 1)  # identity matrix as scaling factors
ks = torch.tensor([[0.5, 0.0], [0.0, 0.5]]).repeat(1, 1, 1)  # identity matrix as scaling factors

# Define the RBF functions
rbf_functions = [rbf_ivq_a_fb, rbf_gauss_a_fb, rbf_ivmq_a_fb, rbf_mqd_a_fb, rbf_expsin_a_fb, rbf_nlin_f_fb]

# Define the names of the RBF functions
rbf_names = ["inverse quadratic", "gaussian", "inverse multiquadratic", "multiquadratic", "exponential sin", "non linear"]

# Create a 3x2 grid of subplots
fig, axs = plt.subplots(3, 2, figsize=(7, 10))

# Iterate over the RBF functions and names
for i, (rbf, name) in enumerate(zip(rbf_functions, rbf_names)):
    # Compute the RBF output for each function
    Z = calc_rbf_2d(rbf, kc, ks)

    # Plot the RBF output on the corresponding subplot
    ax = axs[i // 2, i % 2]
    ax.imshow(Z.cpu().numpy(), extent=[-4, 4, -4, 4], origin='lower', cmap='viridis')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'2D {name} RBF')

# Adjust the spacing between subplots
fig.tight_layout()
fig.savefig("blogpost_assets/2d_plotted_rbfs.png")

# Show the plot
# plt.show()

fig = plt.figure(figsize=(8, 10))
for i, (rbf, name) in enumerate(zip(rbf_functions, rbf_names)):
    # Compute the RBF output for each function
    Z = calc_rbf_2d(rbf, kc, ks)

    # Create a 3D subplot
    ax = fig.add_subplot(3, 2, i+1, projection='3d')

    # Generate a grid of input values
    x = torch.linspace(-4, 4, Z.shape[0])
    y = torch.linspace(-4, 4, Z.shape[1])
    X, Y = torch.meshgrid(x, y)

    # Plot the RBF output as a 3D surface
    ax.plot_surface(X.cpu().numpy(), Y.cpu().numpy(), Z.cpu().numpy(), cmap='viridis')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('RBF output')
    ax.set_title(f'{name} RBF', y=0.98)

# Adjust the spacing between subplots
fig.tight_layout()
fig.savefig("blogpost_assets/3d_plotted_rbfs.png")
plt.show()


# plot_rbf_2d("gaussian", rbf_gauss_a_fb)
# plot_rbf_2d("inverse quadratic", rbf_ivq_a_fb)
# plot_rbf_2d("quadratic", rbf_qd_a_fb)
# plot_rbf_2d("nlin", rbf_nlin_f_fb)
# plot_rbf_2d("inverse multiquadratic", rbf_ivmq_a_fb)
# plot_rbf_2d("multiquadratic", rbf_mqd_a_fb)
# plot_rbf_2d("expsin", rbf_expsin_a_fb)
# plot_rbf_2d("phs1", rbf_phs1_a_fb)

# plot_rbf("gaussian",rbf_gauss_a_fb)
# plot_rbf("inverse quadratic",rbf_ivq_a_fb)
# plot_rbf("quadratic",rbf_qd_a_fb)
# plot_rbf("nlin",rbf_nlin_f_fb)
# plot_rbf("inverse multiquadratic",rbf_ivmq_a_fb)
# plot_rbf("multiquadratic",rbf_mqd_a_fb)
# plot_rbf("expsin",rbf_expsin_a_fb)
# plot_rbf("phs1",rbf_phs1_a_fb)
