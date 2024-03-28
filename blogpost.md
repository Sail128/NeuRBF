# Reproduction and ablation study of "NeuRBF: ANeural Fields Representation with Adaptive Radial Basis Functions"

This blog post documents the reproduction and ablation study from group 42* as part of the CS4240 Deep Learning 2023–24 course.

| Authors    | Student number |
| -------- | ------- |
| Levijn de Jager  | 4903668    |
| Placeholder  |   Placeholder  |
|  Placeholder  | Placeholder   |
|  Placeholder  |  Placeholder |

## Introduction

## Dataset

## RBF function

## Sinusoidal composition
The paper extends the radial basis function by adding a sinusoidal composition on the the radial basis with different frequencies. The formulation is as follows:

![alt text](radial_basis_sinus_function.png)

The different frequencies are determined by setting a maximum and minimum for m. The rest of the elements are obtained by log-linearly dividing the range between the maximum and minimum. Comparing this to the fourier basis or gabor basis seen in the figure it is now possible to have a basis with non-linear paterns.

![Radial basis plot](radial_basis_plot.png)

The sinusoidal composition method is also applied to the output of the first fully connected layer in the MLP. The output is then used as the input for the next layer. 

They claim that using these sinusoidal compositions improve the performance. They also documented the results they got with and without these additional compositions and we want to check these results by doing the same ablation study and compare our results but with a different dataset. 

Our results can be seen in the table below:
| Method                        | Our Average PSNR | Their Average PSNR |
| ----------------------------- | ---------------- | ------------------ |
| With Sinusoidal composition   |   0   | 0 |
| Only on feature vector        |   0   | 0 |
|  Only on RBF function         |   0   | 0 |
|  No Sinusoidal composition    |   0   | 0 |