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
The paper extends the radial basis function by adding a multi-frequency sinusoidal composition (MSC) on the the radial basis with different frequencies. The formulation is as follows:

![alt text](radial_basis_sinus_function.png)

The different frequencies are determined by setting a maximum and minimum for m. The rest of the elements are obtained by log-linearly dividing the range between the maximum and minimum. Comparing this to the fourier basis or gabor basis seen in the figure it is now possible to have a basis with non-linear paterns.

![Radial basis plot](radial_basis_plot.png)

The sinusoidal composition method is also applied to the output of the first fully connected layer in the MLP. The output is then used as the input for the next layer. 

They claim that using these sinusoidal compositions improve the performance. They also documented the results they got with and without these additional compositions and we want to check these results by doing the same ablation study and compare our results but with a different dataset. 

Our results can be seen in the table below. On the left are our results on our own dataset of 50 high resolution images. On the right are the results from the ablation study from the paper itself. They used a dataset of 100 images.
| Method                        | Our Average PSNR | Their Average PSNR |
| ----------------------------- | ---------------- | ------------------ |
| With Sinusoidal composition   |   41.45   | 51.53 |
| Only on feature vector        |   39.76   | 48.19 |
|  Only on RBF function         |   42.38   | 48.46 |
|  No Sinusoidal composition    |   39.96   | 43.81 |

Firstly, the PSNR values between our ablation study and theirs is very different. They report a lot higher values in general. Moreover, our results show that the MSC on the RBF has a bigger influence on the results then MSC on the feature vector whilest they report about the same amount of influence of the MSC on the RBF and the feature vector on the results.