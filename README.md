# Gradient-Origin-Networks-Julia

Reimplementation of Gradient Origin Networks in Julia with Knet Framework (Koç University Deep Learning Framework).

This repo contains the Julia implementation of the paper Gradient Origin Networks [1] with the Knet Framework.

# TO-DO LIST

-> This readme file will contain more details about the implementation.

-> I will provide my LaTex report after I am done with the experimentations

# Implementation Environment Details

-> Julia Version Info: Version 1.6.3 (2021-09-23)

-> Platform Info :  "OS: Linux (x86_64-pc-linux-gnu) CPU: Intel(R) Xeon(R) Gold 6248 CPU @ 2.50GHz, GPU : CuDevice(0): Tesla T4"

-> Required Julia Packages : Specified in requirements.txt file

# Implementation Notes

-> src folder includes Julia implementations for GON, Variational GON and Implicit GON models as well as some other model implementations (AE, VAE, etc.) to compare.

-> Pytorch Debug Files : The codes in Pytorch_Debug_Files are based on the published GON implementation [2] which provides loggers for train and test set loss values for each epoch. I use them to compare my Julia implementation with Pytorch implementation.

-> Experiment Notebooks : "Experimental_Notebooks" and "Experimental_Notebooks_Deprecated" files includes my debugging and experimenting notebooks. However, these notebooks will be deleted soon since I include the finalized version of the codes in "src" folder.


# Refences

[1] Sam Bond-Taylor and Chris G. Willcocks. Gradient origin networks. In International Conference on Learning Representations, 2021. URL https://openreview.net/pdf?id=0O_cQfw6uEh.

[2] https://github.com/cwkx/GON
