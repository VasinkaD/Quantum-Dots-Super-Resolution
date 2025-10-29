# Super-resolution of Quantum Dots

This repository provides data and supplementary material to the paper **Universal super-resolution framework for imaging of quantum dots**, by Dominik Vašinka, Jaewon Lee, et al.

The paper is available on [arXiv](https://doi.org/10.48550/arXiv.2502.18637).

The repository is currently under development.

## The repository structure:
The "Data" directory contains the measured low-resolution camera images presented in the manuscript, which are to be processed by the deep learning model stored in the "Model" directory. <be>
The two files "func_file_Model.py" and "func_file_Localization.py" contain function definitions necessary for evaluating the Jupyter notebooks.

Sample_1.ipynb
- processing the first low-resolution camera image, i.e., the sparse image containing 4 quantum dots
- deep learning model reconstruction of the high-resolution image
- localization of the quantum dots using a Gaussian fit
- generates the reconstructed images presented in the manuscript
- evaluates the resolution and additional variables presented in the text

Sample_2.ipynb
- processing the second low-resolution camera image, i.e., blinking of the quantum dots
- deep learning model reconstruction of both high-resolution images
- localization of the blinking quantum dot using a Gaussian fit on the temporal difference of the images
- generates the reconstructed images presented in the manuscript
- evaluates the resolution and additional variables presented in the text

Sample_3.ipynb
- processing the third low-resolution camera image, i.e., the quantum dots in the WSe$_2$ monolayer
- deep learning model reconstruction of the high-resolution image
- generates the reconstructed image presented in the manuscript
- evaluates the resolution and additional variables presented in the text
