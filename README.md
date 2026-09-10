# ML_Training

A collection of exercises working through the fundamentals of machine learning. Signals, k-nearest neighbours, linear regression, and a neural network.**

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=python&logoColor=white)

---

## What it is

Instead of following a tutorial end to end, I set myself exercises: take one idea from the fundamentals, build a dataset for it, and implement the thing until the plot looks right.

Every dataset here is synthetic and visual: sound waves, colour patterns, paintings and shapes.

Loosely inspired by [ml4a](https://ml4a.net/fundamentals/), Gene Kogan's machine learning course for artists.

## Exercises

 - **TP1: Signals**: Synthesise a 440 Hz sine wave, then take it apart: linear fade envelopes, additive Gaussian noise, smoothing by convolution with a Gaussian kernel. NumPy, plotted with Matplotlib.
 
 - **TP2: Pattern classification with k-NN**: Generate 200 images across four visual styles (stripes, dots, gradients and random shapes), then reduce each one to a 24-value RGB histogram and classify it with k-nearest neighbours. Misclassified images are displayed so the failures can be looked at rather than just counted.
 
 - **TP3: Linear regression, solved by hand**: A small dataset of paintings, each with a year and an average red channel value. Fit a line through it by solving the normal equation directly, then measure the error.
 
 - **TP4: Neural network from scratch**: Generate 10×10 images of squares, circles and triangles, flip a fraction of the pixels for noise, then build the pieces of the network by hand.

## Credits

Inspired by [ml4a — Fundamentals](https://ml4a.net/fundamentals/), maintained by [Gene Kogan](https://github.com/genekogan).

Built by [Samy Abdelazim](https://www.samy-abdelazim.com)
