**MNIST Handwritten Digit Classifier — Python ML**

A machine learning project that trains and compares two classifiers — logistic regression and a neural network — on the MNIST handwritten digit dataset, with analysis of where and why each model fails.

**Dataset**

UCI digits dataset via sklearn: 1,797 images of handwritten digits 0–9, each stored as an 8x8 pixel grid flattened into 64 brightness values ranging from 0 (black) to 16 (white).

**Models**

Logistic regression — learns a weight for each pixel per digit class. Predicts by multiplying pixel values by learned weights and picking the highest scoring class. Simple, fast, and interpretable.

Neural network (MLP) — two hidden layers (128 → 64 neurons) trained via backpropagation. More complex but not necessarily better on small datasets.

**Results**

Model	Accuracy
Logistic Regression	97.50%
Neural Network	97.22%

Logistic regression outperformed the neural network by 0.28%. On a small, clean dataset like this, simpler models generalize better — neural networks need significantly more data to leverage their complexity advantage.

**Error Analysis**

Digit 5 was the hardest for both models — confused with 3, 6, 7, 8, and 9. Analysis of misclassified images revealed that errors occurred when digits were missing their discriminative pixels — the specific pixel positions that uniquely identify a class. When those key pixels are absent or ambiguous, the model falls back on shared features and predicts the wrong digit.

Key insight: feature quality matters more than model complexity on small datasets. A logistic regression with clean, complete images outperforms a neural network on the same data.

**Tools**

Python, scikit-learn, matplotlib, seaborn
Google Colab (browser-based, no local install)

What I'd improve with more time

Test on the full 70,000 image MNIST dataset instead of the 1,797 sample sklearn version
Implement a convolutional neural network (CNN) which uses spatial structure of the image rather than treating pixels independently
Apply data augmentation — rotate and shift training images to make the model more robust to imperfect handwriting
