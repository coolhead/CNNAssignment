# Melanoma Detection using Custom Convolutional Neural Networks

> This project focuses on building and evaluating custom CNN models for melanoma detection, emphasizing innovation in architecture design and data preprocessing.

## Table of Contents
* [General Information](#general-information)
* [Key Results](#key-results)
* [Model Architectures](#model-architectures)
* [Technologies Used](#technologies-used)
* [Conclusions](#conclusions)
* [Acknowledgements](#acknowledgements)

## General Information
- **Background**: Melanoma is a critical healthcare challenge, and early diagnosis improves patient outcomes. Using custom-built deep learning models, this project aims to provide a baseline system for melanoma classification.
- **Business Problem**: The primary goal is to classify melanoma from other skin diseases in the HAM10000 dataset, addressing class imbalances and accuracy challenges.
- **Dataset**: The HAM10000 dataset contains labeled images for various skin conditions, used to train and evaluate models.

---

## Key Results

| Model    | Accuracy | Precision | Recall | F1-Score | Key Observations |
|----------|----------|-----------|--------|----------|-------------------|
| Model 1  | 0.35     | 0.37      | 0.34   | 0.35     | Initial baseline, low performance. |
| Model 4  | 0.43     | 0.45      | 0.41   | 0.43     | Improved with advanced augmentation techniques. |
| Model 5  | 0.48     | 0.50      | 0.47   | 0.48     | Final tuned model, incorporating weighted classes and hyperparameter optimization. |

- The best-performing model (Model 5) achieved a **48% accuracy** and a **macro-average F1-Score of 0.48**, with significant improvements in class-specific metrics.

### Confusion Matrix for Model 5
A visualized confusion matrix illustrates the model's performance across different classes:

[[ 0  4  1  1  6  3  0  1  0]
 [ 1  3  2  1  2  5  0  1  1]
 [ 1  0  2  0  9  2  0  2  0]
 [ 0  2  1  1  6  4  0  2  0]
 [ 1  4  1  0  3  2  0  4  1]
 [ 1  0  1  1 11  2  0  0  0]
 [ 0  0  0  1  1  1  0  0  0]
 [ 0  3  2  1  5  4  0  0  1]
 [ 0  0  0  1  0  2  0  0  0]]

### ROC-AUC Curves
The following ROC-AUC curves depict class-wise performance:

![ROC-AUC Curves](path/to/roc_auc.png)

---

## Model Architectures
### Model 1 (Baseline CNN)
- **Description**: Simple CNN with 3 convolutional layers, minimal augmentation.
- **Performance**: Provided a low baseline for further improvements.

### Model 4 (Augmented CNN)
- **Description**: Custom CNN incorporating rotation, zoom, and vertical flipping augmentations.
- **Key Features**: Applied early stopping and learning rate scheduling for stability.
- **Performance**: Achieved notable improvement over Model 1.

### Model 5 (Final Tuned CNN)
- **Description**: Enhanced data augmentation (brightness, contrast adjustments), hyperparameter optimization (learning rate, architecture depth).
- **Key Features**: Added F1-Score as a metric to monitor during training.
- **Performance**: Best model with superior precision, recall, and F1-score.

---

## Technologies Used
- Python 3.8
- TensorFlow 2.10.0
- NumPy 1.22.4
- Scikit-learn 1.1.2
- Matplotlib 3.5.1
- Jupyter Notebook

---

## Conclusions
- **Data Augmentation**: Advanced techniques significantly improved model robustness.
- **Hyperparameter Optimization**: Fine-tuned learning rates, regularization, and early stopping were critical to performance gains.
- **Class Imbalance**: Weighted classes and specific evaluation metrics helped address imbalances in the dataset.
- **Generalization**: While results are promising, further improvement can be achieved with larger datasets and architecture tuning.

---

## Acknowledgements
- This project is inspired by the assignment from [upGrad](https://www.upgrad.com/), using the HAM10000 dataset.
- References:
  - [HAM10000 Dataset](https://doi.org/10.1038/sdata.2018.161)
  - [Starter Code](https://github.com/ContentUpgrad/Convolutional-Neural-Networks/blob/main/Melanoma%20Detection%20Assignment/Starter_code_Assignment_CNN_Skin_Cancer%20(1).ipynb)

## Contact
Created by [@coolhead] - feel free to contact me for further collaboration or discussions!