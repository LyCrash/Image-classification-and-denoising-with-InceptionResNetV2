# Cotton Water Stress Classification and Denoising Using InceptionResNetV2

This project aims to improve a previous research work on **cotton water stress classification** using UAV-based RGB imagery. The project leverages the advanced **InceptionResNetV2** model, a deep convolutional neural network architecture, to replace the basic CNN used in earlier work. Additionally, the project explores image denoising options through two approaches:

1. **Multi-Task Learning (MTL)**: Denoising and classification are performed in parallel within the Inception model.
2. **Sequential Processing**: An independent **DnCNN** model is used for denoising before feeding the output into the Inception model.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Getting Started](#getting-started)
4. [Results and Discussion](#results-and-discussion)
5. [Conclusion and Future Improvements](#conclusion-and-future-improvements)
6. [References](#references)

---

## Project Overview
The primary goal of this project is to classify cotton water stress into different categories (**Rainfed**, **Fully irrigated**, **Percent deficit**, and **Time delay**) using advanced deep learning techniques for a purpose to analyse the different irrigation strategies and provide insights into optimal practices for enhancing plant growth. 

By integrating **InceptionResNetV2** and exploring denoising techniques, the project improves upon the previous work in terms of accuracy, training efficiency, and inference speed.

---

## Repository Structure
This repository contains both **documentation** and **code** to facilitate understanding and reproducibility.

### Documentation
- **Previous research's article**: Serves as a starting point and baseline for improvements.
- **State of the art**: Includes references and insights into Inception models, noise issues, and denoising models.
- **Presentation slides**: A PowerPoint presentation summarizing step-by-step progress, results, and limitations.

### Code
The project code is organized into Jupyter Notebooks:

1. **`InceptionResNetV2_model.ipynb`**
   - Customizes the pre-trained **InceptionResNetV2** model by adding final layers tailored to classify the target categories.

2. **`InceptionResNetV2_improved.ipynb`**
   - Implements **transfer learning** (TL) by fine-tuning the InceptionResNetV2 model and adding custom layers.

3. **`InceptionResNetV2_improved_with_morphology.ipynb`**
   - Similar to the previous notebook but applies image preprocessing steps described in the earlier research article.

4. **`InceptionResNetV2_with_denoising_DnCNN.ipynb`**
   - Sequential process: An independent **DnCNN** model is used to denoise input images before feeding them into the Inception model.

5. **`InceptionResNetV2_with_denoising_MTL.ipynb`**
   - Parallel process: Implements **multi-task learning (MTL)** where the Inception model simultaneously performs denoising and classification.

### Results
- All result reports, figures, and output images are saved in the **`images`** folder.

---

## Getting Started
To run the project, follow these steps:

### 1. Clone the Repository
```bash
git clone https://github.com/LyCrash/Image-classification-and-denoising-with-InceptionResNetV2.git
cd Image-classification-and-denoising-with-InceptionResNetV2
```

### 2. Set Up a Virtual Environment
It is recommended to use a Python virtual environment to manage dependencies:

**On Windows:**
```bash
python -m venv env
env\Scripts\activate
```

**On Linux/macOS:**
```bash
python3 -m venv env
source env/bin/activate
```

### 3. Install Dependencies
Run the following command to install all required libraries:

```bash
pip install -r requirements.txt
```

---

## Results and Discussion
- **Accuracy**: The **InceptionResNetV2** model achieved "almost" a similar accuracy compared to basic CNN architectures but required significantly fewer epochs for training, which means faster convergence.
- **Efficiency**: Training and inference times were improved due to the more efficient architecture of InceptionResNetV2.
- **Denoising**: Sequential (DnCNN) and parallel (MTL) denoising strategies were explored, showing improvements in image quality and classification performance.

All results, including plots and performance metrics, are stored in the **`images`** folder.

---

## Conclusion and Future Improvements
### Key Findings
- The **InceptionResNetV2** model outperformed basic CNN architectures in terms of efficiency and accuracy.
- The model can reach the same training accuracy in less epochs (10 instead of 70), but the validation/testing epochs suffers from overfitting
- Denoising techniques further enhanced the performance when noise was present in the input images.

### Limitations
- **Overfitting**: Despite improvements, overfitting remains a challenge due to the dataset's size and diversity, the quality or noise wasn't the main issue.
- **Optimisation**: Further tuning of model parameters and regularization techniques can reduce overfitting.
- **MTL Denoising**: The model can't effectively perform the denoising task parallely to the classification which was successful, but we intended to do such a parallel task so that the denoising task can help the classification.

### Future Work
- Increase dataset size and diversity to improve generalization (the images were not noisy)
- Fine-tune hyperparameters and regularization strategies for better performance.
- Gather and include all spatial, spectral, and temporal context for better model accuracy
- Explore more feature engineering strategy to improve the accuracy, as the image denoising wasn't enough
- Investigate the integration of Wide&Deep architecture into MTL model for better feature capture and less overfitting

---

## References
- Niu H, Landivar J, Duffield N. Classification of cotton water stress using convolutional neural networks and UAV-based RGB imagery. Advances in Modern Agriculture. 2024; 5(1): 2457. https://doi.org/10.54517/ama.v5i1.24 57
- Bharath, R. (2019, May 29). A Simple Guide to the Versions of the Inception Network. Towards Data Science. Retrieved from https://towardsdatascience.com/a-simple-guide-to-the-versions-of-the-inception-network-7fc52b863202
- Alifia, G. (2022, May 21). Understanding Architecture Of Inception Network & Applying It To A Real-World Dataset. Retrieved from https://gghantiwala.medium.com/understanding-the-architecture-of-the-inception-network-and-applying-it-to-a-real-world-dataset-169874795540

---

Feel free to contribute to this project or raise any issues! 😊

**Maintainer**: [LyCrash]

---
