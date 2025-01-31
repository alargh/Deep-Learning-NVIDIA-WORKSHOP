# README - Deep Learning Model Assessment

## Overview
This repository contains my submission for the final assessment of the **Fundamentals of Deep Learning** course offered by NVIDIA's Deep Learning Institute (DLI). The course equipped me with essential skills and hands-on experience in building and training deep learning models.

## About the Course
The **Fundamentals of Deep Learning** course provides in-depth training in deep learning techniques and frameworks, focusing on:
- Training deep neural networks for computer vision and natural language processing (NLP)
- Applying convolutional neural networks (CNNs) for image classification tasks
- Enhancing model performance through data augmentation techniques
- Leveraging transfer learning to fine-tune pre-trained models
- Implementing models using frameworks like TensorFlow, Keras, Pandas, and NumPy

### Course Highlights
Throughout the course, participants worked with real-world datasets, trained models from scratch, and optimized deep learning pipelines using GPU-accelerated computing. The final assessment involved training a model to classify images of fresh and rotten fruits.

## About My Assessment
For my final project, I trained a deep learning model to classify images of fresh and rotten fruits. The dataset, sourced from Kaggle, consists of six categories:
- **Fresh Apples**
- **Fresh Oranges**
- **Fresh Bananas**
- **Rotten Apples**
- **Rotten Oranges**
- **Rotten Bananas**

### Model Architecture & Training Details
I used a custom architecture built on a VGG model. The structure is as follows:
- **Neural Network Structure:**
  - Input layer for image data
  - Convolutional layers and average pooling from the pre-trained VGG model
  - Fully connected layers with ReLU activations
  - Output layer with six neurons, corresponding to each fruit category
  - Final output layer uses LogSoftmax activation for classification
- **Loss Function:** Categorical Crossentropy
- **Optimization Algorithm:** Adam
- **Evaluation Metric:** Accuracy on the validation dataset

### Data Augmentation & Preprocessing
To improve model performance, I applied several data augmentation techniques to the training images:
- **Random rotations** and resized crops to introduce variability
- **Horizontal flips** and color jitter to make the model more robust
- **Random affine transformations** to simulate various real-world distortions
- **Gaussian blur** for simulating different image qualities

## Key Learnings & Takeaways
- Mastered the art of data augmentation to prevent overfitting and improve model generalization
- Gained hands-on experience with transfer learning, fine-tuning a pre-trained VGG model to suit the classification task
- Deepened my understanding of model optimization and hyperparameter tuning to achieve high accuracy
- Successfully implemented a solution to classify fruit images with robust performance

## Repository Contents
- **assessment.ipynb** – The Jupyter notebook containing the deep learning model implementation and training code.
- **certificate.pdf** – The official NVIDIA DLI certificate confirming my completion of the **Fundamentals of Deep Learning** course.
