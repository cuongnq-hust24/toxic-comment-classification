# Toxic Comment Classification

This project is aimed at classifying toxic comments into multiple categories such as `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, and `identity_hate`. It uses machine learning techniques to identify and categorize harmful comments in text.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model](#model)
- [Contributing](#contributing)

## Overview

This project trains a machine learning model to classify toxic comments. It leverages TF-IDF for text representation and a multi-label classification model, such as the PassiveAggressiveClassifier, to classify the comments into one or more of the six categories. The model is trained using a dataset containing comments with corresponding labels.

## Installation

### Prerequisites

Make sure you have Python 3.x and the following libraries installed:

- pandas
- scikit-learn
- joblib

You can install the required libraries by running:

```bash
pip install -r requirements.txt
```

## Usage
The trained model and TF-IDF vectorizer will be saved in the models/ directory as toxic_comment_model.pkl and tfidf_vectorizer.pkl.
### Prediction on simple inputs
You can go to the predict_script.py and replace the input you want on the new_comments array. It will return a 1D array of 6 responding categories with 0 and 1 values.
### Prediction on large amount of inputs
You can use pandas to perform the task
- Go to predict_script.py and modify the test_comments and test_vector list
- Replace them with:
```bash
data = pd.read_csv('your_comment_data.csv')
data_comments_tfidf = loaded_vectorizer.transform(data['comment_text'])
#Predictions on the data
prediction = loaded_model.predict(test_comments_tfidf)
```

## Dataset

The dataset used for training the model is sourced from the [Jigsaw Toxic Comment Classification Challenge] on Kaggle. The dataset contains text comments with binary labels for six different toxicity categories: `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, and `identity_hate`.

You can find the dataset and more details about the challenge at the following link:

[Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)

Please note that you need to create a Kaggle account and agree to the competition’s terms to access the dataset.

## Model

The model uses the following architecture:

- **TF-IDF Vectorization**: Text data is transformed using TF-IDF to convert the text into numerical features.
- **Multi-Label Classification**: The model uses a `PassiveAggressiveClassifier` wrapped in a `MultiOutputClassifier` to predict multiple binary labels for each comment.

### Kaggle Submission Scores

This project was submitted to the [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) on Kaggle. The model achieved the following scores:

- **Private Score**: 0.77430
- **Public Score**: 0.75174

Please note that these scores are not particularly high and reflect the nature of this project as a **beginner-level reference**. It can be improved significantly with more advanced techniques, better hyperparameter tuning, and deeper model architectures.

## Contributing

Contributions are welcome! If you'd like to contribute to this project, please follow these steps:

1. **Fork the repository**: Click the "Fork" button at the top right of this repository's page on GitHub.
2. **Clone your fork**: Once you've forked the project, clone your copy to your local machine.
   ```bash
   git clone https://github.com/YOUR-USERNAME/toxic-comment-classification.git
   ```
3. **Commit your changes**: Stage and commit your changes with a descriptive commit message.
4. **Submit a pull request**: Go to the original repository and create a pull request from your forked branch. Provide a clear explanation of what changes you made and why they should be merged.





