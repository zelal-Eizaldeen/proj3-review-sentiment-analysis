# mymain.py

import re
import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import platform
import psutil
import time
import pickle

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import nltk

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import CountVectorizer

from lime.lime_text import LimeTextExplainer

# Ensure necessary NLTK data is downloaded
nltk.download('stopwords')


def get_hardware_info():
    """
    Retrieves and returns hardware information.
    """
    system = platform.system()
    processor = platform.processor()
    cpu_count = psutil.cpu_count(logical=False)
    total_memory = psutil.virtual_memory().total / (1024 ** 3)  # Convert to GB

    hardware_info = {
        'System': system,
        'Processor': processor,
        'Physical Cores': cpu_count,
        'Total Memory (GB)': round(total_memory, 2)
    }

    return hardware_info


def print_hardware_info():
    """
    Prints hardware information.
    """
    hardware_info = get_hardware_info()
    print("\n--- Hardware Information ---")
    for key, value in hardware_info.items():
        print(f"{key}: {value}")
    print("----------------------------\n")


def train_logistic(X_train, y_train):
    """
    Train a Logistic Regression model with Elastic Net regularization.

    Parameters:
    - X_train: Training features.
    - y_train: Training labels.

    Returns:
    - Trained LogisticRegression model.
    """
    model = LogisticRegression(
        penalty='elasticnet',
        solver='saga',
        l1_ratio=0,  # Ridge Regression
        C=5,
        max_iter=1000,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def predict(model, X_test):
    """
    Predict probabilities using the trained model.

    Parameters:
    - model: Trained model.
    - X_test: Test features.

    Returns:
    - Predicted probabilities for the positive class.
    """
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probabilities for positive sentiment
    return y_pred_proba


def review_to_words(raw_review, stops):
    """
    Clean and preprocess a raw review.

    Parameters:
    - raw_review: Raw text of the review.
    - stops: Set of stopwords to remove.

    Returns:
    - Cleaned and preprocessed review as a single string.
    """
    # Remove HTML
    review_text = BeautifulSoup(raw_review, "html.parser").get_text()
    # Remove non-letters
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    # Convert to lower case and split into words
    words = letters_only.lower().split()
    # Remove stop words
    meaningful_words = [w for w in words if w not in stops]
    return " ".join(meaningful_words)


def interpret_model(model, vectorizer, test_reviews, y_test, DATA_DIR):
    """
    Interpret model predictions for 5 positive and 5 negative reviews using LIME.

    Parameters:
    - model: Trained model (LogisticRegression).
    - vectorizer: Fitted CountVectorizer.
    - test_reviews: Original test reviews.
    - y_test: Model's predictions for test data.
    - DATA_DIR: Directory path to save interpretability outputs.
    """
    # Get model predictions to identify positive/negative examples
    test_features = vectorizer.transform([review_to_words(review, set(stopwords.words("english"))) 
                                        for review in test_reviews]).toarray()
    y_pred = model.predict(test_features)

    # Select 5 positive and 5 negative reviews based on model predictions
    np.random.seed(42)  # For reproducibility
    positive_indices = np.where(y_pred == 1)[0]
    negative_indices = np.where(y_pred == 0)[0]

    num_positive = min(5, len(positive_indices))
    num_negative = min(5, len(negative_indices))

    if num_positive < 5 or num_negative < 5:
        print("Warning: Not enough samples to select 5 positive and 5 negative reviews.")

    selected_positive = np.random.choice(positive_indices, size=num_positive, replace=False)
    selected_negative = np.random.choice(negative_indices, size=num_negative, replace=False)
    selected_indices = np.concatenate([selected_positive, selected_negative])

    selected_reviews = test_reviews.iloc[selected_indices].reset_index(drop=True)
    selected_labels = y_pred[selected_indices]

    # Create directory to save interpretability plots and highlighted reviews
    plots_dir = os.path.join(DATA_DIR, 'interpretability_plots')
    os.makedirs(plots_dir, exist_ok=True)
    highlighted_dir = os.path.join(plots_dir, 'highlighted_reviews')
    os.makedirs(highlighted_dir, exist_ok=True)

    # Global Interpretability for LogisticRegression
    if isinstance(model, LogisticRegression):
        if hasattr(model, 'coef_'):
            coefficients = model.coef_[0]
            feature_names = vectorizer.get_feature_names_out()
            coef_df = pd.DataFrame({
                'feature': feature_names,
                'coefficient': coefficients
            }).sort_values('coefficient', ascending=False)

            top_positive = coef_df.head(10)
            top_negative = coef_df.tail(10)

            plt.figure(figsize=(12, 12))
            plt.subplot(1, 2, 1)
            sns.barplot(data=top_positive, x='coefficient', y='feature', palette='Greens_d')
            plt.title('Top 10 Positive Coefficients')

            plt.subplot(1, 2, 2)
            sns.barplot(data=top_negative, x='coefficient', y='feature', palette='Reds_d')
            plt.title('Top 10 Negative Coefficients')

            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'coefficients.png'))
            plt.close()

    # Local Interpretability using LIME
    lime_explainer = LimeTextExplainer(class_names=['Negative', 'Positive'], random_state=42)

    for i in range(len(selected_reviews)):
        review = selected_reviews.iloc[i]
        label = selected_labels[i]
        review_text = review  # Assuming 'review' contains the raw text

        def predict_proba(texts):
            clean_texts = [review_to_words(text, set(stopwords.words("english"))) for text in texts]
            features = vectorizer.transform(clean_texts)
            return model.predict_proba(features)

        explanation = lime_explainer.explain_instance(
            review_text,
            predict_proba,
            num_features=10,
            labels=(1,)
        )

        # Save explanation as HTML
        explanation_html = explanation.as_html()
        explanation_path = os.path.join(plots_dir, f'review_{i+1}_lime.html')
        with open(explanation_path, 'w', encoding='utf-8') as f:
            f.write(explanation_html)

        # Save top contributing words as TXT
        top_features = explanation.as_list(label=1)[:10]

        filename_txt = f'review_{i+1}_{"positive" if label ==1 else "negative"}_lime.txt'
        filepath_txt = os.path.join(plots_dir, filename_txt)

        with open(filepath_txt, 'w') as f:
            f.write(f"Review {i+1} - {'Positive' if label ==1 else 'Negative'} Sentiment\n")
            f.write(f"Original review:\n{review_text}\n\n")
            f.write("Top contributing words (LIME):\n")
            for word, weight in top_features:
                f.write(f"{word}: {weight:.4f}\n")

        # Highlight the top words in the review and save as HTML
        filename_html = f'review_{i+1}_{"positive" if label ==1 else "negative"}_highlighted.html'
        filepath_html = os.path.join(highlighted_dir, filename_html)
        with open(filepath_html, 'w', encoding='utf-8') as f:
            highlighted_text = review_text
            for word, weight in top_features:
                highlighted_text = re.sub(f"\\b{re.escape(word)}\\b", f"<mark>{word}</mark>", highlighted_text, flags=re.IGNORECASE)
            f.write(f"<html><body><p>{highlighted_text}</p></body></html>")

    print(f"\nInterpretability visualizations saved in the directory: {plots_dir}")
    print(f"Highlighted reviews saved in the directory: {highlighted_dir}\n")



def main():
    warnings.filterwarnings('ignore')

    parser = argparse.ArgumentParser(description="Movie Review Sentiment Analysis")
    parser.add_argument('--dev', action='store_true', help='Run in development mode with multiple splits')
    parser.add_argument('--interp', action='store_true', help='Run model interpretation')
    args = parser.parse_args()

    print_hardware_info()

    DATA_DIR = 'p3_data'
    num_splits = 5

    if args.dev:
        print("Running in Development Mode...\n")
        total_time = 0
        for i in range(num_splits):
            split_start = time.time()
            split_dir = os.path.join(DATA_DIR, f"split_{i+1}")
            print(f"Processing Split {i+1}...")
            train = pd.read_csv(os.path.join(split_dir, "train.csv"))
            test = pd.read_csv(os.path.join(split_dir, "test.csv"))
            y_test = pd.read_csv(os.path.join(split_dir, "test_y.csv"))['sentiment']

            X_train = train.drop(columns=['id', 'sentiment', 'review'])
            y_train = train['sentiment']
            X_test = test.drop(columns=['id', 'review'])

            model_path = os.path.join(split_dir, "model.pkl")
            if os.path.exists(model_path):
                print("Loading existing Logistic Regression model...")
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                print("Model loaded successfully")
            else:
                print("Training new Logistic Regression model...")
                train_start = time.time()
                model = train_logistic(X_train, y_train)
                train_time = time.time() - train_start
                print(f"Training time: {train_time:.2f} seconds")
                # Save the model
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                print("Model saved successfully")
            
            y_pred_proba = predict(model, X_test)

            auc_score = roc_auc_score(y_test, y_pred_proba)
            split_time = time.time() - split_start
            total_time += split_time
            print(f"Split {i+1} AUC: {auc_score:.6f}")
            print(f"Split {i+1} execution time: {split_time:.2f} seconds\n")

            submission = pd.DataFrame({
                'id': test['id'],
                'prob': y_pred_proba
            })
            submission.to_csv(os.path.join(split_dir, "mysubmission.csv"), index=False)
            print(f"mysubmission.csv saved for Split {i+1}.\n")

        print(f"Total execution time for all splits: {total_time:.2f} seconds")
        print(f"Average execution time per split: {total_time/num_splits:.2f} seconds\n")

        # Interpretability Analysis for Split 1
        print("Starting Interpretability Analysis for Split 1...\n")
        split_dir = os.path.join(DATA_DIR, "split_1")
        train = pd.read_csv(os.path.join(split_dir, "train.csv"))
        test = pd.read_csv(os.path.join(split_dir, "test.csv"))
        y_test = pd.read_csv(os.path.join(split_dir, "test_y.csv"))['sentiment']

        stops = set(stopwords.words("english"))

        # Clean and preprocess test reviews
        print("Cleaning and preprocessing test reviews...")
        clean_test_reviews = [review_to_words(r, stops) for r in test["review"]]

        # Vectorization
        print("Vectorizing text data...")
        vectorizer = CountVectorizer(
            analyzer="word",
            tokenizer=None,
            preprocessor=None,
            stop_words=None,
            max_features=5000,
            ngram_range=(1, 4),
            min_df=0.001,
            max_df=0.5,
            token_pattern=r"\b[\w+|']+\b"
        )
        vectorizer.fit(clean_test_reviews)
        # Transform training data
        train_data_features = vectorizer.transform([review_to_words(r, stops) for r in train["review"]]).toarray()
        test_data_features = vectorizer.transform(clean_test_reviews).toarray()

        # Train model
        print("Training Logistic Regression model...")
        model_path = os.path.join(split_dir, "model_interp.pkl")
        if os.path.exists(model_path):
            print("Loading existing model...")
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        else:
            print("Training new model...")
            start_time = time.time()
            model = train_logistic(train_data_features, train['sentiment'])
            train_time = time.time() - start_time
            print(f"Model training time: {train_time:.2f} seconds")
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)

        # Interpret model
        interpret_model(
            model=model,
            vectorizer=vectorizer,
            test_reviews=test['review'],
            y_test=y_test,
            DATA_DIR=split_dir
        )
    elif args.interp:
        print("Running in Submission Mode...\n")
        split_dir = os.path.join(DATA_DIR, "split_1")
        #train = pd.read_csv(os.path.join(split_dir, "train.csv"))
        #test = pd.read_csv(os.path.join(split_dir, "test.csv"))
        train = pd.read_csv((os.getcwd()+'/p3_data/split_1/train.csv').replace("\\", "/"))
        test = pd.read_csv((os.getcwd()+'/p3_data/split_1/test.csv').replace("\\", "/"))

        # Preprocess text data
        stops = set(stopwords.words("english"))
        print("Cleaning and preprocessing reviews...")
        clean_train_reviews = [review_to_words(r, stops) for r in train["review"]]
        clean_test_reviews = [review_to_words(r, stops) for r in test["review"]]

        # Vectorize text data
        print("Vectorizing text data...")
        vectorizer = CountVectorizer(
            analyzer="word",
            tokenizer=None,
            preprocessor=None,
            stop_words=None,
            max_features=5000,
            ngram_range=(1, 4),
            min_df=0.001,
            max_df=0.5,
            token_pattern=r"\b[\w+|']+\b"
        )
        vectorizer.fit(clean_train_reviews + clean_test_reviews)
        train_data_features = vectorizer.transform(clean_train_reviews).toarray()
        test_data_features = vectorizer.transform(clean_test_reviews).toarray()

        # Train or load model based on interp flag
        if args.interp:
            model_path = os.path.join(".", "model_interp.pkl")
        else:
            model_path = os.path.join(".", "model.pkl")

        if os.path.exists(model_path):
            print("Loading existing model...")
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        else:
            print("Training new model...")
            start_time = time.time()
            model = train_logistic(train_data_features, train['sentiment'])
            train_time = time.time() - start_time
            print(f"Model training time: {train_time:.2f} seconds")
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)

        # Generate predictions
        y_pred_proba = predict(model, test_data_features)

        # Save submission file
        submission = pd.DataFrame({
            'id': test['id'],
            'prob': y_pred_proba
        })
        submission.to_csv("mysubmission.csv", index=False)
        print("mysubmission.csv has been saved in the root directory.\n")

        # Run interpretation if flag is set
        if args.interp:
            interpret_model(
                model=model,
                vectorizer=vectorizer,
                test_reviews=test['review'],
                y_test=None,
                DATA_DIR="."
            )
    else:
        print("Running in plain Mode...\n")
        total_time = 0
        for i in range(0,1):
            split_start = time.time()
            split_dir = ''
            #print(f"Processing Split {i+1}...")
            train = pd.read_csv('train.csv')
            test = pd.read_csv('test.csv')
            #y_test = pd.read_csv(os.path.join(split_dir, "test_y.csv"))['sentiment']

            X_train = train.drop(columns=['id', 'sentiment', 'review'])
            y_train = train['sentiment']
            X_test = test.drop(columns=['id', 'review'])

            
            try:
                model_path = os.path.join('simple is better', "model.pkl")
                # print("Loading existing Logistic Regression model...")
                # with open(model_path, 'rb') as f:
                    # model = pickle.load(f)
                # print("Model loaded successfully")
            except:
                print("Training new Logistic Regression model...")
            train_start = time.time()
            model = train_logistic(X_train, y_train)
            train_time = time.time() - train_start
            print(f"Training time: {train_time:.2f} seconds")
            # Save the model
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print("Model saved successfully")
            
            y_pred_proba = predict(model, X_test)

            #auc_score = roc_auc_score(y_test, y_pred_proba)
            split_time = time.time() - split_start
            total_time += split_time
            #print(f"Split {i+1} AUC: {auc_score:.6f}")
            print(f"execution time: {split_time:.2f} seconds\n")

            submission = pd.DataFrame({
                'id': test['id'],
                'prob': y_pred_proba
            })
            submission.to_csv("mysubmission.csv", index=False)
            print(f"mysubmission.csv saved.\n")



if __name__ == "__main__":
    main()
