# p3.py

import re
import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from bs4 import BeautifulSoup, NavigableString
from nltk.corpus import stopwords
import nltk

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import CountVectorizer

from lime.lime_text import LimeTextExplainer

# Ensure necessary NLTK data is downloaded
nltk.download('stopwords')


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


def highlight_important_words(review, important_words, output_path):
    """
    Highlights the important words in the original review by wrapping them in HTML span tags.

    Parameters:
    - review: Original review text.
    - important_words: List of words to highlight.
    - output_path: File path to save the highlighted HTML.
    """
    # Parse the HTML content
    soup = BeautifulSoup(review, "html.parser")

    # Compile regex patterns for all important words
    patterns = {word: re.compile(r'\b{}\b'.format(re.escape(word)), re.IGNORECASE) for word in important_words}

    # Function to recursively traverse and highlight words in text nodes
    def recursive_highlight(element):
        for content in element.contents:
            if isinstance(content, NavigableString):
                new_content = str(content)
                for word, pattern in patterns.items():
                    # Replace matched words with highlighted span
                    new_content = pattern.sub(
                        lambda match: f"<span style='background-color: yellow;'>{match.group(0)}</span>",
                        new_content
                    )
                # Replace the old text with the new highlighted text
                if new_content != content:
                    new_fragment = BeautifulSoup(new_content, "html.parser")
                    content.replace_with(new_fragment)
            elif content.name is not None:
                # Recursively process child elements
                recursive_highlight(content)

    # Start the recursive highlighting from the root
    recursive_highlight(soup)

    # Save the highlighted review to an HTML file
    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(str(soup))


def interpret_model(model, vectorizer, test_data_features, test_reviews, y_test, DATA_DIR):
    """
    Interpret model predictions for 5 positive and 5 negative reviews using LIME.

    Parameters:
    - model: Trained model (LogisticRegression).
    - vectorizer: Fitted CountVectorizer.
    - test_data_features: Feature matrix for test data.
    - test_reviews: Original test reviews.
    - y_test: True labels for test data.
    - DATA_DIR: Directory path to save interpretability outputs.
    """
    # Select 5 positive and 5 negative reviews
    np.random.seed(42)  # For reproducibility
    positive_indices = np.where(y_test == 1)[0]
    negative_indices = np.where(y_test == 0)[0]

    num_positive = min(5, len(positive_indices))
    num_negative = min(5, len(negative_indices))

    if num_positive < 5 or num_negative < 5:
        print("Warning: Not enough samples to select 5 positive and 5 negative reviews.")

    selected_positive = np.random.choice(positive_indices, size=num_positive, replace=False)
    selected_negative = np.random.choice(negative_indices, size=num_negative, replace=False)
    selected_indices = np.concatenate([selected_positive, selected_negative])

    selected_reviews = test_reviews.iloc[selected_indices].reset_index(drop=True)
    selected_features = test_data_features[selected_indices]
    selected_labels = y_test.iloc[selected_indices].reset_index(drop=True)

    # Get feature names
    feature_names = vectorizer.get_feature_names_out()

    # Create directory to save interpretability plots and highlighted reviews
    plots_dir = os.path.join(DATA_DIR, 'interpretability_plots')
    os.makedirs(plots_dir, exist_ok=True)
    highlighted_dir = os.path.join(plots_dir, 'highlighted_reviews')
    os.makedirs(highlighted_dir, exist_ok=True)

    # Global Interpretability for LogisticRegression
    if isinstance(model, LogisticRegression):
        if hasattr(model, 'coef_'):
            coefficients = model.coef_[0]
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
    if isinstance(model, LogisticRegression):
        lime_explainer = LimeTextExplainer(class_names=['Negative', 'Positive'], random_state=42)

        for i in range(len(selected_reviews)):
            review = selected_reviews.iloc[i]
            label = selected_labels.iloc[i]
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

            explanation_html = explanation.as_html()
            explanation_path = os.path.join(plots_dir, f'review_{i+1}_lime.html')
            with open(explanation_path, 'w', encoding='utf-8') as f:
                f.write(explanation_html)

            top_features = explanation.as_list(label=1)[:10]
            top_words = [word for word, weight in top_features]

            filename_txt = f'review_{i+1}_{"positive" if label ==1 else "negative"}_lime.txt'
            filepath_txt = os.path.join(plots_dir, filename_txt)

            with open(filepath_txt, 'w') as f:
                f.write(f"Review {i+1} - {'Positive' if label ==1 else 'Negative'} Sentiment\n")
                f.write(f"Original review:\n{review_text}\n\n")
                f.write("Top contributing words (LIME):\n")
                for word, weight in top_features:
                    f.write(f"{word}: {weight:.4f}\n")

            filename_html = f'review_{i+1}_{"positive" if label ==1 else "negative"}_highlighted.html'
            filepath_html = os.path.join(highlighted_dir, filename_html)
            highlight_important_words(review_text, top_words, filepath_html)

            print(f"\nReview {i+1} - {'Positive' if label ==1 else 'Negative'} Sentiment")
            print(f"LIME explanation saved to {explanation_path}")
            print(f"Top contributing words saved to {filepath_txt}")
            print(f"Highlighted review saved to {filepath_html}")

    print(f"\nInterpretability visualizations saved in the directory: {plots_dir}")
    print(f"Highlighted reviews saved in the directory: {highlighted_dir}")


def main():
    warnings.filterwarnings('ignore')

    DATA_DIR = 'p3_data'
    num_splits = 5

    for i in range(num_splits):
        split_dir = os.path.join(DATA_DIR, f"split_{i+1}")
        train_path = os.path.join(split_dir, "train.csv")
        test_path = os.path.join(split_dir, "test.csv")
        test_y_path = os.path.join(split_dir, "test_y.csv")

        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
        y_test = pd.read_csv(test_y_path)['sentiment']

        X_train = train.drop(columns=['id', 'sentiment', 'review'])
        y_train = train['sentiment']
        X_test = test.drop(columns=['id', 'review'])

        model = train_logistic(X_train, y_train)
        y_pred_proba = predict(model, X_test)

        auc_baseline = roc_auc_score(y_test, y_pred_proba)
        print(f"Baseline Logistic Regression AUC in split {i+1}: {auc_baseline:.3f}")

        submission = pd.DataFrame({
            'id': test['id'],
            'prob': y_pred_proba
        })
        submission.to_csv(os.path.join(split_dir, "mysubmission.csv"), index=False)

        test_auc_score = roc_auc_score(y_test, y_pred_proba)
        print(f"Best AUC Score on Test Data in split {i+1}: {test_auc_score:.3f}")

    split = 1
    split_dir = os.path.join(DATA_DIR, f"split_{split}")
    train_path = os.path.join(split_dir, "train.csv")
    test_path = os.path.join(split_dir, "test.csv")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    stops = set(stopwords.words("english"))

    clean_train_reviews = []
    num_train_reviews = train["review"].size
    print("\nCleaning and parsing the training set movie reviews...\n")
    for i in range(num_train_reviews):
        if (i+1) % 1000 == 0:
            print(f"Review {i+1} of {num_train_reviews}")
        clean_train_reviews.append(review_to_words(train["review"][i], stops))

    print("\nCreating the bag of words...\n")
    vectorizer_path = os.path.join(split_dir, "vectorizer.pkl")
    if os.path.exists(vectorizer_path):
        print("Loading existing vectorizer...")
        vectorizer = pd.read_pickle(vectorizer_path)
        train_data_features = vectorizer.transform(clean_train_reviews)
    else:
        print("Creating and fitting new vectorizer...")
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
        train_data_features = vectorizer.fit_transform(clean_train_reviews)
        pd.to_pickle(vectorizer, vectorizer_path)

    train_data_features = train_data_features.toarray()

    vocab = vectorizer.get_feature_names_out()
    print("\nVocabulary:")
    print(vocab)

    dist = np.sum(train_data_features, axis=0)
    print("\nWord Counts:")
    for tag, count in zip(vocab, dist):
        print(f"{count} {tag}")

    print("\ntraining the logistic regression...\n")
    model_path = os.path.join(split_dir, "logistic_model.pkl")
    if os.path.exists(model_path):
        print("Loading existing logistic regression model...")
        logistic_model = pd.read_pickle(model_path)
    else:
        print("Training new logistic regression model...")
        logistic_model = train_logistic(train_data_features, train["sentiment"])
        pd.to_pickle(logistic_model, model_path)

    clean_test_reviews = []
    num_test_reviews = len(test["review"])
    print("\nCleaning and parsing the test set movie reviews...\n")
    for i in range(num_test_reviews):
        if (i+1) % 1000 == 0:
            print(f"Review {i+1} of {num_test_reviews}")
        clean_review = review_to_words(test["review"][i], stops)
        clean_test_reviews.append(clean_review)

    test_data_features = vectorizer.transform(clean_test_reviews)
    test_data_features = test_data_features.toarray()

    y_pred_logistic = predict(logistic_model, test_data_features)
    submission_logistic = pd.DataFrame({
        'id': test['id'],
        'prob': y_pred_logistic
    })
    submission_logistic.to_csv(os.path.join(split_dir, "logistic_submission.csv"), index=False)
    print(f"\nLogistic Regression predictions saved for split {split}.")

    print("\nStarting interpretability analysis...\n")
    y_test_split1 = pd.read_csv(os.path.join(split_dir, "test_y.csv"))['sentiment']
    interpret_model(
        model=logistic_model,
        vectorizer=vectorizer,
        test_data_features=test_data_features,
        test_reviews=test['review'],
        y_test=y_test_split1,
        DATA_DIR=split_dir
    )


if __name__ == "__main__":
    main()
