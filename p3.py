# p3.py

import re
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import CountVectorizer

from bs4 import BeautifulSoup, NavigableString
import nltk
from nltk.corpus import stopwords

def train_logistic(X_train, y_train):
    model = LogisticRegression(
        penalty='elasticnet',
        solver='saga',
        l1_ratio=0,  # Ridge Regression
        C=5,
        max_iter=1000
    )
    model.fit(X_train, y_train)
    return model

def predict(model, X_test):
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probabilities for positive sentiment
    return y_pred_proba

def review_to_words(raw_review, stops):
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
    """
    from bs4 import BeautifulSoup, NavigableString
    import re

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
                        lambda match: f"<span style='background-color: yellow'>{match.group(0)}</span>",
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
    Interpret model predictions for 5 positive and 5 negative reviews using feature importance.
    """
    # Select 5 positive and 5 negative reviews
    np.random.seed(42)  # For reproducibility
    positive_indices = np.where(y_test == 1)[0]
    negative_indices = np.where(y_test == 0)[0]

    selected_positive = np.random.choice(positive_indices, size=5, replace=False)
    selected_negative = np.random.choice(negative_indices, size=5, replace=False)
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

    # For Random Forest, we can use feature importances
    if isinstance(model, RandomForestClassifier):
        # Get feature importances
        importances = model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        })
        feature_importance = feature_importance.sort_values('importance', ascending=False)

        # Plot top 20 most important features
        plt.figure(figsize=(12, 6))
        sns.barplot(data=feature_importance.head(20), x='importance', y='feature')
        plt.title('Top 20 Most Important Features')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'feature_importance.png'))
        plt.close()

    # Analyze individual reviews
    for i in range(len(selected_reviews)):
        review = selected_reviews.iloc[i]
        label = selected_labels.iloc[i]
        features = selected_features[i]

        # Get the words present in this review
        words_present = features > 0
        words = feature_names[words_present]

        if isinstance(model, RandomForestClassifier):
            # Create a text explainer for this specific review
            te = TextExplainer(random_state=42,
                             vectorizer=vectorizer,
                             clf=RandomForestClassifier(n_estimators=100, random_state=42))

            # Fit the explainer on this specific prediction
            te.fit(review, model.predict_proba)

            # Get local explanation for this review
            explanation = te.explain_prediction(top_targets=1, target_names=['negative', 'positive'])

            # Extract the weights for each word in this specific prediction
            local_weights = []
            local_words = []
            for feature in explanation.targets[0].feature_weights.pos:
                local_words.append(feature.feature)
                local_weights.append(feature.weight)

            review_words_importance = pd.DataFrame({
                'word': local_words,
                'importance': local_weights
            }).sort_values('importance', ascending=False)

            # Get top 10 locally important words
            top_words = review_words_importance.head(10)['word'].tolist()

            # Save top words and their importance for this review
            filename_txt = f'review_{i+1}_{"positive" if label ==1 else "negative"}.txt'
            filepath_txt = os.path.join(plots_dir, filename_txt)

            with open(filepath_txt, 'w') as f:
                f.write(f"Review {i+1} - {'Positive' if label ==1 else 'Negative'} Sentiment\n")
                f.write(f"Original review:\n{review}\n\n")
                f.write("Top contributing words:\n")
                for _, row in review_words_importance.head(10).iterrows():
                    f.write(f"{row['word']}: {row['importance']:.4f}\n")


            # Highlight important words in the original review
            filename_html = f'review_{i+1}_{"positive" if label ==1 else "negative"}_highlighted.html'
            filepath_html = os.path.join(highlighted_dir, filename_html)
            highlight_important_words(review, top_words, filepath_html)

            print(f"\nReview {i+1} - {'Positive' if label ==1 else 'Negative'} Sentiment")
            print(f"Analysis saved to {filepath_txt}")
            print(f"Highlighted review saved to {filepath_html}")

    print(f"\nInterpretability visualizations saved in the directory: {plots_dir}")
    print(f"Highlighted reviews saved in the directory: {highlighted_dir}")

def main():
    warnings.filterwarnings('ignore')
    nltk.download('stopwords')

    DATA_DIR = 'p3_data'
    num_splits = 5

    # Part 1: Logistic Regression on Multiple Splits
    for i in range(num_splits):
        train_path = os.path.join(DATA_DIR, f"split_{i+1}", "train.csv")
        test_path = os.path.join(DATA_DIR, f"split_{i+1}", "test.csv")
        test_y_path = os.path.join(DATA_DIR, f"split_{i+1}", "test_y.csv")

        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
        y_test = pd.read_csv(test_y_path)['sentiment']

        X_train = train.drop(columns=['id', 'sentiment', 'review'])
        y_train = train['sentiment']
        X_test = test.drop(columns=['id', 'review'])

        model = train_logistic(X_train, y_train)
        y_pred_proba = predict(model, X_test)
        auc_baseline = roc_auc_score(y_train, y_pred_proba)
        print(f"Baseline Logistic Regression AUC in split {i+1}: {auc_baseline:.3f}")

        submission = pd.DataFrame({
            'id': test['id'],
            'prob': y_pred_proba
        })
        submission.to_csv(os.path.join(DATA_DIR, f"split_{i+1}", "mysubmission.csv"), index=False)

        test_auc_score = roc_auc_score(y_test, y_pred_proba)
        print(f"Best AUC Score on Test Data in split {i+1}: {test_auc_score:.3f}")

    # Part 2: Random Forest with Bag of Words on Split 1
    split = 1
    train_path = os.path.join(DATA_DIR, f"split_{split}", "train.csv")
    test_path = os.path.join(DATA_DIR, f"split_{split}", "test.csv")

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
    train_data_features = train_data_features.toarray()

    vocab = vectorizer.get_feature_names_out()
    print("\nVocabulary:")
    print(vocab)

    dist = np.sum(train_data_features, axis=0)
    print("\nWord Counts:")
    for tag, count in zip(vocab, dist):
        print(f"{count} {tag}")

    print("\nTraining the random forest...")
    forest = RandomForestClassifier(n_estimators=100, random_state=42)
    forest.fit(train_data_features, train["sentiment"])

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

    # Make predictions with the Random Forest
    y_pred_forest = forest.predict_proba(test_data_features)[:, 1]
    submission_forest = pd.DataFrame({
        'id': test['id'],
        'prob': y_pred_forest
    })
    submission_forest.to_csv(os.path.join(DATA_DIR, f"split_{split}", "forest_submission.csv"), index=False)
    print(f"\nRandom Forest predictions saved for split {split}.")

    # Interpretability analysis on Split 1 Random Forest
    print("\nStarting interpretability analysis...\n")
    y_test_split1 = pd.read_csv(os.path.join(DATA_DIR, f"split_{split}", "test_y.csv"))['sentiment']
    interpret_model(
        model=forest,
        vectorizer=vectorizer,
        test_data_features=test_data_features,
        test_reviews=test['review'],
        y_test=y_test_split1,
        DATA_DIR=os.path.join(DATA_DIR, f"split_{split}")
    )

if __name__ == "__main__":
    main()
