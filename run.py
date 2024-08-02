import os
import argparse
import json

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, classification_report, balanced_accuracy_score, confusion_matrix
from scipy.stats import boxcox
from catboost import CatBoostClassifier
from catboost.core import CatBoost
import pandas as pd
import numpy as np

def pull_data() -> pd.DataFrame:
    """
    Pull Petfinder dataset from google storage

    Returns:
        DataFrame: a pandas DataFrame with Petfinder dataset
    """

    url = 'gs://cloud-samples-data/ai-platform-unified/datasets/tabular/petfinder-tabular-classification.csv'
    df_raw = pd.read_csv(url)
    return df_raw

def preprocess(df: pd.DataFrame, read_encoder_mappings:bool =True) -> pd.DataFrame:
    """
    Preprocess raw Petfinder dataset

    Args:
        df (DataFrame): raw Petfinder dataset
        read_encoder_mappings (bool): set True to read the encoder mappings from disk, False to compute them.

    Returns:
        DataFrame: preprocessed Petfinder dataset.
    """

    columns = ['Type', 'Breed1', 'Gender', 'Color1', 'Color2', 'MaturitySize', 'FurLength', 'Vaccinated', 'Sterilized', 'Health', 'Adopted']

    if read_encoder_mappings:
        with open('artifacts/encoder_mappings.json', 'r') as em_file:
            encoder_mappings = json.load(em_file)
        for col in columns: df[col] = df[col].map(encoder_mappings[col])
    else: 
        encoder_mappings = {}
        encoder_mappings['Adopted'] = {'Yes': 1, 'No': 0}
        for col in columns:
            encoder_mappings[col] =  {val: i for i, val in enumerate(df[col].unique())}
            df[col] = df[col].map(encoder_mappings[col])

        if not os.path.exists('artifacts'):
            os.mkdir('artifacts')

        with open('artifacts/encoder_mappings.json', 'w') as em_file:
            json.dump(encoder_mappings, em_file, indent=4)

    df['Age'] = np.log1p(df['Age'])
    df['Breed1'] = boxcox(df['Breed1'] + 1)[0]
    df.loc[df['Fee'] > 0, 'Fee'] = 1
    df['PhotoAmt'] = np.sqrt(df['PhotoAmt'])
    df.loc[df['Color1'] >= 2, 'Color1']= 2

    return df

def train_catboost(df: pd.DataFrame, use_roc_thresh:bool =False) -> CatBoost:
    """
    Trains a CatBoost model on preprocessed Petfinder data and persists the model on disk

    Args:
        df (DataFrame): preprocessed Petfinder data
        use_roc_thresh (bool): set True to use ROC optimal threshold, False to use CatBoost's threshold

    Returns:
        CatBoost: trained catboost model
    """

    X = df.drop(['Adopted'], axis=1)
    y = df['Adopted']
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42,stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42,stratify=y_temp)

    neg_weight = 1 - sum(y_train == 0)/y_train.shape[0]
    pos_weight = 1 - sum(y_train == 1)/y_train.shape[0]

    model = CatBoostClassifier(
            class_weights={0: neg_weight, 1: pos_weight},
            verbose=0,
            allow_writing_files=False
    )
    model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=200)

    if use_roc_thresh:
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, roc_thresholds = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = roc_thresholds[optimal_idx]
        y_pred = y_pred_proba > optimal_threshold
        model.set_probability_threshold(optimal_threshold)
    else:
        y_pred = model.predict(X_test)

    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)
    cm = cm/cm.sum(axis=1)[:, np.newaxis]

    print(f"Balanced Accuracy: {balanced_accuracy * 100:.2f}%")
    print("Confusion Matrix: ")
    print(cm)
    print(classification_report(y_test, y_pred))
    return model

def infer_catboost(df: pd.DataFrame, df_raw: pd.DataFrame) -> None:
    """
    Performs inference on an input Petfinder DataFrame and returns a new DataFrame with the prediction
    of the catboost model.

    Args:
        df (DataFrame): preprocessed Petfinder data
        df_raw (DataFrame): raw Petfinder dataset.
    """

    X = df.drop(['Adopted'], axis=1)
    y = df['Adopted']

    model = CatBoostClassifier()
    model.load_model('artifacts/model')

    df_raw['Adopted_prediction'] = model.predict(X)
    df_raw['Adopted_prediction'] = df_raw['Adopted_prediction'].map(lambda x: 'Yes' if x == 1 else 'No')
    columns = ['Type', 'Age', 'Breed1', 'Gender',
                'Color1', 'Color2', 'MaturitySize',
                'FurLength', 'Vaccinated', 'Sterilized',
                'Health', 'Fee', 'PhotoAmt', 'Adopted', 'Adopted_prediction']
    df_raw = df_raw[columns]

    return df_raw

def test_correct_predictions():
    values = [[0.0, 1.38, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
            [0.0, 1.79, 0.93, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0]]
    raw_values = [['Cat', 3, 'Tabby', 'Male', 'Black', 'White', 'Small', 'Short', 'No', 'No', 'Healthy', 100, 1, 'Yes'],
                ['Cat', 5, 'Domestic Short Hair', 'Male', 'Black', 'White', 'Medium', 'Short', 'No', 'No', 'Healthy', 30, 4, 'No']]
    columns = ['Type', 'Age', 'Breed1', 'Gender', 'Color1', 'Color2', 'MaturitySize',
               'FurLength', 'Vaccinated', 'Sterilized', 'Health', 'Fee', 'PhotoAmt',
              'Adopted']
    df = pd.DataFrame(values, columns=columns)
    df_raw = pd.DataFrame(raw_values, columns=columns)
    df_results = df_raw.copy()
    df_results['Adopted_prediction'] = ['Yes','No']
    pd.testing.assert_frame_equal(infer_catboost(df, df_raw), df_results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="A script with flag arguments")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--train', action='store_true', help="Train the model")
    group.add_argument('--infer', action='store_true', help="Infer using the model")

    args = parser.parse_args()

    if args.train:
        df = pull_data()
        df = preprocess(df, read_encoder_mappings=False)
        model = train_catboost(df, use_roc_thresh=True)
        model.save_model('artifacts/model')
        print("The model is saved!")
    elif args.infer:
        test_correct_predictions()
        df_raw = pull_data()
        df = preprocess(df_raw.copy(), read_encoder_mappings=True)
        df_results = infer_catboost(df, df_raw)
        if not os.path.exists('output'):
            os.mkdir('output')
        df_results.to_csv('output/results.csv')
        print("The results are saved!")
