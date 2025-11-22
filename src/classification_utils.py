# Требовалось перенести функции в отдельные .py, так как есть куча разных экспериментов по отдельным .ipynb

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from scipy.sparse import load_npz
from imblearn.pipeline import Pipeline as ImbPipeline
import numpy as np
import mlflow
import pickle

def split_data(df, text_column='lemmatized_text', label_column='answer', test_size=0.3, random_state=1):
    """
    Разделяет данные на тренировочные и тестовые

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame с данными
    text_column : str, optional
        Имя колонки с текстом, by default 'lemmatized_text'
    label_column : str, optional
        Имя колонки с метками, by default 'answer'
    test_size : float, optional
        Размер тестового набора, by default 0.3
    random_state : int, optional
        Случайный сид, by default 1

    Returns
    -------
    tuple
        Кортеж текста и меток из тренировочных и тестовых данных (X_train, X_test, y_train, y_test)
    """
    X = df[text_column]
    y = df[label_column]

    # Может стоит попробовать Stratified K-Fold, раз классы не сбалансированы, посмотреть как как модели стабильны
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y
    )

    return X_train, X_test, y_train, y_test

def encode_labels(y_train, y_test):
    """
    Преобразует текстовые метки в числовые с помощью LabelEncoder

    Parameters
    ----------
    y_train : pd.Series
        Тренировочные метки
    y_test : pd.Series
        Тестовые метки

    Returns
    -------
    tuple
        Кортеж из закодированных тренировочных меток (y_train_enc),
        закодированных тестовых меток (y_test_enc) и обученного LabelEncoder (le).
    """
    le = LabelEncoder()

    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    return y_train_enc, y_test_enc, le

def load_np_arrays(train_path, test_path):
    """
    Берёт вектора из npy или npz файла

    Parameters
    ----------
    train_path : string
        Путь к файлу тренировочных данных
    test_path : string
        Путь к файлу тестовых данных

    Returns
    -------
    tuple
        Векторы np.ndarray тренировочных и тестовых данных
    """
    if train_path.endswith('.npz'):
        X_train_vec = load_npz(train_path).toarray()
        X_test_vec = load_npz(test_path).toarray()
    else: # .npy
        X_train_vec = np.load(train_path)
        X_test_vec = np.load(test_path)

    return X_train_vec, X_test_vec

def crossval_report(classifier, 
                    X_train_vec, y_train, y_train_enc, le, 
                    pipeline=None, n_splits=5):
    """
    Репорт кросс-валидации модели

    Parameters
    ----------
    classifier : tuple
        Метод классификации
    X_train_vec : np.ndarray
        Векторизированные тренировочные данные
    y_train : pd.Series
        Тренировочные метки
    y_train_enc : np.ndarray
        Закодированные тренировочные метки
    le : sklearn.LabelEncoder
        Обученный LabelEncoder
    pipeline : sklearn.Pipeline or imblearn.Pipeline, optional
        Метод пайплайна, by default None
    n_splits : int, optional
        Количество фолдов для кросс-валидации, by default 5

    Returns
    -------
    dict
        Репорт кросс-валидации c метриками по классам и усреднёнными метриками
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1)

    if pipeline is not None:
        model = pipeline
    else:
        model = classifier
    
    y_pred_cv = cross_val_predict(model, X_train_vec, y_train_enc, cv=skf, method='predict')
    y_pred_cv_labels = le.inverse_transform(y_pred_cv)
    return classification_report(y_train, y_pred_cv_labels, output_dict=True, zero_division=0)

def log_report(report, vec_name, classifier_name, resampler_name):
    """
    Передача параметров, метрик в MLFlow

    Parameters
    ----------
    report : dict
        Присовенная переменная classification_report с метриками
    vec_name : string
        Название векторизатора
    classifier_name : string
        Название классификатора
    resampler_name : string
        Название балансировщика
    """
    with mlflow.start_run(run_name=f"{vec_name}_{classifier_name}_{resampler_name}"):
        mlflow.log_param('vectorizer', vec_name)
        mlflow.log_param('classifier', classifier_name)
        mlflow.log_param('resampler', resampler_name)

        # # Если пригодятся параметры
        # mlflow.log_param("n_splits", n_splits)
        
        for key, value in report.items():
            if isinstance(value, dict):
                if key in ['macro avg', 'weighted avg']:
                    for metric_name, metric_val in value.items():
                        mlflow.log_metric(f'{key}_{metric_name}', metric_val)
                else:
                    for metric_name, metric_val in value.items():
                        mlflow.log_metric(f'class_{key}_{metric_name}', metric_val)
            else:
                mlflow.log_metric(f'{key}', value)

def fit_and_save(classifier, X, y_enc, pipeline=None, model_path='model.pkl'):
    if pipeline:
        model = pipeline
    else:
        model = classifier
    model.fit(X, y_enc)

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    return model