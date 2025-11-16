from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

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


