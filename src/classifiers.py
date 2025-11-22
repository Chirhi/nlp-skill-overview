from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

def get_classifiers(le=None):
    if le == None:
        raise ValueError("Требуется LabelEncoder для автоматического подсчёта классов")

    classifiers = [
        # ('Naive Bayes', MultinomialNB()), # требует только bag-of-words или tf-idf, а планируются разные векторизации с отслеживанием метрик всех моделей. Имеет худший результат. Можно исключить.
        ("logistic_regression", LogisticRegression(max_iter=1000, 
                                                random_state=1, 
                                                class_weight='balanced')), # балансируем, так как слишком редкие классы
        ("svm", LinearSVC(max_iter=2000, 
                        random_state=1, 
                        class_weight='balanced')),
        ("decision_tree", DecisionTreeClassifier(max_depth=6, # попытался ограничить глубину деревьев, так как кажется слишком сложные модели для такой задачи и идёт переобучение
                                                random_state=1,
                                                class_weight='balanced')),
        ("random_forest", RandomForestClassifier(max_depth=6,
                                                random_state=1,
                                                class_weight='balanced')),
        # ("gradient_boosting", GradientBoostingClassifier(random_state=1, # слишком долго обучается, особенно на кросс-валидации, а качество модели худшее из всех бустинг моделей
        #                                                 max_depth=6,
        #                                                 n_estimators=100,)),
        ("histgradientboosting", HistGradientBoostingClassifier(max_iter=100, # тот же gradient boosting, но с оптимизациями и идеей от LightGBM. Обучается сравнительно долго
                                                                max_depth=6,
                                                                random_state=1,
                                                                class_weight='balanced')),
        ("catboost", CatBoostClassifier(verbose=0, # обучается сравнительно долго
                                        task_type='GPU',
                                        devices='0',
                                        random_state=1,
                                        n_estimators=100,
                                        max_depth=6,
                                        loss_function='MultiClass',
                                        auto_class_weights='Balanced',
                                        train_dir=None)), # всё равно сохраняет папку
        ("xgboost", XGBClassifier(objective='multi:softmax',
                                device='cuda',
                                # updater='grow_gpu_hist',
                                tree_method='hist',
                                random_state=1, 
                                n_estimators=100,
                                learning_rate=0.1,
                                max_depth=6,
                                num_class=len(le.classes_))),
        ("lightgbm", LGBMClassifier(objective='multiclass',
                                    device='gpu',
                                    n_estimators=100,
                                    max_depth=6,
                                    num_class=len(le.classes_),
                                    class_weight='balanced',
                                    random_state=1,
                                    verbosity=-1))
    ]
    
    return classifiers