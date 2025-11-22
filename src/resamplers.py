from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN, BorderlineSMOTE
from imblearn.combine import SMOTEENN

# В идеале бы лучше проверить каждый параметр через grid-search, но это бы потребовало очень много времени, учитывая, что у меня 280 всего сочетаний
# Может и стоит попробовать на только одной модели через кросс-валидацию
def get_resamplers():
    balancing_methods = [
        ("smoteen", SMOTEENN(sampling_strategy='auto',
                              smote=SMOTE(k_neighbors=5,random_state=1),
                              random_state=1,
                              n_jobs=3)),
        ("borderlinesmote", BorderlineSMOTE(sampling_strategy='auto',
                                            k_neighbors=5, # из-за слишком редких маленьких классов нужно уменьшить ближайших соседей
                                            random_state=1)),
        ("smote", SMOTE(sampling_strategy='auto',
                        k_neighbors=5, 
                        random_state=1)),
        ("randomoversampler", RandomOverSampler(sampling_strategy='auto',
                                                random_state=1)),
        ("adasyn", ADASYN(sampling_strategy='auto',
                          n_neighbors=5,
                          random_state=1))
    ]

    return balancing_methods