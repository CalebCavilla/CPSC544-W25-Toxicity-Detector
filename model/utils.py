# Utility file for training models and stuff
import numpy as np
from scipy.stats import uniform, loguniform, randint
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.combine import SMOTETomek

# Random seed
RANDOM_STATE = 42

# Parameter distributions for HalvingRandomSearchCV
rf_param_dist = {
    'n_estimators': randint(50, 500),
    'max_depth': [None] + list(randint(5, 50).rvs(10, random_state=RANDOM_STATE)),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True, False],
    'class_weight': ['balanced', 'balanced_subsample', None]
}

gb_param_dist = {
    'n_estimators': randint(50, 500),
    'learning_rate': loguniform(0.001, 0.5),
    'max_depth': randint(3, 15),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'subsample': uniform(0.6, 0.4),
    'max_features': ['sqrt', 'log2', None]
}

nb_param_dist = {
    'alpha': loguniform(1e-5, 10),
    'fit_prior': [True, False]
}

svm_param_dist = {
    'C': loguniform(0.1, 100),
    'gamma': loguniform(1e-5, 1),
    'kernel': ['linear', 'rbf'],
    'probability': [True],
    'class_weight': ['balanced', None]
}

xgb_param_dist = {
    'n_estimators': randint(50, 500),
    'max_depth': randint(3, 15),
    'learning_rate': loguniform(0.001, 0.5),
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.6, 0.4),
    'gamma': uniform(0, 5),
    'min_child_weight': randint(1, 10)
}

lgbm_param_dist = {
    'n_estimators': randint(50, 500),
    'max_depth': randint(3, 15),
    'learning_rate': loguniform(0.001, 0.5),
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.6, 0.4),
    'num_leaves': randint(20, 200)
}

et_param_dist = {
    'n_estimators': randint(50, 500),
    'max_depth': [None] + list(randint(5, 50).rvs(10, random_state=RANDOM_STATE)),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True, False],
    'class_weight': ['balanced', 'balanced_subsample', None]
}

ada_param_dist = {
    'n_estimators': randint(50, 500),
    'learning_rate': loguniform(0.001, 1),
    'algorithm': ['SAMME', 'SAMME.R']
}


def evaluate_smote_methods(X_train, y_train, X_test, y_test, random_state=RANDOM_STATE):
    """Evaluate different SMOTE methods on a small subset of data -> Returns best method to use for training full data"""
    methods = {
        'original': None,
        'smote': SMOTE(random_state=random_state),
        'borderline_smote': BorderlineSMOTE(random_state=random_state),
        'adasyn': ADASYN(random_state=random_state),
        'smotetomek': SMOTETomek(random_state=random_state)
    }

    # Use 10% subset for quick evaluation
    sample_size = min(5000, int(len(y_train) * 0.1))
    indices = np.random.RandomState(random_state).choice(
        len(y_train), sample_size, replace=False,
        p=None if np.unique(y_train).size <= 1 else None
    )
    X_sample, y_sample = X_train.iloc[indices], y_train.iloc[indices]

    results = {}
    base_model = LogisticRegression(max_iter=1000, random_state=random_state)

    for name, method in methods.items():
        X_res, y_res = (X_sample, y_sample) if method is None else method.fit_resample(X_sample, y_sample)

        base_model.fit(X_res, y_res)
        y_pred = base_model.predict(X_test)
        f1 = f1_score(y_test, y_pred)

        results[name] = {
            'f1_score': f1,
            'resampled_shape': X_res.shape,
            'class_distribution': np.bincount(y_res)
        }

        print(f"{name} - F1: {f1:.4f}, Shape: {X_res.shape}, Pos/Neg: {np.bincount(y_res)}")

    best_method = max(results.items(), key=lambda x: x[1]['f1_score'])[0]
    print(f"\nBest method: {best_method} with F1: {results[best_method]['f1_score']:.4f}")

    return results, best_method


def create_stacking_ensemble(base_models, meta_learner=None, cv=5, n_jobs=-1):
    """Create a stacking ensemble from base models"""
    from sklearn.ensemble import StackingClassifier

    if meta_learner is None:
        meta_learner = LogisticRegression(max_iter=1000)

    estimators = [(name, model) for name, model in base_models.items()]

    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=meta_learner,
        cv=cv,
        n_jobs=n_jobs,
        passthrough=False
    )

    return stacking


def create_weighted_voting_ensemble(base_models, validation_scores, voting='soft', n_jobs=-1):
    """Create a weighted voting ensemble based on validation scores"""
    from sklearn.ensemble import VotingClassifier

    estimators = [(name, model) for name, model in base_models.items()]
    weights = [validation_scores.get(name, 1.0) for name, _ in estimators]

    voting = VotingClassifier(
        estimators=estimators,
        voting=voting,
        weights=weights,
        n_jobs=n_jobs
    )

    return voting