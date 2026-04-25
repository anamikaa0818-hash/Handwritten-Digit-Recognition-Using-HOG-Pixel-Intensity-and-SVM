"""
Model Training Module: SVM, KNN, Logistic Regression, Random Forest
Authors: Anamika, Ardra, Anugopal
"""

import numpy as np
import time
import pickle
import os

from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


class ModelTrainer:
    """
    Train and manage multiple machine learning models.
    """

    def __init__(self, random_state=42):

        self.random_state = random_state

        self.models = {}
        self.training_times = {}
        self.best_params = {}

    # =========================================================
    # SVM
    # =========================================================
    def train_svm_rbf(
        self,
        X_train,
        y_train,
        C=1.0
    ):
        """
        Train SVM classifier.
        """

        print("\n=== TRAINING SVM ===")

        start_time = time.time()

        model = LinearSVC(
            C=C,
            random_state=self.random_state,
            max_iter=5000
        )

        model.fit(X_train, y_train)

        training_time = time.time() - start_time

        self.models['SVM'] = model
        self.training_times['SVM'] = training_time

        print(f"✓ SVM trained in {training_time:.2f}s")
        print(f"  Hyperparameters: C={C}")

        return model

    # =========================================================
    # KNN
    # =========================================================
    def train_knn(
        self,
        X_train,
        y_train,
        n_neighbors=5
    ):
        """
        Train K-Nearest Neighbors classifier.
        """

        print("\n=== TRAINING KNN ===")

        start_time = time.time()

        model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            n_jobs=-1
        )

        model.fit(X_train, y_train)

        training_time = time.time() - start_time

        self.models['KNN'] = model
        self.training_times['KNN'] = training_time

        print(f"✓ KNN trained in {training_time:.2f}s")
        print(f"  Hyperparameters: n_neighbors={n_neighbors}")

        return model

    # =========================================================
    # LOGISTIC REGRESSION
    # =========================================================
    def train_logistic_regression(
        self,
        X_train,
        y_train,
        C=1.0,
        max_iter=1000
    ):
        """
        Train Logistic Regression classifier.
        """

        print("\n=== TRAINING LOGISTIC REGRESSION ===")

        start_time = time.time()

        model = LogisticRegression(
            C=C,
            max_iter=max_iter,
            solver='saga',
            random_state=self.random_state,
            n_jobs=-1
        )

        model.fit(X_train, y_train)

        training_time = time.time() - start_time

        self.models['LogisticRegression'] = model
        self.training_times['LogisticRegression'] = training_time

        print(f"✓ Logistic Regression trained in {training_time:.2f}s")
        print(f"  Hyperparameters: C={C}, max_iter={max_iter}")

        return model

    # =========================================================
    # RANDOM FOREST
    # =========================================================
    def train_random_forest(
        self,
        X_train,
        y_train,
        n_estimators=50,
        max_depth=None
    ):
        """
        Train Random Forest classifier.
        """

        print("\n=== TRAINING RANDOM FOREST ===")

        start_time = time.time()

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=self.random_state,
            n_jobs=-1
        )

        model.fit(X_train, y_train)

        training_time = time.time() - start_time

        self.models['RandomForest'] = model
        self.training_times['RandomForest'] = training_time

        print(f"✓ Random Forest trained in {training_time:.2f}s")

        print(
            f"  Hyperparameters: "
            f"n_estimators={n_estimators}, "
            f"max_depth={max_depth}"
        )

        return model

    # =========================================================
    # TRAIN ALL MODELS
    # =========================================================
    def train_all_models(
        self,
        X_train,
        y_train
    ):
        """
        Train all classifiers.
        """

        print("=" * 50)
        print("TRAINING ALL MODELS")
        print("=" * 50)

        self.train_svm_rbf(X_train, y_train)

        self.train_knn(
            X_train,
            y_train,
            n_neighbors=5
        )

        self.train_logistic_regression(
            X_train,
            y_train
        )

        self.train_random_forest(
            X_train,
            y_train
        )

        print("\n✓ All models trained successfully!")

        return self.models

    # =========================================================
    # HYPERPARAMETER TUNING
    # =========================================================
    def hyperparameter_tuning_svm(
        self,
        X_train,
        y_train,
        C_values=[0.1, 1, 10]
    ):
        """
        Perform Grid Search for SVM.
        """

        print("\n=== SVM HYPERPARAMETER TUNING ===")

        param_grid = {
            'C': C_values
        }

        grid_search = GridSearchCV(
            LinearSVC(
                random_state=self.random_state,
                max_iter=5000
            ),
            param_grid=param_grid,
            cv=3,
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train, y_train)

        print(f"✓ Best parameters: {grid_search.best_params_}")
        print(f"  Best CV score: {grid_search.best_score_:.4f}")

        self.models['SVM_Tuned'] = grid_search.best_estimator_

        self.best_params['SVM'] = grid_search.best_params_

        return grid_search.best_estimator_

    # =========================================================
    # SAVE MODELS
    # =========================================================
    def save_models(
        self,
        save_dir='../models/'
    ):
        """
        Save trained models to disk.
        """

        os.makedirs(save_dir, exist_ok=True)

        for model_name, model in self.models.items():

            filepath = os.path.join(
                save_dir,
                f"{model_name}.pkl"
            )

            with open(filepath, 'wb') as f:
                pickle.dump(model, f)

            print(f"✓ {model_name} saved to {filepath}")
