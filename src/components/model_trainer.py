import os
import sys
from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.metrics import r2_score
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
import subprocess  # For tracking DVC dataset version
from pathlib import Path
from sklearn.model_selection import GridSearchCV

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "best_model.pkl")
    project_root: Path = Path(__file__).resolve().parent.parent.parent  # Adjust path as needed
    train_data_path: Path = project_root / "data/processed/train_transformed.npy"
    test_data_path: Path = project_root / "data/processed/test_transformed.npy"

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self):
        try:
            logging.info("Loaded train_array and test_array from DataTransformation")
            train_array = np.load(self.model_trainer_config.train_data_path)
            test_array = np.load(self.model_trainer_config.test_data_path)
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            logging.info("Starting RandomForestRegressor with predefined hyperparameters.")
            random_forest_params = {
                'bootstrap': False,
                'criterion': 'entropy',
                'max_depth': 20,
                'max_features': 'sqrt',
                'min_samples_leaf': 5,
                'min_samples_split': 2,
                'n_estimators': 75
            }
            '''
            random_forest_params = {
            'n_estimators': [25, 50, 75, 100],
            'max_depth': [2, 3, 5, 10, 20],
            'min_samples_leaf': [5, 10, 20, 50, 100],
            'min_samples_split': [2, 5, 10],
            'criterion': ["absolute_error", "squared_error"],
            'max_features': ['sqrt'],
            'bootstrap': [True, False]
            }

            #Commented out GridSearchCV for future use
            logging.info("Starting RandomForestRegressor hyperparameter tuning with GridSearchCV.")
            grid_search = GridSearchCV(estimator=RandomForestRegressor(), param_grid=random_forest_params,
                                       scoring='r2', cv=5, n_jobs=-1)
            
            grid_search.fit(X_train, y_train)
            
            best_params = grid_search.best_params_
            logging.info(f"Best Hyperparameters: {best_params}")
            
            best_rf_model = grid_search.best_estimator_
            logging.info(f"Best RandomForestRegressor model: {best_rf_model}")
            '''
            # Using predefined hyperparameters
            best_rf_model = RandomForestRegressor(**random_forest_params)
            best_rf_model.fit(X_train, y_train)
            logging.info(f"Trained RandomForestRegressor model with predefined hyperparameters: {random_forest_params}")

            # Save Best RandomForestRegressor Model Locally in `artifacts/`
            save_object(self.model_trainer_config.trained_model_file_path, best_rf_model)

            print(f"Best RandomForestRegressor model saved to artifacts/{self.model_trainer_config.trained_model_file_path}")

            # Evaluate the best model on the test set
            y_pred = best_rf_model.predict(X_test)
            test_r2_score = r2_score(y_test, y_pred)
            logging.info(f"Test R² score: {test_r2_score}")

            return test_r2_score

        except Exception as e:
            raise CustomException(e, sys)