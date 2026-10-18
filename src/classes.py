import os

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline

class Batch:
    def __init__(self, batch_id, ph_opt, temperature_opt):
        self.batch_id = batch_id
        self.ph_opt = ph_opt
        self.temperature_opt = temperature_opt

    def display_info(self):
        print(f"Batch ID: {self.batch_id:4d} | "
              f"Optimal pH: {self.ph_opt:4.1f} | "
              f"Optimal Temperature: {self.temperature_opt:4.1f} C")

class CellGrowthModel:
    def __init__(self):
        self.df = pd.read_csv(os.path.join("datasets", "dataset_cell_growth_4factor.csv"))
    def prepare_data(self):
        feature = ["Temperature [C]", "pH", "Glucose [mM]", "Lactate [mM]"]
        target = "Specific Growth Rate [d^-1]"

        x = self.df.loc[:, feature].to_numpy()
        y = self.df.loc[:, target].to_numpy()

        return x, y

    def split_data(self):
        x, y = self.prepare_data()
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

        return x_train, x_test, y_train, y_test

    def train_model(self):
        x_train, x_test, y_train, y_test = self.split_data()

        self.scaler = StandardScaler()
        self.scaler.fit(x_train)

        x_train_std = self.scaler.transform(x_train)
        x_test_std = self.scaler.transform(x_test)

        self.model = LinearRegression()
        self.model.fit(x_train_std, y_train)

        return x_train_std, x_test_std, y_train, y_test

    def evaluate_model(self):
        x, y = self.prepare_data()

        model = make_pipeline(StandardScaler(), LinearRegression())

        cv_results = cross_validate(model, x, y, cv=5, scoring=("neg_root_mean_squared_error", "r2"), return_train_score=True)

        train_rmse = -cv_results["train_neg_root_mean_squared_error"]
        test_rmse = -cv_results["test_neg_root_mean_squared_error"]
        train_r2 = cv_results["train_r2"]
        test_r2 = cv_results["test_r2"]

        results = {"Model": "Linear Regression", "Training RMSE": np.mean(train_rmse), "Testing RMSE": np.mean(test_rmse), "Training R2": np.mean(train_r2), "Testing R2": np.mean(test_r2)}

        return results

    def evaluate_polynomial_model(self, degree=2):
        x, y = self.prepare_data()

        model = make_pipeline(StandardScaler(), PolynomialFeatures(degree=degree, include_bias=False), LinearRegression())

        cv_results = cross_validate(model, x, y, cv=5, scoring=("neg_root_mean_squared_error", "r2"), return_train_score=True)

        train_rmse = -cv_results["train_neg_root_mean_squared_error"]
        test_rmse = -cv_results["test_neg_root_mean_squared_error"]
        train_r2 = cv_results["train_r2"]
        test_r2 = cv_results["test_r2"]

        results = {"Model": f"Polynomial Regression (degree={degree})","Training RMSE": np.mean(train_rmse),"Testing RMSE": np.mean(test_rmse),"Training R2": np.mean(train_r2),"Testing R2": np.mean(test_r2)}

        return results


    def evaluate_knn_model(self):
        x, y = self.prepare_data()

        model = make_pipeline(StandardScaler(), KNeighborsRegressor())
        param_grid = {"kneighborsregressor__n_neighbors": range(1, 81), "kneighborsregressor__weights": ["uniform", "distance"]}

        model_gscv = GridSearchCV(model, param_grid=param_grid, cv=5, scoring="neg_root_mean_squared_error", return_train_score=True)

        model_gscv.fit(x, y)

        best_model = model_gscv.best_estimator_

        cv_results = cross_validate(best_model, x, y, cv=5, scoring=("neg_root_mean_squared_error", "r2"), return_train_score=True)

        train_rmse = -cv_results["train_neg_root_mean_squared_error"]
        test_rmse = -cv_results["test_neg_root_mean_squared_error"]
        train_r2 = cv_results["train_r2"]
        test_r2 = cv_results["test_r2"]

        print("\nKNN Regression (5-fold CV)")
        print("--------------------------")
        print(f"Best parameters: {model_gscv.best_params_}")
        print(f"Training RMSE: {np.mean(train_rmse):.4f} +/- {np.std(train_rmse):.4f} d^-1")
        print(f"Testing RMSE:  {np.mean(test_rmse):.4f} +/- {np.std(test_rmse):.4f} d^-1")
        print(f"Training R2:   {np.mean(train_r2):.4f} +/- {np.std(train_r2):.4f}")
        print(f"Testing R2:    {np.mean(test_r2):.4f} +/- {np.std(test_r2):.4f}")

        results = {"Model": "KNN Regression","Training RMSE": np.mean(train_rmse),"Testing RMSE": np.mean(test_rmse),"Training R2": np.mean(train_r2),"Testing R2": np.mean(test_r2)}

        return results


    def predict_growth(self, temperature, ph, glucose, lactate):
        x, y = self.prepare_data()

        model = make_pipeline(StandardScaler(),PolynomialFeatures(degree=2, include_bias=False), LinearRegression())

        model.fit(x, y)

        new_data = np.array([[temperature, ph, glucose, lactate]])

        predicted_rate = model.predict(new_data)

        results = {"Temperature [C]": temperature,"pH": ph,"Glucose [mM]": glucose,"Lactate [mM]": lactate,"Predicted Growth Rate [d^-1]": predicted_rate[0]}

        return results

    def analyze_condition(self, condition, values):
        x, y = self.prepare_data()
        model = make_pipeline(StandardScaler(),PolynomialFeatures(degree=2, include_bias=False),LinearRegression())

        model.fit(x, y)

        average_conditions = self.df[["Temperature [C]", "pH", "Glucose [mM]", "Lactate [mM]"]].mean()

        results = []

        for value in values:
            test_conditions = average_conditions.copy()
            test_conditions[condition] = value

            new_data = test_conditions.to_numpy().reshape(1, -1)
            predicted_rate = model.predict(new_data)[0]

            results.append({condition: value,"Predicted Growth Rate [d^-1]": predicted_rate})

        return pd.DataFrame(results)

