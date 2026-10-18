### Cell Growth Monitoring Using Regression Models
The main objective of my project is to implement regression models to predict the specific growth rate of cells based on the temperature, pH, glucose concentration, and lactate concentration data. The resulting model is therefore able to identify which culture conditions promote high cell growth.

#### Highlights:
- Engineered: A program capable of evaluating cell growth.
- Identified: Determined the most effective regression model for the task.
- Programmed: Implemented linear regression, polynomial regression and KNN model. 
- Transformed: Experimental data into model ready datasets.
- Evaluated: Model performance using RMSE and R^2 metrics. 

#### Environment:
- Python 3.13.16
- NumPy 2.5.4
- Pandas 3.0.6
- Scikit-learn 1.9.1

#### **Code Design:**

##### Data loading and preparation
Relevant code: CellGrowthModel.__init__() and prepare_data()

###### What does this stage of the workflow do: 

This section of the code loads the dataset_cell_growth_4factor.csv file into pandas. From their, the prepare_data() section separates the four feature categories of temperature, pH, glucose and lactate concentration with the target variable set as the specific growth rate. The features are stored in x while the target is stored in y NumPy arrays.

###### Why this stage is necessary: 

This stage is necessary as it organizes the input target and clearly defines the feature parameters. 

###### How does this stage contribute to the overall workflow: 

This stage provides the prepared data that will later be used for all four regression approaches implemented later on in the program.

##### Model Training
Relevant code: evaluate_model(), evaluate_polynomial_model(), and evaluate_knn_model()

###### What does this stage of the workflow do: 

I created the program to run four different regression approaches including: linear regression, polynomial regression with degree, polynomial regression with degree 3, and K-nearest neighbors (KNN) regression. 

###### Why this stage is necessary: 

This stage is necessary as the relationship between operating conditions and cell growth may be non-linear and complex. Therefore, using multiple modeling techniques is beneficial as each model is able to predict various non-linear aspects in the data.

###### How does this stage contribute to the overall workflow: 

This stage establishes the different regression approaches that wil be compared in order to select the model suitable regression technique. 

##### Model evaluation and comparison: 
Relevant code: cross_validation(), and RMSE and R^2 parameters obtained in the results of running main.py

###### What does this stage of the workflow do: 

The program uses a five-fold cross validation to repeatedly train and evaluate each model of different subsets of the dataset. The program then calculates key metrics including the root mean squared error (RMSE) and the coefficient of determination (R^2) across the fold which are exported to a results table.

###### Why this stage is necessary: 

The cross validation provides a more reliable estimation of the model performance rather than relying on the train test split it evaluates multiple subsections of the dataset. 

###### How does this stage contribute to the overall workflow: 

The evalution of R^2 and RMSE is crucial in selecting the most suitable model for determining the specific cell growth rate. 

##### Growth rate prediction: 
Relevant code: predict_growth()

###### What does this stage of the workflow do: 
From the results of the model evaluation and comparison I had chosen the degree-2 polynomial regression technique. This portion of the program uses the selected model regression approach and predicts the specific growth rate based on user inputted temperature, pH, glucose and lactate concentration values.

###### Why this stage is necessary:
Evaluating the specific growth rate of the culture is the intended goal of this program and this stage provide qualitative results for further analysis. 

###### How does this stage contribute to the overall workflow: 
This stage of the program completes the process by evaluating the fermentation data to obtain a prediction of the specific cell growth rate. The results are then saved to a table for concrete results.


#### Analysis:

##### Model Selection and Justification 
Four regression approaches were evaluated to determine the most suitable model for predicting specific growth rate for the provided dataset. The four models impleted include: linear regression, polynomial regression with degree 2 and 3, and K-nearest neighbors (KNN) regression. Model selection was primarily based on the results of the coefficient of determination (R^2) and the root mean squared error (RMSE) obtained from five-fold cross validation. In terms of the RMSE metric the best score is zero with lower values indicating a better predictive model. Consequently, the R^2 metric indicates how much variation in specific growth it explained by the mode, therefore, 1 is the best score with higher values indicating better performance. The resulting metrics obtained from each regression model is displayed in the following table:

|Model                           |Training RMSE|Testing RMSE|Training R2|Testing R2|
|--------------------------------|-------------|------------|-----------|----------|
|Linear Regression               |0.1749       |0.1791      |0.1335     |0.0529    |
|Polynomial Regression (degree=2)|0.1452       |0.1545      |0.4025     |0.2882    |
|Polynomial Regression (degree=3)|0.134        |0.158       |0.4909     |0.2525    |
|KNN Regression                  |0.0          |0.1556      |1.0        |0.2833    |

From the summary table it was observed that polynomial regression with degree 2 had the highest testing R^2 value and the lowest testing RMSE value amongst the four approaches. Specifically the RMSE value obtained from the selected model was 0.1545 which indicates smaller prediction errors than all other regression models. Furthermore, the obtained R^2 was 0.2882 which was slightly higher than the KNN regression technique which gave a value of 0.2833 which suggests that it explained a greater proportion of the variation in specific growth rate. However, as the R^2 values for all four regression models were relatively low, this indicates that none of the model were able to explain a substantial proportion of the variation in the specific growth rate. Furthermore, the RMSE values were also slightly high suggesting that the predicted growth rates noticeable deviated from the observed values. Therefore, although degree-2 polynomial regression had the best performance metrics values, its overall predictive ability is limited. 

##### Model Architecture and Hyperparameters

The selected regression model, polynomial regression was implemented use PolynomialFeature() with LinearRegression(). Two polynomial degrees were evaluated, degree 2 and degree 3. The degree determines the order of polynomial terms generated from the input features, temperature pH glucose and lactate, which allows the model to represent nonlinear relationship between them. The polynomial model uses StandardScaler() before polynomial feature generation and Linear Regression() to fit the resulting features. 
From the non-selected regression model linear regression uses the LinearRegression() algorithm which establishes a baseline by fitting a linear relationship between the four input features and the specific growth rate. The input features were first standardized using StandardScaler() before implementing the regression algorithm. In terms of the K-nearest neighbors regression model, it was implemented using KNeighborsRegression() within a pipeline containing StandardScaler(). The model's hyperparameters were optimized using GridSearchCV() with five-fold cross validation. The number of neighbors was tested across 1 to 80, while the weighting parameter was tested using both uniform and distance. The best-performing configuration in the reported run used 12 neighbors with distance weighting. 

##### Model Limitations and Advantages 
The main advantage in using polynomial regression is its ability to capture nonlinear relationships between the feature operating conditions and the target specific growth rate. Unlike linear regression, polynomial regression uses higher order polynomial terms which allows it to capture more complex and nonlinear patterns which may be present in the dataset. In this program the second-order polynomial model obtained the best overall performance metrics with a RMSE vale of 0.1545 and an R^2 value of 0.2882. Therefore, the second-degree polynomial model outperformed both the third-degree polynomial and K-nearest neighbors model suggesting that it provided a better balance between model complexity and predictive performance. 
However, polynomial regression also has several limitation. One of the main disadvantages in using polynomial regression is that increasing the polynomial degree and therefore the model complexity may make the model more susceptible to overfitting. Therefore, a higher degree polynomial may fit the training data well but may have less accurate predictions when given new data it has not seen before. This disadvantage was observed in the evaluation metrics of polynomial regression with degree 2 and degree 3. It was observed that increase the polynomial degree caused the RMSE value to increase and the R^2 to decrease. Therefore, although polynomial regression performed best amongst the evaluated models, its prediction are still limited and should be further validated. 

##### Influence of Operating Conditions 
The second-degree polynomial regression model was implemented to investigate the relationship between the operating conditions and the specific growth rate. Each condition was varied while the remaining conditions were held constant as their dataset average value. The model predicted that increasing glucose concentration from 5 to 25 mM increased the specific growth rate from 0.328 to 0.400 d^-1. While increasing the lactate concentration from 5 to 25 mM decreased the predicted growth rate from 0.441 to 0.309 d^-1. In terms of the effect of pH on the specific growth rate, increasing the pH from 6.5 to 7.5 had a non-linear effect in the predicted growth rate. At a pH value of 7 the growth rate peaked at a value of 0.363 and the lowest growth rate value was obtained at a pH of 0.205 therefore indicating that a neutral pH provides a more favourable environment for specific cell growth. The effect of temperature follows a similar trend as pH where the specific growth rate peaks at a temperature of 36 degrees and is at its lowest at 40 degrees. 
With these trends it mind, we could predict that high specific growth rate would be seen with culture conditions of: high glucose concentration, low lactate concentration, neutral pH, and a temperature of around 36 degrees. Inversely, cell culture conditions that would result in low specific growth rate would be: low glucose concentration, high lactate concentration, slightly acidic pH, and a temperature value of approximately 40 degrees. However, these finding should be evaluated further as the predictions were generated while the other conditions were held constant. Therefore the model does not full take into account for the interactions between these operating conditions and how it may influence the specific cell culture growth rate.

|Glucose [mM]                    |Predicted Growth Rate [d^-1]|
|--------------------------------|----------------------------|
|5                               |0.32764751023976363         |
|10                              |0.34608835991302245         |
|15                              |0.3643726892894691          |
|20                              |0.3825004983691036          |
|25                              |0.400471787151926           |

|Lactate [mM]                    |Predicted Growth Rate [d^-1]|
|--------------------------------|----------------------------|
|5                               |0.4408261440938047          |
|10                              |0.3980711325025347          |
|15                              |0.3618356447166637          |
|20                              |0.33211968073619175         |
|25                              |0.30892324056111875         |

|pH                              |Predicted Growth Rate [d^-1]|
|--------------------------------|----------------------------|
|6.5                             |0.23704417220093585         |
|6.75                            |0.3354841485122934          |
|7.0                             |0.36309531068800593         |
|7.25                            |0.3198776587280734          |
|7.5                             |0.20583119263249572         |

|Temperature [C]                 |Predicted Growth Rate [d^-1]|
|--------------------------------|----------------------------|
|34                              |0.3411202835826818          |
|35                              |0.36008280154410527         |
|36                              |0.36713958208137387         |
|37                              |0.3622906251944877          |
|38                              |0.34553593088344664         |
|39                              |0.31687549914825075         |
|40                              |0.2763093299889             |


#### Potential Applications of this Model
In terms of practical applications of a cell growth monitoring process, this model could be used as an initial screening tool to indentify general operating condition trends that would result in higher predicted cell growth. Using this program as an initial screening tool, could potentially minimize the amount of experiments needed to optimize cell culture growth. However, as there are several limitation to this model it should be used only for primary investigations and should not be relied on for selecting operating conditions. 