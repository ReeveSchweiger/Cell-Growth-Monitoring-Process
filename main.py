import src.functions as sf
import pandas as pd
from src.classes import Batch, CellGrowthModel
from src.constants import *
import os

ph_optimal = sf.calculate_average(PH_LIMITS)
temperature_optimal = sf.calculate_average(TEMPERATURE_LIMITS)

batch = Batch(batch_id=1, ph_opt=ph_optimal, temperature_opt=temperature_optimal)
batch.display_info()

growth_model = CellGrowthModel()

results = []

results.append(growth_model.evaluate_model())
results.append(growth_model.evaluate_polynomial_model(degree=2))
results.append(growth_model.evaluate_polynomial_model(degree=3))
results.append(growth_model.evaluate_knn_model())

df_results = pd.DataFrame(results)

df_results = df_results.round(4)

df_results.to_csv(os.path.join("tables", "model_results.csv"), index=False)

print("\nModel Comparison")
print("----------------")
print(df_results)


prediction_results = []

prediction_results.append(growth_model.predict_growth(temperature=36,ph=6.0,glucose=7,lactate=2))

df_predictions = pd.DataFrame(prediction_results)
df_predictions = df_predictions.round(4)

df_predictions.to_csv(os.path.join("tables", "growth_predictions.csv"),index=False)

print("\nGrowth Rate Predictions")
print("-----------------------")
print(df_predictions.to_string(index=False))


lactate_results = growth_model.analyze_condition("Lactate [mM]", [5, 10, 15, 20, 25])

glucose_results = growth_model.analyze_condition("Glucose [mM]", [5, 10, 15, 20, 25])

temperature_results = growth_model.analyze_condition("Temperature [C]", [34, 35, 36, 37, 38, 39, 40])

ph_results = growth_model.analyze_condition("pH", [6.5, 6.75, 7.0, 7.25, 7.5])


print("\nEffect of Lactate on Growth")
print("---------------------------")
print(lactate_results.round(4).to_string(index=False))

print("\nEffect of Glucose on Growth")
print("---------------------------")
print(glucose_results.round(4).to_string(index=False))

print("\nEffect of Temperature on Growth")
print("-------------------------------")
print(temperature_results.round(4).to_string(index=False))

print("\nEffect of pH on Growth")
print("----------------------")
print(ph_results.round(4).to_string(index=False))


lactate_results.to_csv(os.path.join("tables", "lactate_analysis.csv"),index=False)

glucose_results.to_csv(os.path.join("tables", "glucose_analysis.csv"),index=False)

temperature_results.to_csv(os.path.join("tables", "temperature_analysis.csv"),index=False)

ph_results.to_csv(os.path.join("tables", "ph_analysis.csv"),index=False)
