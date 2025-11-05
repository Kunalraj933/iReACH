import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from statsmodels.discrete.conditional_models import ConditionalLogit
from sklearn.model_selection import GroupShuffleSplit
import matplotlib.pyplot as plt


############### Create fake hospital choice data
############### Needs replacing with real HES data
np.random.seed(42)

n_patients = 200
n_hospitals = 3

# Setup choice sets randomised
patient_ids = np.repeat(np.arange(n_patients), n_hospitals)
alts = np.tile(np.arange(n_hospitals), n_patients)

distance = np.random.uniform(1, 20, size=n_patients * n_hospitals) # distance from patien's LSOA to hospitals
wait_time = np.random.uniform(10, 200, size=n_patients * n_hospitals) # average wait times from HES data
rating = np.random.uniform(1, 5, size=n_patients * n_hospitals) # 1-5 stars ?google ratings

# Distance major negative utility factor, wait time minor negative factor, rating positive factor
# Can these coefficients be computed from HES data?
beta = np.array([-0.3, -0.01, 0.1])
utilities = beta[0]*distance + beta[1]*wait_time + beta[2]*rating + np.random.normal(0,1,n_patients*n_hospitals)

# Each patient picks the hospital with highest utility
choices = []
for patient in range(n_patients):
    u = utilities[patient*n_hospitals:(patient+1)*n_hospitals]
    chosen = np.argmax(u)
    choices.extend((alts[patient*n_hospitals:(patient+1)*n_hospitals] == chosen).astype(int))

df = pd.DataFrame({
    "patient_id": patient_ids,
    "hospital_id": alts,
    "distance": distance,
    "wait_time": wait_time,
    "rating": rating,
    "choice": choices
})

# Print summary of sample data
print("Sample data:")
print(df.head(10))

############### Split into test and train 25/75%

splitter = GroupShuffleSplit(test_size=0.25, n_splits=1, random_state=42)
split = splitter.split(df, groups=df['patient_id'])
train_idx, test_idx = next(split)

df_train = df.iloc[train_idx]
df_test = df.iloc[test_idx]

############### Conditional Logit model

model = ConditionalLogit(endog = df_train["choice"], 
                         exog = df_train[["distance", "wait_time", "rating"]], 
                         groups = df_train["patient_id"]
)

results = model.fit()

print("\nTraining Data Summary:")
print(results.summary())

############### Predicting probabilities manulally : probability that each patient chooses each hospital
def compute_pred_probs(results, df, n_hospitals):
    beta = results.params.values  # estimated coefficients
    X = df[["distance", "wait_time", "rating"]].values
    n_patients = df['patient_id'].nunique()
    
    # reshape X to (patients, hospitals, features)
    X_reshaped = X.reshape(n_patients, n_hospitals, -1)
    
    # compute utilities
    utilities = np.einsum('ijk,k->ij', X_reshaped, beta)  # shape (patients, hospitals)
    
    # softmax to get probabilities
    expU = np.exp(utilities)
    probs = expU / expU.sum(axis=1, keepdims=True)

    return probs

############### Model validation : accuracy of model predictions (patient-level)
def validate_model_manual(results, df, n_hospitals):
    probs = compute_pred_probs(results, df, n_hospitals)
    
    # Actual choices per patient
    actual_choices = df.groupby('patient_id')['choice'].apply(lambda x: np.argmax(x.values)).values
    
    # Predicted choices per patient
    predicted_choices = probs.argmax(axis=1)
    
    accuracy = np.mean(actual_choices == predicted_choices)
    return accuracy

print("\nModel Validation Results (Manual Probabilities):")
train_accuracy = validate_model_manual(results, df_train, n_hospitals)
test_accuracy = validate_model_manual(results, df_test, n_hospitals)
print(f"Training accuracy: {train_accuracy:.2%}")
print(f"Test accuracy: {test_accuracy:.2%}")


############### Provider-level RMSE (manual) : accuracy of predicted demand per hospital (provider-level)
def calculate_provider_rmse_manual(results, df, n_hospitals):
    probs = compute_pred_probs(results, df, n_hospitals)
    
    # sum predicted probabilities per hospital
    pred_demand = probs.sum(axis=0)
    
    # actual demand per hospital
    actual_demand = df.groupby('hospital_id')['choice'].sum().sort_index()
    
    rmse = np.sqrt(np.mean((actual_demand.values - pred_demand)**2))
    return rmse, actual_demand, pred_demand

train_rmse, train_obs, train_pred = calculate_provider_rmse_manual(results, df_train, n_hospitals)
test_rmse, test_obs, test_pred = calculate_provider_rmse_manual(results, df_test, n_hospitals)

print("\nProvider-Level Demand Analysis (Manual Probabilities):")
print(f"Training data RMSE: {train_rmse:.2f}")
print(f"Test data RMSE: {test_rmse:.2f}")
comparison = pd.DataFrame({
    'Observed_Demand': test_obs,
    'Predicted_Demand': test_pred,
    'Difference': test_obs.values - test_pred
})
print("\nDetailed Provider Demand Comparison (Test Set):")
print(comparison)



############### Visualization of Provider Demand
def plot_provider_demand(observed, predicted, title="Provider Demand"):
    hospital_ids = observed.index
    width = 0.35  # width of the bars

    fig, ax = plt.subplots(figsize=(8,5))
    ax.bar(hospital_ids - width/2, observed.values, width, label='Observed', color='skyblue')
    ax.bar(hospital_ids + width/2, predicted, width, label='Predicted', color='salmon')

    ax.set_xlabel("Hospital ID")
    ax.set_ylabel("Demand (Number of Patients)")
    ax.set_title(title)
    ax.set_xticks(hospital_ids)
    ax.legend()
    plt.show()

# Plot for test set
plot_provider_demand(test_obs, test_pred, title="Test Set: Observed vs Predicted Provider Demand")
