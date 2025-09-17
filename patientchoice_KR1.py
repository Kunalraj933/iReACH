import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from statsmodels.discrete.conditional_models import ConditionalLogit
from sklearn.model_selection import GroupShuffleSplit



############### Create fake hospital choice data with more hospitals
# Needs replacing with real HES data
np.random.seed(42)
pat = 200
hosp = 10 
pat_ids = np.repeat(np.arange(pat), hosp)
hosp_ids = np.tile(np.arange(hosp), pat)

# Fake LSOA codes for patients
pat_lsoa = {i: f"LSOA_{i//10:03d}" for i in range(pat)}
lsoa_codes = [pat_lsoa[pid] for pid in pat_ids]

# Distance from patient to hospitals (in HES from LSOA centroids), wait_time (from HES), rating (from Google/CQC)
distance = np.random.uniform(1, 50, size=pat * hosp)
wait_time = np.random.uniform(10, 200, size=pat * hosp)
rating = np.random.uniform(1, 5, size=pat * hosp)

df = pd.DataFrame({
    "pat_id": pat_ids,
    "hosp_id": hosp_ids,
    "lsoa_code": lsoa_codes,
    "distance": distance,
    "wait_time": wait_time,
    "rating": rating
})

############### Choice set from 3 closest hospitals: distance from patient's LSOA to hospitals
def closest_hosp(df, n_closest=3):
    f_data = []
    for pat_id in df['pat_id'].unique():
        pat_data = df[df['pat_id'] == pat_id].copy()
        pat_data = pat_data.dropna(subset=['distance'])
        closest = pat_data.sort_values('distance').head(3)
        f_data.append(closest)
    
    return pd.concat(f_data, ignore_index=True)
df_f = closest_hosp(df, n_closest=3)

############### Compute choices based on utility
# Distance -0.3, wait_time -0.01, rating 0.1
beta = np.array([-0.3, -0.01, 0.1])
utilities = (beta[0]*df_f['distance'] + 
            beta[1]*df_f['wait_time'] + 
            beta[2]*df_f['rating'] + 
            np.random.normal(0, 1, len(df_f)))

# Highest utility picked
choices = []
for patient in df_f['pat_id'].unique():
    pat_mask = df_f['pat_id'] == patient
    pat_utilities = utilities[pat_mask]
    chosen_idx = np.argmax(pat_utilities)
    
# Create choice vector for this patient
    pat_choices = np.zeros(len(pat_utilities))
    pat_choices[chosen_idx] = 1
    choices.extend(pat_choices)

df_f['choice'] = choices

# Print summary of sample data
print(df_f.head(15))

############### Split into test and train 27/75%
splitter = GroupShuffleSplit(test_size=0.25, n_splits=1, random_state=42)
split = splitter.split(df_f, groups=df_f['pat_id'])
train_idx, test_idx = next(split)

df_train = df_f.iloc[train_idx]
df_test = df_f.iloc[test_idx]

############### Conditional Logit model
model = ConditionalLogit(endog=df_train["choice"], 
                        exog=df_train[["distance", "wait_time", "rating"]], 
                        groups=df_train["pat_id"])

results = model.fit()
print(results.summary())
