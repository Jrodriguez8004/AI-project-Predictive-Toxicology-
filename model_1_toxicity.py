import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
file_path = r"C:\\Users\\brian\\OneDrive\\Escritorio\\Lambton\\Semester2\\Toxicology\\Datasets\\PubChem_compound_cache.csv"
data = pd.read_csv(file_path)

# Step 1: Handle Missing Values
data.fillna(data.median(numeric_only=True), inplace=True)  # Fill numeric columns with median
data.fillna("Unknown", inplace=True)  # Fill non-numeric columns with 'Unknown'

# Step 2: Drop Irrelevant Columns
columns_to_drop = ['cmpdsynonym', 'iupacname', 'meshheadings', 'annotation']
data.drop(columns=columns_to_drop, inplace=True, errors='ignore')

# Step 3: Label Encode Categorical Variables
categorical_columns = data.select_dtypes(include=['object']).columns
for col in categorical_columns:
    data[col] = LabelEncoder().fit_transform(data[col])

# Step 4: Select Features and Target
# Replace 'annotation' with your actual target column if applicable
target_column = 'mw'  # Example: use molecular weight for testing (update with your target variable)
if target_column not in data.columns:
    raise ValueError("Specify a valid target column.")
X = data.drop(columns=[target_column])
y = data[target_column]


#################################3
# Preparing features and target for the first model (e.g., is_toxic classification)
data['is_toxic'] = (data['hbonddonor'] > 0).astype(int)  # Example logic for toxicity

# Define features and target for Model 1
features_model_1 = ['mw', 'polararea', 'complexity', 'xlogp', 'heavycnt', 
                    'hbonddonor', 'rotbonds', 'totalatomstereocnt', 'totalbondstereocnt']
target_model_1 = 'is_toxic'  # Update this if your target column name is different

# Prepare data for Model 1
X1 = data[features_model_1]
y1 = data[target_model_1]

# Train/Test Split
from sklearn.model_selection import train_test_split
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=42)

# Train RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
model_1 = RandomForestClassifier(random_state=42)
model_1.fit(X1_train, y1_train)

# Evaluate Model 1
y1_pred = model_1.predict(X1_test)
print(f"Model 1 Accuracy: {accuracy_score(y1_test, y1_pred)}")

# Save Model 1
import os
import joblib
output_dir = r"C:\\Users\\brian\\OneDrive\\Escritorio\\Lambton\\Semester2\\Toxicology"
joblib.dump(model_1, os.path.join(output_dir, 'model_1_toxicity.pkl'))

