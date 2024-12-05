
# For Linear Regression model

import pandas as pd     
import matplotlib.pyplot as plt              
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

df = pd.read_excel("Predictive maintenance system for network infrastructure.xlsx")
df.head(10)

# Convert 'Timestamp' column to datetime format
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
# Drop rows with missing values for the relevant columns
df_cleaned = df.dropna(subset=['CPU (%)', 'Memory (%)', 'Errors (Count)', 'Traffic (Mbps)'])
df_cleaned.columns

features = ["Traffic (Mbps)", "Memory (%)", "Errors (Count)"]
target = "CPU (%)"

X = df_cleaned[features]
y = df_cleaned[target]

# Preparation of the data

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the model

model = LinearRegression()
model.fit(X_train, y_train)

# Predict the CPU percentage on test
y_pred = model.predict(X_test)
y_pred[:10]

# Model evaluation
# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)

# Calculate R-squared (R²) score
r2 = r2_score(y_test, y_pred)

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)

# Print evaluation results
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R²): {r2}")
print(f"Mean Absolute Error (MAE): {mae}")

# Visualize results
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
plt.title("Actual vs Predicted CPU Utilization Percentage")
plt.xlabel("Actual CPU Utilization (%)")
plt.ylabel("Predicted CPU Utilization (%)")
plt.grid()
plt.show()

# Display feature importance (coefficients)
coefficients = pd.DataFrame({
    "Feature": features,
    "Coefficient": model.coef_
}).sort_values(by="Coefficient", ascending=False)

print("Feature Importance:")
print(coefficients)


# For Vector Machine Classifier

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_excel("Predictive maintenance system for network infrastructure.xlsx")  # Adjust if necessary
df.head(10)

# Select features and target variable
features = ["CPU (%)", "Memory (%)", "Errors (Count)", "Traffic (Mbps)"]
target = "Current Status"

# Drop rows with missing target or feature values
df_cleaned = df[features + [target]].dropna()

# Encode the target variable
label_encoder = LabelEncoder()
df_cleaned[target] = label_encoder.fit_transform(df_cleaned[target])

# Separate features (X) and target (y)
X = df_cleaned[features]
y = df_cleaned[target]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the SVM classifier
svm = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)
svm.fit(X_train, y_train)

# Make predictions
y_pred = svm.predict(X_test)
y_pred[:10]

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}\n")

# Assuming `label_encoder.classes_` contains the class names
class_names = label_encoder.classes_

# Generate the classification report with class names
classification_rep = classification_report(y_test, y_pred, target_names=class_names)

print("Classification Report:")
print(classification_rep)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show();



# For Random Forest Classifier 


# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Step 1: Load the dataset
df = pd.read_excel('Predictive maintenance system for network infrastructure.xlsx')
df.head(10)

# Step 2: Preprocess the data
# Define features and target
features = ['CPU (%)', 'Memory (%)', 'Errors (Count)', 'Traffic (Mbps)']
target = 'Current Status'

X = df[features]  # Features
y = df[target]    # Target

# Encode target variable if necessary
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Step 3: Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 4: Train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Step 5: Make predictions
y_pred = rf_model.predict(X_test)
y_pred[:10]

# Step 6: Evaluate the model
# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues",
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("Actual Labels")
plt.show();

# Classification Report
class_report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
print("Classification Report:")
print(class_report)

# Step 7: Analyze Feature Importance
feature_importances = rf_model.feature_importances_
feature_importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": feature_importances
}).sort_values(by="Importance", ascending=False)

# Plot feature importance
plt.figure(figsize=(8, 6))
bar_plot = sns.barplot(x="Importance", y="Feature", data=feature_importance_df, width=0.6)
colors = ['#FF6347', '#4682B4', '#32CD32', '#FFD700']
# Apply the colors to each bar
for i, bar in enumerate(bar_plot.patches):
    bar.set_facecolor(colors[i % len(colors)])

plt.title("Feature Importance in Random Forest")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.show();
