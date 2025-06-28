# --- Colab: Upload insurance.csv if not already present ---
import os

# --- Main Regression Script ---
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# Load and preprocess data
df = pd.read_csv('insurance.csv')

region_map = {'southwest': 0, 'southeast': 1, 'northwest': 2, 'northeast': 3}
df['sex'] = df['sex'].map({'female': 0, 'male': 1})
df['smoker'] = df['smoker'].map({'no': 0, 'yes': 1})
df['region'] = df['region'].map(region_map)

X = df.drop('expenses', axis=1) #Features
y = df['expenses']       #Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X_train_scaled, y_train)
y_pred_lin = lin_reg.predict(X_test_scaled)

print("## Linear Regression Results")
print(f"R² Score: {r2_score(y_test, y_pred_lin):.4f}")
print(f"MAE: ${mean_absolute_error(y_test, y_pred_lin):.2f}")

# Polynomial Regression (degree=3)
poly = PolynomialFeatures(degree=9)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

poly_reg = LinearRegression()
poly_reg.fit(X_train_poly, y_train)
y_pred_poly = poly_reg.predict(X_test_poly)

print("\n## Polynomial Regression (Degree=3) Results")
print(f"R² Score: {r2_score(y_test, y_pred_poly):.4f}")
print(f"MAE: ${mean_absolute_error(y_test, y_pred_poly):.2f}")

# --- User Input for Prediction ---
def predict_user_input():
    print("\n--- Predict Insurance Charges ---")
    age = float(input("Enter age: "))
    sex = input("Enter gender (male/female): ").strip().lower()
    bmi = float(input("Enter BMI: "))
    children = int(input("Enter number of children: "))
    smoker = input("Smoker? (yes/no): ").strip().lower()
    region = input("Enter region (southwest/southeast/northwest/northeast): ").strip().lower()

    user_data = [[
        age,
        0 if sex == 'female' else 1,
        bmi,
        children,
        1 if smoker == 'yes' else 0,
        region_map.get(region, 0)
    ]]

    user_scaled = scaler.transform(user_data)
    user_poly = poly.transform(user_scaled)

    lin_pred = lin_reg.predict(user_scaled)[0]
    poly_pred = poly_reg.predict(user_poly)[0]

    print(f"\nPredicted Insurance Charges:")
    print(f"Linear Regression: ${lin_pred:.2f}")
    print(f"Polynomial Regression: ${poly_pred:.2f}")

if __name__ == "__main__":
    predict_user_input()