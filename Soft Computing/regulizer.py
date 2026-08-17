import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# ============================================================
# 1. Generate Random Dataset
# ============================================================

np.random.seed(42)

X = np.linspace(-5, 5, 50).reshape(-1, 1)

# Actual function
y = 3 * X[:, 0]**3 - 2 * X[:, 0]**2 + 5 * X[:, 0] + 10

# Add noise
noise = np.random.normal(0, 30, len(X))
y = y + noise


# ============================================================
# 2. Train-Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ============================================================
# 3. Polynomial Degree
# ============================================================

degree = 20


# ============================================================
# 4. Overfitted Polynomial Regression
# ============================================================

poly_model = make_pipeline(
    PolynomialFeatures(degree),
    LinearRegression()
)

poly_model.fit(X_train, y_train)

poly_pred = poly_model.predict(X_test)


# ============================================================
# 5. Lasso Polynomial Regression
# ============================================================

lasso_model = make_pipeline(
    PolynomialFeatures(degree),
    Lasso(alpha=10, max_iter=100000)
)

lasso_model.fit(X_train, y_train)

lasso_pred = lasso_model.predict(X_test)


# ============================================================
# 6. Ridge Polynomial Regression
# ============================================================

ridge_model = make_pipeline(
    PolynomialFeatures(degree),
    Ridge(alpha=10)
)

ridge_model.fit(X_train, y_train)

ridge_pred = ridge_model.predict(X_test)


# ============================================================
# 7. Calculate Performance
# ============================================================

print("========== RESULTS ==========")

print("\nOverfitted Polynomial")
print("MSE :", mean_squared_error(y_test, poly_pred))
print("R2  :", r2_score(y_test, poly_pred))

print("\nLasso Polynomial")
print("MSE :", mean_squared_error(y_test, lasso_pred))
print("R2  :", r2_score(y_test, lasso_pred))

print("\nRidge Polynomial")
print("MSE :", mean_squared_error(y_test, ridge_pred))
print("R2  :", r2_score(y_test, ridge_pred))


# ============================================================
# 8. Generate Smooth X Values
# ============================================================

X_plot = np.linspace(-5, 5, 500).reshape(-1, 1)

poly_curve = poly_model.predict(X_plot)
lasso_curve = lasso_model.predict(X_plot)
ridge_curve = ridge_model.predict(X_plot)


# ============================================================
# 9. Three Subplots
# ============================================================

plt.figure(figsize=(18, 5))


# ------------------------------------------------------------
# Plot 1: Overfitted Polynomial
# ------------------------------------------------------------

plt.subplot(1, 3, 1)

plt.scatter(
    X_train,
    y_train,
    label="Training Data"
)

plt.scatter(
    X_test,
    y_test,
    label="Test Data"
)

plt.plot(
    X_plot,
    poly_curve,
    linewidth=2,
    label="Polynomial Degree 15"
)

plt.title("Overfitted Polynomial Regression")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)


# ------------------------------------------------------------
# Plot 2: Lasso
# ------------------------------------------------------------

plt.subplot(1, 3, 2)

plt.scatter(
    X_train,
    y_train,
    label="Training Data"
)

plt.scatter(
    X_test,
    y_test,
    label="Test Data"
)

plt.plot(
    X_plot,
    lasso_curve,
    linewidth=2,
    label="Lasso"
)

plt.title("Lasso Polynomial Regression")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)


# ------------------------------------------------------------
# Plot 3: Ridge
# ------------------------------------------------------------

plt.subplot(1, 3, 3)

plt.scatter(
    X_train,
    y_train,
    label="Training Data"
)

plt.scatter(
    X_test,
    y_test,
    label="Test Data"
)

plt.plot(
    X_plot,
    ridge_curve,
    linewidth=2,
    label="Ridge"
)

plt.title("Ridge Polynomial Regression")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)


# ============================================================
# Show All Plots
# ============================================================

plt.tight_layout()
plt.show()