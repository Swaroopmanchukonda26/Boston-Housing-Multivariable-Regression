# Multivariable Linear Regression & Log-Data Pricing Transformations

An advanced real estate analytics and predictive machine learning project engineered to estimate 1970s residential property valuations in Boston, Massachusetts by evaluating multi-column architectural and environmental spatial attributes using Seaborn and scikit-learn.

## 🚀 Key Architectural Discoveries
* **Data-Shaping Interventions:** Initial residual diagnostic analysis revealed significant heteroscedasticity due to right-skewed property tiers. Applying a natural log transformation ($y \rightarrow \ln(y)$) to target valuation parameters normalized variance boundaries and significantly boosted validation performance.
* **Generalization Security:** Implemented an 80/20 train-test split framework matrix to guarantee the regression model can successfully predict valuations on entirely unseen neighborhood characteristics rather than overfitting training metrics.
* **Mathematical Inversion:** Because the linear optimization engine learns on logarithmic distributions, predictive inference queries are cleanly processed back into real-world dollar amounts using inverse exponential broadcasting (`np.exp()`).

## 🛠️ Data Science Toolkit Used
* **Languages & Environments:** Python, Google Colab
* **Exploratory Visualizations:** Seaborn (`sns.pairplot()`) to map cross-correlation matrices across room counts (RM), employment distances (DIS), and student-teacher dynamics (PTRATIO).
* **Machine Learning Pipeline:** Scikit-Learn Validation Suite (`train_test_split`, `LinearRegression`), tracking multivariate slope coefficients ($\theta_n$) and residual error variances.
