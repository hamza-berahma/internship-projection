# 📊 Google Trends "Internship" Analysis

This project analyzes Google Trends data for the search term **"internship"** using time-series forecasting methods: **Prophet** and **Linear Regression**.
## 🧭 Overview
* **Goal:** Model and predict future search interest in *internships* from **August 2025 to August 2026**.
* **Tools:** `pandas`, `numpy`, `matplotlib`, `prophet`, `scikit-learn`, `scipy`
* **Evaluation Metrics:**
  * MAE (Mean Absolute Error)
  * RMSE (Root Mean Squared Error)
  * MAPE (Mean Absolute Percentage Error)
  * $R^2$ (Coefficient of Determination)
## ⚙️ Requirements

```bash
pip install pandas numpy matplotlib prophet scikit-learn scipy
```
## 🧹 Data Preprocessing
* **Input:** CSV with `date` (YYYY-MM) and `internship` (search interest, $0 \leq y \leq 100$).
* **Cleaning:**
  * Replace `<1` with `0`
  * Convert values to integers
  * Parse `date` as `datetime`
* **Feature Engineering:**
  * **Time Index**: $t = 0, 1, \ldots, n-1$
  * **Month Dummies**: $m_2, m_3, \ldots, m_{12}$
  * **Quarter Dummies**: $q_2, q_3, q_4$

## 🔮 Models
### 1. Prophet
* **Model:**
  $y(t) = g(t) + s(t) + h(t) + \epsilon_t$
  * $g(t)$: trend
  * $s(t)$: seasonality
  * $h(t)$: holiday effects
  * $\epsilon_t \sim \mathcal{N}(0, \sigma^2)$: noise

* **Procedure:**

  * Train on 80% of data
  * Test on 20%
  * Forecast 12 months ahead (Aug 2025 – Aug 2026)

* **Forecast Output:**

  $\hat{y}(t),\quad yhat_{\text{lower}},\quad yhat_{\text{upper}}$

* **Metrics:** 
  $\text{MAE} = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|$
  $\text{RMSE} = \sqrt{ \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2 }$

### 2. Linear Regression

* **Features:**

  * Time Index $t$
  * Month dummies $m_i$
  * Quarter dummies $q_j$

* **Model:**
  $y = \beta_0 + \beta_1 t + \sum_{i=2}^{12} \beta_i m_i + \sum_{j=2}^{4} \beta_j q_j + \epsilon$
  where $\epsilon \sim \mathcal{N}(0, \sigma^2)$

* **Metrics:**

  $\text{MAE} = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|$
  $\text{RMSE} = \sqrt{ \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2 }$
  $\text{MAPE} = \frac{100}{n} \sum_{i=1}^n \left| \frac{y_i - \hat{y}_i}{y_i} \right|$
  $R^2 = 1 - \frac{ \sum_{i=1}^n (y_i - \hat{y}_i)^2 }{ \sum_{i=1}^n (y_i - \bar{y})^2 } \quad \text{with} \quad \bar{y} = \frac{1}{n} \sum_{i=1}^n y_i$

## 📈 Analysis

* **Monthly Averages:**
  $\bar{y}_m = \frac{1}{n_m} \sum_{i \in m} y_i$
  Smoothed using **cubic spline interpolation** with $k = 3$.
* **Quarterly Averages:**
  $\bar{y}_q = \frac{1}{n_q} \sum_{i \in q} y_i$
* **Linear Regression Results:**
  * MAPE: **14.78%**
  * Accuracy: **85.22%** $= 100\% - \text{MAPE}$
  * $R^2$: **0.519**

## 📊 Visualizations
* **Linear Regression Forecast vs Actual (Test Set):**
  * Blue: Actual values
  * Orange: Predicted values
  * Time span: **July 2021 – July 2025**
<img width="1189" height="490" alt="image" src="https://github.com/user-attachments/assets/11dafb92-9f1b-4787-81ff-95807b718e9e" />

## 💻 Usage

1. Export Google Trends CSV as `multiTimeline.csv`
2. Run the script:

   * Compute $\bar{y}_m$, $\bar{y}_q$
   * Fit Prophet & Linear Regression
   * Forecast $\hat{y}(t)$ for next 12 months
   * Compute metrics (MAE, RMSE, MAPE, $R^2$)

---

## 📤 Outputs

* **Forecasts**:

  * $\hat{y}(t)$ for Aug 2025 – Aug 2026
  * Prophet forecast with confidence intervals
* **Metrics:**

  * MAE & RMSE (both models)
  * MAPE & $R^2$ (Linear Regression only)

---

## 📅 Current Date and Time

* **Date & Time:** 05:50 PM +01, Thursday, July 17, 2025

---

## 📝 License

**MIT License**
