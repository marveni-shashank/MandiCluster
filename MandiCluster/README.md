# 🌾 MandiCluster: Agricultural Market & Price-Pattern Segmentation

> **Domain:** Unsupervised Machine Learning, Agricultural Data Analytics, Market Intelligence  
> **Academic Context:** B.Tech Computer Science & Engineering Capstone Project  
> **Data Source:** Agmarknet (Directorate of Marketing & Inspection, Ministry of Agriculture & Farmers Welfare, GoI)  
> **Tech Stack:** Python Flask, Scikit-learn, Pandas, NumPy, Chart.js, HTML5/CSS3  

---

## 📌 Project Overview

**MandiCluster** is an AI-driven agricultural analytics and market intelligence web application. Across India, agricultural produce is traded across more than 2,400 principal APMC (*Agricultural Produce Market Committee*) mandis. Due to logistical friction, regional supply variations, and localized intermediary trading, mandis exhibit stark disparities in price levels and intra-day price volatility.

Traditional portals provide only flat tabular listings for single mandis without macro-level pattern discovery. **MandiCluster** applies **Unsupervised Machine Learning (K-Means Clustering)** on verified Agmarknet daily price records to automatically segment agricultural markets into coherent economic clusters based on price tiers, intra-mandi price spreads, and volatility profiles.

---

## 🌟 Key Features

1. **🏠 Modern AI Analytics Dashboard:**
   - Agritech-themed design (white & light mint theme with dark forest green accents).
   - Real-time KPIs tracking active mandis, tracked commodities, and state coverage.
   - Responsive layout optimized for desktop, tablet, and mobile screens.

2. **📊 Market Analytics Module:**
   - Dynamic filters by Commodity, State, and Mandi.
   - Parametric price range calculations (Minimum, Modal, Maximum Prices in ₹/Quintal).
   - Intra-mandi price spread and relative percentage volatility metrics.
   - Interactive Chart.js comparative visualizations.

3. **🎯 Price Pattern Segmentation (K-Means Clustering):**
   - User-selectable cluster count ($K \in [2, 7]$) with live hyperparameter tuning.
   - Feature selection (`Modal_Price`, `Price_Spread`, `Price_Spread_Pct`, `Min_Price`, `Max_Price`).
   - Feature normalization using `StandardScaler` (Z-score scaling: $\mu=0, \sigma=1$).
   - 2D Principal Component Analysis (PCA) projection scatter plot with hover metadata.
   - Domain-specific cluster archetypes (e.g. *Premium Commercial Spices*, *Stable Essential Foodgrains*, *Volatile Perishables*) derived from actual cluster statistics.

4. **📈 Model Performance & Evaluation:**
   - Within-Cluster Sum of Squares (Inertia / WCSS).
   - Silhouette Coefficient analysis across $K \in [2, 8]$.
   - Davies-Bouldin Index & Calinski-Harabasz Variance Ratio Criterion.
   - Real-time Elbow Method curve indicating optimal mathematical $K$.
   - **Academic Rigor Guarantee:** Explicit disclaimer clarifying why classification "accuracy" is mathematically undefined in unsupervised clustering, focusing instead on cluster compactness and separation.

5. **📁 Dataset Explorer & CSV Ingestion:**
   - Summary statistics (Record count, missing value analysis, parametric distributions).
   - Drag-and-drop CSV upload for any standard Agmarknet dataset with schema validation.
   - Graceful handling of missing values, negative prices, and currency symbols.
   - One-click restore to built-in verified Agmarknet dataset.
   - Paginated raw data table with live search.

6. **📖 B.Tech CSE Project Documentation:**
   - Formal problem statement, objectives, existing vs. proposed system comparative matrix.
   - End-to-end methodology workflow, real-world agritech impact, and limitations/future scope.

---

## 📁 Project Folder Structure

```text
MandiCluster/
│
├── app.py                     # Flask web server, page routes, and RESTful API endpoints
├── data_loader.py             # Agmarknet CSV loading, standardization, validation & stats
├── ml_engine.py               # Scikit-learn K-Means, StandardScaler, PCA, Elbow & Silhouette
├── test_app.py                # Automated 12-test suite for routes, ML engine, and APIs
├── requirements.txt           # Python dependency specifications
├── README.md                  # Complete documentation and setup manual
│
├── data/
│   └── agmarknet_sample.csv   # Verified Agmarknet dataset across major Indian mandis & crops
│
├── uploads/                   # Secure storage directory for user-uploaded custom CSVs
│
├── static/
│   └── css/
│       └── style.css          # Modern, minimal agritech UI stylesheet
│
└── templates/
    ├── layout.html            # Master base template with responsive navigation & footer
    ├── index.html             # Home page with project overview & architecture pipeline
    ├── analytics.html         # Market analytics with filters & price comparison charts
    ├── segmentation.html      # K-Means clustering interface, 2D PCA & cluster profiles
    ├── performance.html       # Elbow method curve, Silhouette score & evaluation metrics
    ├── explorer.html          # Dataset summary, paginated records table & CSV upload
    └── about.html             # Academic B.Tech project documentation & methodology
```

---

## 💻 Exact Setup & Run Instructions (Windows PowerShell)

Follow these exact commands in **Windows PowerShell**:

### 1. Open PowerShell and Navigate to the Project Directory
```powershell
cd "C:\Users\Marav\.gemini\antigravity\scratch\MandiCluster"
```

### 2. Verify Python Installation (Python 3.10+ recommended)
```powershell
python --version
```
*(If `python` is not directly on PATH, use the full path: `& "C:\Users\Marav\AppData\Local\Programs\Python\Python312\python.exe"`)*

### 3. Install Required Dependencies
```powershell
pip install -r requirements.txt
```
*(Or via direct Python invocation:)*
```powershell
& "C:\Users\Marav\AppData\Local\Programs\Python\Python312\python.exe" -m pip install -r requirements.txt
```

### 4. Run the Automated Verification Test Suite
```powershell
python test_app.py
```
*(All 12 unit and integration tests should report `OK`)*

### 5. Start the MandiCluster Web Server
```powershell
python app.py
```

### 6. Access the Application
Open your web browser (Chrome, Edge, Firefox) and navigate to:
```text
http://127.0.0.1:5000
```

---

## 🧪 Machine Learning Pipeline & Mathematical Foundations

### 1. Feature Engineering
From the raw Agmarknet auction data, domain-specific features are synthesized:
- **Modal Price ($P_{\text{modal}}$):** The price at which the highest volume of transactions occurred (₹/Quintal).
- **Price Spread ($S$):** Measure of intra-day market volatility:
  $$S = P_{\text{max}} - P_{\text{min}}$$
- **Spread Ratio Percentage ($S_{\%}$):** Normalized intra-day volatility:
  $$S_{\%} = \left( \frac{S}{P_{\text{modal}}} \right) \times 100$$

### 2. Preprocessing & Z-Score Normalization
Because prices for commodities vary wildly (e.g., Arecanut at ₹48,000/Q vs. Wheat at ₹2,300/Q), raw Euclidean distance would be biased toward high-nominal commodities. Features are transformed using **StandardScaler**:
$$z = \frac{x - \mu}{\sigma}$$

### 3. K-Means++ Clustering
The objective function minimizes Within-Cluster Sum of Squares (Inertia):
$$J = \sum_{k=1}^{K} \sum_{x \in C_k} \| x - \mu_k \|^2$$
We utilize `k-means++` initialization with `n_init=15` to ensure stable convergence and avoid sub-optimal local minima.

### 4. Dimensionality Reduction (2D PCA)
To visualize multi-dimensional feature space (`Modal_Price`, `Price_Spread`, `Price_Spread_Pct`) on a 2D canvas, Principal Component Analysis calculates orthogonal eigenvectors capturing maximum variance:
$$X_{\text{PCA}} = X_{\text{scaled}} \cdot W_{\text{top 2}}$$

### 5. Evaluation Metrics
- **Elbow Method (WCSS):** Identifies the inflection point where adding more clusters yields diminishing returns.
- **Silhouette Coefficient ($s$):** Evaluates how cohesive an item is within its cluster compared to neighboring clusters (range $[-1, +1]$):
  $$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$
  A score of $> 0.50$ confirms high separation and cluster cohesion.
- **Davies-Bouldin Index:** Measures similarity between clusters (lower indicates better separation).
- **Calinski-Harabasz Index:** Variance ratio criterion (higher indicates dense, well-separated clusters).

---

## 🎓 Academic Defense Highlights (Viva / Presentation Q&A)

| Question | Engineering Answer |
|---|---|
| **Why is accuracy not reported?** | Unsupervised clustering has no ground-truth labels ($y$). Claiming "98% accuracy" in clustering is scientifically invalid. Instead, we use mathematical validity metrics (Inertia, Silhouette Score, Davies-Bouldin) and domain validity. |
| **Why is feature scaling essential?** | Without `StandardScaler`, K-Means Euclidean distance would treat a ₹500 variation in ₹50,000 Cardamom identically to a ₹500 variation in ₹1,500 Potato, distorting clustering. |
| **How does MandiCluster help farmers?** | It identifies nearby mandis belonging to higher price clusters with low intra-day spreads, enabling informed transportation decisions to prevent distress selling. |
| **What happens if a custom CSV has missing prices?** | The preprocessing pipeline logically imputes missing modal prices using the midpoint of minimum and maximum prices, strips currency artifacts, and rejects invalid entries. |

---

## 📜 License & Acknowledgments
Developed for academic assessment in **B.Tech Computer Science & Engineering**. Data sourced under Open Government Data (OGD) Platform India / Agmarknet.
