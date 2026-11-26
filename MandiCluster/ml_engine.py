"""
MandiCluster - Machine Learning Engine
Implements K-Means clustering, preprocessing, feature scaling,
Elbow method evaluation, cluster profiling, and dimensional projection.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA


FEATURE_DESCRIPTIONS = {
    'Modal_Price': 'Modal Price (₹/Quintal) - Dominant trading price level',
    'Price_Spread': 'Intra-market Price Spread (Max - Min in ₹/Quintal)',
    'Price_Spread_Pct': 'Relative Volatility Spread (%) - (Spread / Modal Price * 100)',
    'Min_Price': 'Minimum Price (₹/Quintal) - Floor price recorded',
    'Max_Price': 'Maximum Price (₹/Quintal) - Ceiling price recorded'
}


def prepare_clustering_features(df: pd.DataFrame, feature_names: list[str] = None):
    """
    Extracts selected numerical features and applies standard scaling.
    Returns:
        X_raw: pd.DataFrame of unscaled features
        X_scaled: np.ndarray of scaled features
        scaler: fitted StandardScaler
        selected_features: list of feature names used
    """
    if feature_names is None or len(feature_names) == 0:
        feature_names = ['Modal_Price', 'Price_Spread', 'Price_Spread_Pct']

    # Ensure all requested features exist in df
    valid_features = [f for f in feature_names if f in df.columns]
    if len(valid_features) == 0:
        valid_features = ['Modal_Price', 'Price_Spread', 'Price_Spread_Pct']

    X_raw = df[valid_features].copy()
    # Handle any remaining inf or nan
    X_raw = X_raw.replace([np.inf, -np.inf], np.nan).fillna(X_raw.median())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    return X_raw, X_scaled, scaler, valid_features


def compute_elbow_curve(df: pd.DataFrame, feature_names: list[str] = None, max_k: int = 8) -> dict:
    """
    Computes Within-Cluster Sum of Squares (Inertia/WCSS) and Silhouette scores
    across k = 2 to max_k to help determine optimal cluster number.
    """
    if len(df) < 5:
        return {'k_values': [], 'inertias': [], 'silhouettes': [], 'error': 'Dataset too small for clustering.'}

    _, X_scaled, _, _ = prepare_clustering_features(df, feature_names)

    effective_max_k = min(max_k, len(df) - 1, 10)
    if effective_max_k < 2:
        return {'k_values': [], 'inertias': [], 'silhouettes': [], 'error': 'Not enough data points.'}

    k_values = list(range(2, effective_max_k + 1))
    inertias = []
    silhouettes = []

    for k in k_values:
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        inertias.append(round(float(kmeans.inertia_), 2))
        try:
            sil = float(silhouette_score(X_scaled, cluster_labels))
            silhouettes.append(round(sil, 4))
        except Exception:
            silhouettes.append(0.0)

    # Determine recommended K (highest silhouette score with reasonable inertia drop)
    best_k = k_values[int(np.argmax(silhouettes))] if silhouettes else 3

    return {
        'k_values': k_values,
        'inertias': inertias,
        'silhouettes': silhouettes,
        'recommended_k': int(best_k)
    }


def generate_market_segment_tag(stats: dict) -> tuple[str, str]:
    """
    Derives genuine agricultural market segment archetype from cluster statistics.
    Returns (segment_title, market_characteristics_description).
    """
    avg_modal = stats.get('avg_modal', 0)
    avg_spread = stats.get('avg_spread', 0)
    avg_spread_pct = stats.get('avg_spread_pct', 0)

    if avg_modal >= 20000:
        return (
            "Premium High-Value Commercial & Spices",
            f"High-capital commercial commodities (e.g. Cardamom, Jeera, Arecanut) traded at elevated price tiers (avg ₹{avg_modal:,.0f}/Q). High investment capital required."
        )
    elif avg_modal >= 8000:
        return (
            "High-Tier Cash Crops & Pulses",
            f"Valuable cash crops, pulses, and quality spices with steady commercial demand (avg ₹{avg_modal:,.0f}/Q) and moderate intra-market spread (₹{avg_spread:,.0f}/Q)."
        )
    elif avg_spread_pct >= 35.0:
        return (
            "High Volatility Perishable & Vegetable Mandis",
            f"Perishable commodities characterized by wide intraday price swings (avg spread {avg_spread_pct:.1f}%), sensitive to daily arrivals and shelf-life."
        )
    elif avg_modal <= 2600 and avg_spread_pct <= 15.0:
        return (
            "Stable Essential Foodgrains & Bulk Staples",
            f"Core MSP-supported grains and staple cereals (e.g. Wheat, Paddy) displaying minimal volatility (avg spread {avg_spread_pct:.1f}%) and stable bulk trading."
        )
    else:
        return (
            "Mid-Tier Commercial & General Produce",
            f"Balanced volume commodities (e.g. Oilseeds, Coarse Cereals) with balanced price points (avg ₹{avg_modal:,.0f}/Q) and normal market dynamics."
        )


def run_kmeans_clustering(df: pd.DataFrame, n_clusters: int = 4, feature_names: list[str] = None) -> dict:
    """
    Performs full K-Means clustering pipeline:
    1. Preprocessing and feature scaling
    2. Training K-Means
    3. Performance metric calculation (Inertia, Silhouette, Davies-Bouldin, Calinski-Harabasz)
    4. 2D PCA projection for visualization
    5. Cluster profiling with actual statistics and agricultural domain interpretation
    """
    if len(df) < n_clusters:
        raise ValueError(f"Dataset has only {len(df)} records, which is less than requested clusters ({n_clusters}).")

    X_raw, X_scaled, scaler, valid_features = prepare_clustering_features(df, feature_names)

    # Train K-Means
    kmeans = KMeans(
        n_clusters=n_clusters,
        init='k-means++',
        n_init=15,
        max_iter=300,
        random_state=42
    )
    labels = kmeans.fit_predict(X_scaled)

    # Attach labels to df copy
    df_result = df.copy()
    df_result['Cluster'] = labels

    # Performance Metrics
    inertia_val = round(float(kmeans.inertia_), 2)
    sil_val = round(float(silhouette_score(X_scaled, labels)), 4) if len(np.unique(labels)) > 1 else 0.0
    db_val = round(float(davies_bouldin_score(X_scaled, labels)), 4) if len(np.unique(labels)) > 1 else 0.0
    ch_val = round(float(calinski_harabasz_score(X_scaled, labels)), 2) if len(np.unique(labels)) > 1 else 0.0

    # 2D PCA Projection for Visualizing High-Dimensional Clusters
    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(X_scaled)
    df_result['PCA_1'] = np.round(pca_coords[:, 0], 3)
    df_result['PCA_2'] = np.round(pca_coords[:, 1], 3)
    pca_variance = [round(float(v) * 100, 2) for v in pca.explained_variance_ratio_]

    # Cluster Profiles & Statistical Summaries
    cluster_profiles = []
    total_samples = len(df_result)

    for cluster_id in range(n_clusters):
        subset = df_result[df_result['Cluster'] == cluster_id]
        count = len(subset)
        pct = round((count / total_samples) * 100, 1)

        avg_modal = round(float(subset['Modal_Price'].mean()), 2)
        min_modal = round(float(subset['Modal_Price'].min()), 2)
        max_modal = round(float(subset['Modal_Price'].max()), 2)
        avg_spread = round(float(subset['Price_Spread'].mean()), 2)
        avg_spread_pct = round(float(subset['Price_Spread_Pct'].mean()), 2)

        # Top commodities and states
        top_commodities = subset['Commodity'].value_counts().head(3).to_dict()
        top_states = subset['State'].value_counts().head(3).to_dict()

        stats = {
            'avg_modal': avg_modal,
            'min_modal': min_modal,
            'max_modal': max_modal,
            'avg_spread': avg_spread,
            'avg_spread_pct': avg_spread_pct
        }
        tag, desc = generate_market_segment_tag(stats)

        cluster_profiles.append({
            'cluster_id': int(cluster_id),
            'record_count': int(count),
            'percentage': pct,
            'avg_modal_price': avg_modal,
            'min_modal_price': min_modal,
            'max_modal_price': max_modal,
            'avg_spread': avg_spread,
            'avg_spread_pct': avg_spread_pct,
            'top_commodities': top_commodities,
            'top_states': top_states,
            'segment_tag': tag,
            'segment_description': desc
        })

    # Prepare scatter data for Plotly / Chart.js
    records_list = []
    for idx, row in df_result.iterrows():
        records_list.append({
            'id': int(idx),
            'state': str(row.get('State', '')),
            'market': str(row.get('Market', '')),
            'commodity': str(row.get('Commodity', '')),
            'variety': str(row.get('Variety', '')),
            'modal_price': float(row.get('Modal_Price', 0)),
            'min_price': float(row.get('Min_Price', 0)),
            'max_price': float(row.get('Max_Price', 0)),
            'price_spread': float(row.get('Price_Spread', 0)),
            'price_spread_pct': float(row.get('Price_Spread_Pct', 0)),
            'cluster': int(row['Cluster']),
            'pca_1': float(row['PCA_1']),
            'pca_2': float(row['PCA_2'])
        })

    return {
        'n_clusters': int(n_clusters),
        'features_used': valid_features,
        'metrics': {
            'inertia': inertia_val,
            'silhouette_score': sil_val,
            'davies_bouldin_index': db_val,
            'calinski_harabasz_index': ch_val,
            'pca_explained_variance_ratio': pca_variance
        },
        'cluster_profiles': cluster_profiles,
        'records': records_list
    }
