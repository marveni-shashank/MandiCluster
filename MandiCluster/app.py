"""
MandiCluster - Flask Application
Market & Price-Pattern Segmentation for Agricultural Commodities using Unsupervised Machine Learning.
"""

import os
import json
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import pandas as pd
import numpy as np

from data_loader import (
    load_dataset,
    get_dataset_summary,
    validate_csv_structure
)
from ml_engine import (
    run_kmeans_clustering,
    compute_elbow_curve,
    FEATURE_DESCRIPTIONS
)

app = Flask(__name__)
app.secret_key = 'mandicluster-super-secret-key-agritech-2026'

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DEFAULT_DATA_PATH = os.path.join(BASE_DIR, 'data', 'agmarknet_sample.csv')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload

# Global active dataset state
ACTIVE_DATASET_PATH = DEFAULT_DATA_PATH
CURRENT_DF = None


def get_current_data() -> pd.DataFrame:
    """Helper to ensure active dataset is loaded in memory."""
    global CURRENT_DF, ACTIVE_DATASET_PATH
    if CURRENT_DF is None or CURRENT_DF.empty:
        if os.path.exists(ACTIVE_DATASET_PATH):
            CURRENT_DF = load_dataset(ACTIVE_DATASET_PATH)
        elif os.path.exists(DEFAULT_DATA_PATH):
            ACTIVE_DATASET_PATH = DEFAULT_DATA_PATH
            CURRENT_DF = load_dataset(DEFAULT_DATA_PATH)
        else:
            CURRENT_DF = pd.DataFrame()
    return CURRENT_DF


# ==========================================
# PAGE ROUTES
# ==========================================

@app.route('/')
def home():
    """Home Page - Project overview, architecture, and technology highlights."""
    df = get_current_data()
    summary = get_dataset_summary(df)
    return render_template('index.html', summary=summary)


@app.route('/analytics')
def analytics():
    """Market Analytics Page - Interactive price comparison, filters, and mandi trends."""
    df = get_current_data()
    commodities = sorted(df['Commodity'].dropna().unique().tolist()) if not df.empty else []
    states = sorted(df['State'].dropna().unique().tolist()) if not df.empty else []
    markets = sorted(df['Market'].dropna().unique().tolist()) if not df.empty else []
    return render_template('analytics.html', commodities=commodities, states=states, markets=markets)


@app.route('/segmentation')
def segmentation():
    """Price Pattern Segmentation Page - Scikit-learn K-Means clustering interface."""
    df = get_current_data()
    summary = get_dataset_summary(df)
    return render_template(
        'segmentation.html',
        features=FEATURE_DESCRIPTIONS,
        total_records=summary.get('total_records', 0)
    )


@app.route('/performance')
def performance():
    """Model Performance Page - WCSS/Inertia, Silhouette score, and clustering evaluation."""
    return render_template('performance.html')


@app.route('/explorer')
def explorer():
    """Dataset Explorer Page - Data inspection, column statistics, and CSV upload."""
    df = get_current_data()
    summary = get_dataset_summary(df)
    is_custom = ACTIVE_DATASET_PATH != DEFAULT_DATA_PATH
    current_filename = os.path.basename(ACTIVE_DATASET_PATH)
    return render_template(
        'explorer.html',
        summary=summary,
        is_custom=is_custom,
        current_filename=current_filename
    )


@app.route('/about')
def about():
    """About Project Page - Academic documentation, problem statement, methodology."""
    return render_template('about.html')


# ==========================================
# REST API ENDPOINTS
# ==========================================

@app.route('/api/dataset-info', methods=['GET'])
def api_dataset_info():
    """Returns dataset overview and statistics."""
    df = get_current_data()
    summary = get_dataset_summary(df)
    return jsonify(summary)


@app.route('/api/market-data', methods=['GET'])
def api_market_data():
    """Returns filtered market data records and aggregated metrics."""
    df = get_current_data()
    if df.empty:
        return jsonify({'records': [], 'stats': {}, 'comparison': []})

    commodity_filter = request.args.get('commodity', '').strip()
    state_filter = request.args.get('state', '').strip()
    market_filter = request.args.get('market', '').strip()
    search_query = request.args.get('search', '').strip().lower()

    filtered = df.copy()

    if commodity_filter and commodity_filter != 'ALL':
        filtered = filtered[filtered['Commodity'] == commodity_filter]

    if state_filter and state_filter != 'ALL':
        filtered = filtered[filtered['State'] == state_filter]

    if market_filter and market_filter != 'ALL':
        filtered = filtered[filtered['Market'] == market_filter]

    if search_query:
        mask = (
            filtered['Commodity'].str.lower().str.contains(search_query) |
            filtered['Market'].str.lower().str.contains(search_query) |
            filtered['State'].str.lower().str.contains(search_query) |
            filtered['District'].str.lower().str.contains(search_query)
        )
        filtered = filtered[mask]

    # Calculate statistics on filtered set
    stats = {}
    if not filtered.empty:
        stats = {
            'count': int(len(filtered)),
            'avg_modal_price': round(float(filtered['Modal_Price'].mean()), 2),
            'min_price': round(float(filtered['Min_Price'].min()), 2),
            'max_price': round(float(filtered['Max_Price'].max()), 2),
            'avg_spread': round(float(filtered['Price_Spread'].mean()), 2),
            'avg_spread_pct': round(float(filtered['Price_Spread_Pct'].mean()), 2)
        }

    # Top comparison items (for charts)
    top_records = filtered.head(25).to_dict(orient='records')

    # Price comparison by commodity or market
    group_col = 'Commodity' if (not commodity_filter or commodity_filter == 'ALL') else 'Market'
    grouped = (
        filtered.groupby(group_col)
        .agg({
            'Modal_Price': 'mean',
            'Min_Price': 'mean',
            'Max_Price': 'mean',
            'Price_Spread': 'mean'
        })
        .round(2)
        .reset_index()
        .sort_values(by='Modal_Price', ascending=False)
        .head(15)
        .to_dict(orient='records')
    )

    return jsonify({
        'stats': stats,
        'records': top_records,
        'group_comparison': grouped,
        'group_by': group_col,
        'total_filtered': len(filtered)
    })


@app.route('/api/elbow-analysis', methods=['GET'])
def api_elbow_analysis():
    """Computes Elbow Curve (WCSS) and Silhouette scores for k=2..8."""
    df = get_current_data()
    if df.empty or len(df) < 5:
        return jsonify({'error': 'Insufficient dataset records for clustering evaluation.'}), 400

    features = request.args.getlist('features')
    if not features:
        features = ['Modal_Price', 'Price_Spread', 'Price_Spread_Pct']

    result = compute_elbow_curve(df, feature_names=features, max_k=8)
    return jsonify(result)


@app.route('/api/run-clustering', methods=['POST'])
def api_run_clustering():
    """Executes K-Means clustering with user-specified K and features."""
    df = get_current_data()
    if df.empty or len(df) < 5:
        return jsonify({'error': 'Insufficient dataset records to perform clustering.'}), 400

    data = request.get_json() or {}
    try:
        n_clusters = int(data.get('n_clusters', 4))
    except (TypeError, ValueError):
        n_clusters = 4

    features = data.get('features', ['Modal_Price', 'Price_Spread', 'Price_Spread_Pct'])
    if not isinstance(features, list) or len(features) == 0:
        features = ['Modal_Price', 'Price_Spread', 'Price_Spread_Pct']

    if n_clusters < 2:
        n_clusters = 2
    if n_clusters > min(10, len(df) - 1):
        n_clusters = min(10, len(df) - 1)

    try:
        clustering_result = run_kmeans_clustering(df, n_clusters=n_clusters, feature_names=features)
        return jsonify(clustering_result)
    except Exception as e:
        return jsonify({'error': f'Clustering execution error: {str(e)}'}), 500


@app.route('/api/records-table', methods=['GET'])
def api_records_table():
    """Provides paginated, searchable records for the Dataset Explorer table."""
    df = get_current_data()
    if df.empty:
        return jsonify({'total': 0, 'rows': []})

    page = int(request.args.get('page', 1))
    page_size = int(request.args.get('page_size', 15))
    search = request.args.get('search', '').strip().lower()

    filtered = df.copy()
    if search:
        mask = (
            filtered['Commodity'].str.lower().str.contains(search) |
            filtered['Market'].str.lower().str.contains(search) |
            filtered['State'].str.lower().str.contains(search) |
            filtered['Variety'].str.lower().str.contains(search)
        )
        filtered = filtered[mask]

    total = len(filtered)
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    rows = filtered.iloc[start_idx:end_idx].to_dict(orient='records')

    return jsonify({
        'total': total,
        'page': page,
        'page_size': page_size,
        'total_pages': max(1, (total + page_size - 1) // page_size),
        'rows': rows
    })


@app.route('/api/upload-dataset', methods=['POST'])
def api_upload_dataset():
    """Handles CSV dataset file uploads, structural validation, and caching."""
    global ACTIVE_DATASET_PATH, CURRENT_DF

    if 'file' not in request.files:
        flash('No file part selected in upload request.', 'danger')
        return redirect(url_for('explorer'))

    file = request.files['file']
    if file.filename == '':
        flash('No file selected.', 'warning')
        return redirect(url_for('explorer'))

    if not file.filename.lower().endswith('.csv'):
        flash('Invalid file format. Please upload a standard .CSV file.', 'danger')
        return redirect(url_for('explorer'))

    save_path = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_agmarknet.csv')
    try:
        file.save(save_path)
        is_valid, msg = validate_csv_structure(save_path)

        if not is_valid:
            if os.path.exists(save_path):
                os.remove(save_path)
            flash(f'Dataset validation failed: {msg}', 'danger')
            return redirect(url_for('explorer'))

        # Load and verify
        new_df = load_dataset(save_path)
        if len(new_df) < 5:
            if os.path.exists(save_path):
                os.remove(save_path)
            flash('Uploaded dataset contains fewer than 5 valid records with prices.', 'danger')
            return redirect(url_for('explorer'))

        ACTIVE_DATASET_PATH = save_path
        CURRENT_DF = new_df
        flash(f'Success! Successfully loaded {len(new_df)} records from {file.filename}.', 'success')
        return redirect(url_for('explorer'))

    except Exception as e:
        flash(f'Error processing CSV file: {str(e)}', 'danger')
        return redirect(url_for('explorer'))


@app.route('/api/reset-dataset', methods=['POST'])
def api_reset_dataset():
    """Resets the active dataset back to the built-in Agmarknet agricultural dataset."""
    global ACTIVE_DATASET_PATH, CURRENT_DF
    ACTIVE_DATASET_PATH = DEFAULT_DATA_PATH
    CURRENT_DF = load_dataset(DEFAULT_DATA_PATH)
    flash('Reset successfully to the default verified Agmarknet agricultural dataset.', 'info')
    return redirect(url_for('explorer'))


if __name__ == '__main__':
    # Initialize dataset
    get_current_data()
    print("MandiCluster Server running at http://127.0.0.1:5000")
    app.run(host='127.0.0.1', port=5000, debug=True)
