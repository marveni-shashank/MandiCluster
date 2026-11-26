"""
MandiCluster - Automated Integration & Route Verification Test
Validates all Flask endpoints, Machine Learning workflows, and API responses.
"""

import sys
import io
import os
import unittest
import json

# Ensure MandiCluster directory is in sys.path
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app import app, get_current_data


class MandiClusterTestSuite(unittest.TestCase):

    def setUp(self):
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        self.client = app.test_client()

    def test_01_home_page(self):
        """Verify home page loads successfully with status 200 and key text."""
        res = self.client.get('/')
        self.assertEqual(res.status_code, 200)
        self.assertIn(b'MandiCluster', res.data)
        self.assertIn(b'Agricultural Market Price Analysis Overview', res.data)
        print("PASS: Home page route verified.")

    def test_02_analytics_page(self):
        """Verify market analytics page loads with 200."""
        res = self.client.get('/analytics')
        self.assertEqual(res.status_code, 200)
        self.assertIn(b'Agricultural Market Analytics', res.data)
        print("PASS: Market Analytics page route verified.")

    def test_03_segmentation_page(self):
        """Verify price pattern segmentation page loads with 200."""
        res = self.client.get('/segmentation')
        self.assertEqual(res.status_code, 200)
        self.assertIn(b'Price Pattern Segmentation', res.data)
        print("PASS: Price Pattern Segmentation page route verified.")

    def test_04_performance_page(self):
        """Verify model performance page loads with 200."""
        res = self.client.get('/performance')
        self.assertEqual(res.status_code, 200)
        self.assertIn(b'Model Performance & Evaluation', res.data)
        self.assertIn(b'The Elbow Method', res.data)
        print("PASS: Model Performance page route verified.")

    def test_05_explorer_page(self):
        """Verify dataset explorer page loads with 200."""
        res = self.client.get('/explorer')
        self.assertEqual(res.status_code, 200)
        self.assertIn(b'Dataset Explorer', res.data)
        print("PASS: Dataset Explorer page route verified.")

    def test_06_about_page(self):
        """Verify about project page loads with 200."""
        res = self.client.get('/about')
        self.assertEqual(res.status_code, 200)
        self.assertIn(b'Project Documentation', res.data)
        print("PASS: About Project page route verified.")

    def test_07_api_dataset_info(self):
        """Verify dataset summary API returns correct structure."""
        res = self.client.get('/api/dataset-info')
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertIn('total_records', data)
        self.assertGreater(data['total_records'], 50)
        self.assertIn('unique_commodities', data)
        self.assertIn('price_summary', data)
        print(f"PASS: Dataset info API verified. Total records: {data['total_records']}")

    def test_08_api_market_data(self):
        """Verify market analytics data API with and without filters."""
        # Unfiltered
        res = self.client.get('/api/market-data')
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertIn('records', data)
        self.assertIn('stats', data)
        self.assertIn('avg_modal_price', data['stats'])

        # Filtered by commodity
        res_filter = self.client.get('/api/market-data?commodity=Wheat')
        self.assertEqual(res_filter.status_code, 200)
        filter_data = json.loads(res_filter.data)
        self.assertGreater(filter_data['total_filtered'], 0)
        for r in filter_data['records']:
            self.assertEqual(r['Commodity'], 'Wheat')
        print(f"PASS: Market data API verified with filtering. Wheat count: {filter_data['total_filtered']}")

    def test_09_api_elbow_analysis(self):
        """Verify elbow method calculations and optimal K output."""
        res = self.client.get('/api/elbow-analysis')
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertIn('k_values', data)
        self.assertIn('inertias', data)
        self.assertIn('silhouettes', data)
        self.assertIn('recommended_k', data)
        self.assertTrue(2 <= data['recommended_k'] <= 8)
        print(f"PASS: Elbow analysis API verified. Recommended K: {data['recommended_k']}, Inertias: {data['inertias']}")

    def test_10_api_run_clustering(self):
        """Verify K-Means clustering execution for K=4."""
        payload = {
            'n_clusters': 4,
            'features': ['Modal_Price', 'Price_Spread', 'Price_Spread_Pct']
        }
        res = self.client.post(
            '/api/run-clustering',
            data=json.dumps(payload),
            content_type='application/json'
        )
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertEqual(data['n_clusters'], 4)
        self.assertEqual(len(data['cluster_profiles']), 4)
        self.assertIn('metrics', data)
        self.assertIn('silhouette_score', data['metrics'])
        self.assertIn('pca_explained_variance_ratio', data['metrics'])
        self.assertGreater(len(data['records']), 50)
        print(f"PASS: Clustering API verified for K=4. Silhouette: {data['metrics']['silhouette_score']}, Inertia: {data['metrics']['inertia']}")

    def test_11_api_records_table(self):
        """Verify paginated records table API."""
        res = self.client.get('/api/records-table?page=1&page_size=10&search=Onion')
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertIn('rows', data)
        self.assertIn('total', data)
        self.assertIn('total_pages', data)
        print(f"PASS: Paginated records table API verified. Found {data['total']} Onion records.")

    def test_12_api_upload_and_validation(self):
        """Verify CSV upload validation rejects non-csv or empty files and handles uploads."""
        # Test non-csv rejection
        data = {'file': (io.BytesIO(b"hello world"), 'test.txt')}
        res = self.client.post('/api/upload-dataset', data=data, content_type='multipart/form-data', follow_redirects=True)
        self.assertEqual(res.status_code, 200)
        self.assertIn(b'Invalid file format', res.data)

        # Test valid CSV upload
        valid_csv = (
            b"State,District,Market,Commodity,Variety,Arrival_Date,Min_Price,Max_Price,Modal_Price\n"
            b"Punjab,Ludhiana,Khanna,Wheat,HD-2967,15/03/2024,2250,2350,2300\n"
            b"Haryana,Karnal,Karnal,Rice,Basmati,15/03/2024,3500,4200,3850\n"
            b"Maharashtra,Nashik,Lasalgaon,Onion,Red,15/03/2024,1200,1800,1500\n"
            b"Gujarat,Rajkot,Gondal,Groundnut,Local,15/03/2024,5500,6500,6000\n"
            b"Karnataka,Kolar,Kolar,Tomato,Hybrid,15/03/2024,1100,1700,1400\n"
            b"Rajasthan,Kota,Kota,Soybean,Yellow,15/03/2024,4200,4700,4500\n"
        )
        upload_data = {'file': (io.BytesIO(valid_csv), 'custom_agri.csv')}
        res_upload = self.client.post('/api/upload-dataset', data=upload_data, content_type='multipart/form-data', follow_redirects=True)
        self.assertEqual(res_upload.status_code, 200)
        self.assertIn(b'Successfully loaded 6 records', res_upload.data)

        # Test reset to default
        res_reset = self.client.post('/api/reset-dataset', follow_redirects=True)
        self.assertEqual(res_reset.status_code, 200)
        self.assertIn(b'Reset successfully to the default', res_reset.data)
        print("PASS: Upload validation and dataset reset verified.")


if __name__ == '__main__':
    unittest.main()
