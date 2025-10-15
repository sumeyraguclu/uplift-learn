"""
Unit tests for data.py module

Çalıştırmak için:
    pytest tests/test_data.py -v
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Modülü import et
sys.path.append(str(Path(__file__).parent.parent))
from src.data import (
    get_features_target_treatment,
    train_test_split_uplift,
    check_treatment_balance,
    calculate_baseline_metrics,
    create_toy_dataset
)


class TestToyDataset:
    """Toy dataset oluşturma testleri"""
    
    def test_toy_dataset_shape(self):
        """Dataset boyutu doğru mu?"""
        df = create_toy_dataset(n_samples=1000, n_features=5)
        assert df.shape == (1000, 7)  # 5 features + treatment + visit
    
    def test_toy_dataset_columns(self):
        """Sütunlar doğru mu?"""
        df = create_toy_dataset(n_samples=100, n_features=3)
        expected_cols = ['f0', 'f1', 'f2', 'treatment', 'visit']
        assert list(df.columns) == expected_cols
    
    def test_toy_dataset_binary(self):
        """Treatment ve outcome binary mi?"""
        df = create_toy_dataset(n_samples=100)
        assert set(df['treatment'].unique()).issubset({0, 1})
        assert set(df['visit'].unique()).issubset({0, 1})
    
    def test_toy_dataset_reproducible(self):
        """Aynı seed aynı sonucu verir mi?"""
        df1 = create_toy_dataset(random_state=42)
        df2 = create_toy_dataset(random_state=42)
        pd.testing.assert_frame_equal(df1, df2)


class TestGetFeaturesTargetTreatment:
    """Feature/target/treatment ayırma testleri"""
    
    @pytest.fixture
    def sample_df(self):
        """Test için örnek DataFrame"""
        return create_toy_dataset(n_samples=100, n_features=3)
    
    def test_split_shapes(self, sample_df):
        """Boyutlar doğru mu?"""
        X, y, t = get_features_target_treatment(sample_df)
        assert X.shape == (100, 3)
        assert y.shape == (100,)
        assert t.shape == (100,)
    
    def test_split_types(self, sample_df):
        """Veri tipleri doğru mu?"""
        X, y, t = get_features_target_treatment(sample_df)
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert isinstance(t, pd.Series)
    
    def test_custom_features(self, sample_df):
        """Özel feature seçimi çalışıyor mu?"""
        X, y, t = get_features_target_treatment(sample_df, features=['f0', 'f1'])
        assert list(X.columns) == ['f0', 'f1']
    
    def test_custom_target(self, sample_df):
        """Özel target seçimi çalışıyor mu?"""
        X, y, t = get_features_target_treatment(sample_df, target='visit')
        assert y.name == 'visit'


class TestTrainTestSplit:
    """Train/test split testleri"""
    
    @pytest.fixture
    def sample_df(self):
        return create_toy_dataset(n_samples=1000, n_features=5)
    
    def test_split_sizes(self, sample_df):
        """Split oranı doğru mu?"""
        X_train, X_test, y_train, y_test, t_train, t_test = \
            train_test_split_uplift(sample_df, test_size=0.25)
        
        assert len(X_train) == 750
        assert len(X_test) == 250
    
    def test_no_data_leakage(self, sample_df):
        """Train ve test arasında veri sızıntısı var mı?"""
        X_train, X_test, y_train, y_test, t_train, t_test = \
            train_test_split_uplift(sample_df, test_size=0.25, random_state=42)
        
        # Index'ler kesişmemeli
        train_idx = X_train.index
        test_idx = X_test.index
        assert len(set(train_idx) & set(test_idx)) == 0
    
    def test_stratify_treatment(self, sample_df):
        """Treatment dengesi korunuyor mu?"""
        X_train, X_test, y_train, y_test, t_train, t_test = \
            train_test_split_uplift(sample_df, test_size=0.25, stratify_treatment=True)
        
        # Treatment oranları benzer olmalı
        ratio_train = t_train.mean()
        ratio_test = t_test.mean()
        assert abs(ratio_train - ratio_test) < 0.05  # %5'ten az fark
    
    def test_reproducible(self, sample_df):
        """Aynı random_state aynı split'i verir mi?"""
        result1 = train_test_split_uplift(sample_df, random_state=42)
        result2 = train_test_split_uplift(sample_df, random_state=42)
        
        # X_train'ler aynı olmalı
        pd.testing.assert_frame_equal(result1[0], result2[0])


class TestTreatmentBalance:
    """Treatment dengesi kontrolü testleri"""
    
    def test_balanced_treatment(self):
        """Dengeli treatment'ı tespit ediyor mu?"""
        t = np.array([0, 1] * 500)  # 50-50 dengeli
        stats = check_treatment_balance(t, "Test")
        
        assert stats['treatment_ratio'] == 0.5
        assert stats['control_ratio'] == 0.5
    
    def test_imbalanced_treatment(self):
        """Dengesiz treatment'ı tespit ediyor mu?"""
        t = np.array([0] * 100 + [1] * 900)  # 10-90 dengesiz
        stats = check_treatment_balance(t, "Test")
        
        assert stats['treatment_ratio'] == 0.9
        assert stats['control_ratio'] == 0.1
    
    def test_return_dict(self):
        """Doğru dictionary döndürüyor mu?"""
        t = np.array([0, 1, 0, 1])
        stats = check_treatment_balance(t, "Test")
        
        required_keys = ['total', 'treatment', 'control', 
                        'treatment_ratio', 'control_ratio']
        assert all(key in stats for key in required_keys)


class TestBaselineMetrics:
    """Baseline metrik testleri"""
    
    def test_positive_ate(self):
        """Pozitif ATE doğru hesaplanıyor mu?"""
        # Treatment grubunda daha yüksek conversion
        y = np.array([0, 0, 1, 1, 0, 1, 1, 1])
        t = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        
        metrics = calculate_baseline_metrics(y, t)
        
        assert metrics['conversion_rate_treatment'] == 0.75  # 3/4
        assert metrics['conversion_rate_control'] == 0.25    # 1/4
        assert metrics['ate'] == 0.5  # 0.75 - 0.25
        assert metrics['relative_uplift'] == 2.0  # 0.5 / 0.25
    
    def test_negative_ate(self):
        """Negatif ATE doğru hesaplanıyor mu?"""
        # Control grubunda daha yüksek conversion
        y = np.array([1, 1, 1, 1, 0, 0, 0, 1])
        t = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        
        metrics = calculate_baseline_metrics(y, t)
        
        assert metrics['conversion_rate_treatment'] == 0.25  # 1/4
        assert metrics['conversion_rate_control'] == 1.0     # 4/4
        assert metrics['ate'] == -0.75  # 0.25 - 1.0
    
    def test_zero_ate(self):
        """ATE = 0 doğru hesaplanıyor mu?"""
        # Her iki grup da aynı
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        t = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        
        metrics = calculate_baseline_metrics(y, t)
        
        assert metrics['ate'] == 0.0
        assert metrics['relative_uplift'] == 0.0
    
    def test_return_dict(self):
        """Doğru dictionary döndürüyor mu?"""
        y = np.array([0, 1, 0, 1])
        t = np.array([0, 0, 1, 1])
        
        metrics = calculate_baseline_metrics(y, t)
        
        required_keys = ['conversion_rate_treatment', 'conversion_rate_control',
                        'ate', 'relative_uplift']
        assert all(key in metrics for key in required_keys)


class TestEdgeCases:
    """Edge case testleri"""
    
    def test_empty_treatment_group(self):
        """Boş treatment grubu ile başa çıkabiliyor mu?"""
        y = np.array([0, 1, 0, 1])
        t = np.array([0, 0, 0, 0])  # Hepsi control
        
        # Error vermemeli, ama sonuç mantıksız olacak
        metrics = calculate_baseline_metrics(y, t)
        assert np.isnan(metrics['conversion_rate_treatment']) or \
               metrics['conversion_rate_treatment'] == 0
    
    def test_all_zeros_outcome(self):
        """Tüm outcome'lar 0 ise?"""
        y = np.zeros(100)
        t = np.array([0, 1] * 50)
        
        metrics = calculate_baseline_metrics(y, t)
        assert metrics['ate'] == 0.0
    
    def test_all_ones_outcome(self):
        """Tüm outcome'lar 1 ise?"""
        y = np.ones(100)
        t = np.array([0, 1] * 50)
        
        metrics = calculate_baseline_metrics(y, t)
        assert metrics['ate'] == 0.0


# Integration test
class TestFullPipeline:
    """Tüm pipeline'ın entegrasyon testi"""
    
    def test_end_to_end(self):
        """Baştan sona pipeline çalışıyor mu?"""
        # 1. Veri oluştur
        df = create_toy_dataset(n_samples=1000, n_features=5, random_state=42)
        
        # 2. Split yap
        X_train, X_test, y_train, y_test, t_train, t_test = \
            train_test_split_uplift(df, test_size=0.25, random_state=42)
        
        # 3. Balance kontrol et
        stats_train = check_treatment_balance(t_train, "Train")
        stats_test = check_treatment_balance(t_test, "Test")
        
        # 4. Metrikleri hesapla
        metrics_train = calculate_baseline_metrics(y_train, t_train)
        metrics_test = calculate_baseline_metrics(y_test, t_test)
        
        # Assertions
        assert len(X_train) + len(X_test) == 1000
        assert 'ate' in metrics_train
        assert 'ate' in metrics_test
        assert stats_train['total'] == 750
        assert stats_test['total'] == 250


if __name__ == "__main__":
    # pytest'i kod içinden çalıştır
    pytest.main([__file__, '-v', '--tb=short'])