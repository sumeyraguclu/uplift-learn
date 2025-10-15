"""
Veri yükleme ve hazırlama fonksiyonları

Bu modül, Criteo uplift veri setini yükleme ve işleme fonksiyonlarını içerir.
scikit-uplift'ten esinlenilerek geliştirilmiştir.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Union
from sklearn.model_selection import train_test_split as sklearn_split


def load_criteo_sample(path: str = "data/criteo_sample.parquet") -> pd.DataFrame:
    """
    Criteo uplift veri setini yükle
    
    Parameters
    ----------
    path : str
        Veri dosyasının yolu (parquet veya csv)
    
    Returns
    -------
    pd.DataFrame
        Yüklenen veri
        
    Raises
    ------
    FileNotFoundError
        Dosya bulunamazsa
    
    Examples
    --------
    >>> df = load_criteo_sample()
    >>> print(df.shape)
    (10000, 16)
    
    >>> print(df.columns.tolist())
    ['f0', 'f1', ..., 'f11', 'treatment', 'visit', 'conversion', 'exposure']
    
    Notes
    -----
    Veri seti yapısı:
    - f0-f11: Özellikler (anonimleştirilmiş)
    - treatment: 1 = reklam gösterildi, 0 = gösterilmedi
    - visit: 1 = siteyi ziyaret etti, 0 = etmedi
    - conversion: 1 = satın aldı, 0 = almadı
    - exposure: 1 = reklamı gördü, 0 = görmedi
    """
    file_path = Path(path)
    
    if not file_path.exists():
        raise FileNotFoundError(
            f"Veri dosyası bulunamadı: {path}\n"
            f"Lütfen önce 'python scripts/prepare_data.py' çalıştırın."
        )
    
    # Dosya formatına göre oku
    if file_path.suffix == '.parquet':
        df = pd.read_parquet(path)
    elif file_path.suffix == '.csv':
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Desteklenmeyen dosya formatı: {file_path.suffix}")
    
    return df


def get_features_target_treatment(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    target: str = 'visit',
    treatment: str = 'treatment'
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    DataFrame'den X, y, treatment'ı ayır
    
    Parameters
    ----------
    df : pd.DataFrame
        Ham veri
    features : list of str, optional
        Kullanılacak özellikler. None ise f0-f11 kullanılır.
    target : str
        Hedef değişken sütun adı
    treatment : str
        Treatment sütun adı
    
    Returns
    -------
    X : pd.DataFrame
        Özellikler
    y : pd.Series
        Hedef değişken
    treatment : pd.Series
        Treatment göstergesi
    
    Examples
    --------
    >>> df = load_criteo_sample()
    >>> X, y, t = get_features_target_treatment(df)
    >>> print(X.shape, y.shape, t.shape)
    (10000, 12) (10000,) (10000,)
    """
    if features is None:
        # Varsayılan: f0-f11
        features = [f'f{i}' for i in range(12)]
    
    X = df[features].copy()
    y = df[target].copy()
    t = df[treatment].copy()
    
    return X, y, t


def train_test_split_uplift(
    df: pd.DataFrame,
    test_size: float = 0.25,
    random_state: int = 42,
    stratify_treatment: bool = True,
    features: Optional[List[str]] = None,
    target: str = 'visit',
    treatment: str = 'treatment'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, 
           pd.Series, pd.Series]:
    """
    Uplift modeling için train/test split
    
    ÖNEMLI: Treatment dengesi korunmalı!
    
    Parameters
    ----------
    df : pd.DataFrame
        Tüm veri
    test_size : float
        Test set oranı (0-1 arası)
    random_state : int
        Random seed
    stratify_treatment : bool
        Treatment oranını korumak için stratify kullan
    features : list of str, optional
        Kullanılacak özellikler
    target : str
        Hedef değişken sütun adı
    treatment : str
        Treatment sütun adı
    
    Returns
    -------
    X_train, X_test : pd.DataFrame
        Train ve test özellikleri
    y_train, y_test : pd.Series
        Train ve test hedef değişkenleri
    t_train, t_test : pd.Series
        Train ve test treatment göstergeleri
    
    Examples
    --------
    >>> df = load_criteo_sample()
    >>> X_train, X_test, y_train, y_test, t_train, t_test = \
    ...     train_test_split_uplift(df, test_size=0.25)
    >>> 
    >>> print(f"Train size: {len(X_train)}")
    >>> print(f"Test size: {len(X_test)}")
    >>> print(f"Treatment ratio (train): {t_train.mean():.2%}")
    >>> print(f"Treatment ratio (test): {t_test.mean():.2%}")
    
    Notes
    -----
    Treatment dengesi kritik! Eğer dengeli olmazsa:
    - Model yanlış öğrenir
    - Metrikler hatalı hesaplanır
    - Optimizasyon bozulur
    """
    # X, y, treatment'ı ayır
    X, y, t = get_features_target_treatment(
        df, features=features, target=target, treatment=treatment
    )
    
    # Stratify için kullanılacak değişken
    stratify_var = t if stratify_treatment else None
    
    # Train/test split
    X_train, X_test, y_train, y_test, t_train, t_test = sklearn_split(
        X, y, t,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_var
    )
    
    return X_train, X_test, y_train, y_test, t_train, t_test


def check_treatment_balance(
    treatment: Union[pd.Series, np.ndarray],
    data_name: str = "Dataset"
) -> dict:
    """
    Treatment grubunun dengesini kontrol et
    
    Parameters
    ----------
    treatment : array-like
        Treatment göstergesi (0 veya 1)
    data_name : str
        Veri setinin adı (raporlama için)
    
    Returns
    -------
    dict
        Treatment istatistikleri
    
    Examples
    --------
    >>> stats = check_treatment_balance(t_train, "Training Set")
    >>> print(f"Treatment ratio: {stats['treatment_ratio']:.2%}")
    """
    treatment = np.asarray(treatment)
    
    n_total = len(treatment)
    n_treatment = np.sum(treatment == 1)
    n_control = np.sum(treatment == 0)
    ratio = n_treatment / n_total
    
    stats = {
        'total': n_total,
        'treatment': n_treatment,
        'control': n_control,
        'treatment_ratio': ratio,
        'control_ratio': 1 - ratio
    }
    
    print(f"\n{'='*60}")
    print(f"🎯 TREATMENT BALANCE: {data_name}")
    print(f"{'='*60}")
    print(f"Total samples:     {n_total:>8,}")
    print(f"Treatment group:   {n_treatment:>8,} ({ratio:.2%})")
    print(f"Control group:     {n_control:>8,} ({1-ratio:.2%})")
    
    # Uyarı: Çok dengesizse
    if ratio < 0.3 or ratio > 0.7:
        print("\n⚠️  WARNING: Treatment groups are imbalanced!")
        print("   Recommended: 30%-70% treatment ratio")
    else:
        print("\n✅ Treatment groups are balanced")
    
    print("="*60)
    
    return stats


def calculate_baseline_metrics(
    y: Union[pd.Series, np.ndarray],
    treatment: Union[pd.Series, np.ndarray]
) -> dict:
    """
    Baseline metriklerini hesapla (ATE, conversion rates)
    
    Parameters
    ----------
    y : array-like
        Outcome (0 veya 1)
    treatment : array-like
        Treatment göstergesi (0 veya 1)
    
    Returns
    -------
    dict
        Baseline metrikleri
    
    Examples
    --------
    >>> metrics = calculate_baseline_metrics(y_train, t_train)
    >>> print(f"ATE: {metrics['ate']:.4f}")
    >>> print(f"Relative uplift: {metrics['relative_uplift']:.2%}")
    
    Notes
    -----
    ATE (Average Treatment Effect):
        ATE = E[Y|T=1] - E[Y|T=0]
    
    Relative Uplift:
        RU = (E[Y|T=1] - E[Y|T=0]) / E[Y|T=0]
    """
    y = np.asarray(y)
    treatment = np.asarray(treatment)
    
    # Conversion rates
    cr_treatment = y[treatment == 1].mean()
    cr_control = y[treatment == 0].mean()
    
    # ATE
    ate = cr_treatment - cr_control
    
    # Relative uplift
    relative_uplift = ate / cr_control if cr_control > 0 else 0
    
    metrics = {
        'conversion_rate_treatment': cr_treatment,
        'conversion_rate_control': cr_control,
        'ate': ate,
        'relative_uplift': relative_uplift
    }
    
    print(f"\n{'='*60}")
    print(f"📊 BASELINE METRICS")
    print(f"{'='*60}")
    print(f"Conversion rate (treatment): {cr_treatment:.4f} ({cr_treatment:.2%})")
    print(f"Conversion rate (control):   {cr_control:.4f} ({cr_control:.2%})")
    print(f"ATE (absolute):              {ate:+.4f}")
    print(f"Relative uplift:             {relative_uplift:+.2%}")
    
    if ate > 0:
        print("\n✅ Treatment has POSITIVE effect")
    elif ate < 0:
        print("\n⚠️  Treatment has NEGATIVE effect")
    else:
        print("\n➖ Treatment has NO effect")
    
    print("="*60)
    
    return metrics


def create_toy_dataset(
    n_samples: int = 1000,
    n_features: int = 5,
    treatment_effect_size: float = 0.1,
    noise_level: float = 0.3,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Simülasyon için toy dataset oluştur
    
    Parameters
    ----------
    n_samples : int
        Örnek sayısı
    n_features : int
        Özellik sayısı
    treatment_effect_size : float
        Ortalama tedavi etkisi
    noise_level : float
        Gürültü seviyesi
    random_state : int
        Random seed
    
    Returns
    -------
    pd.DataFrame
        Simüle edilmiş veri
    
    Examples
    --------
    >>> df = create_toy_dataset(n_samples=1000)
    >>> X, y, t = get_features_target_treatment(df)
    
    Notes
    -----
    Bu fonksiyon test ve debugging için kullanışlıdır.
    Gerçek uplift behavior'ı simüle eder.
    """
    np.random.seed(random_state)
    
    # Özellikler
    X = np.random.randn(n_samples, n_features)
    
    # Treatment assignment (randomized)
    treatment = np.random.binomial(1, 0.5, n_samples)
    
    # Base response probability (özelliklere bağlı)
    base_prob = 1 / (1 + np.exp(-X[:, 0] * 0.5))
    
    # Treatment effect (heterogeneous)
    treatment_effect = treatment_effect_size + X[:, 1] * 0.1
    
    # Final probability
    prob = base_prob + treatment * treatment_effect + np.random.randn(n_samples) * noise_level
    prob = np.clip(prob, 0, 1)
    
    # Outcome
    y = np.random.binomial(1, prob)
    
    # DataFrame oluştur
    df = pd.DataFrame(
        X,
        columns=[f'f{i}' for i in range(n_features)]
    )
    df['treatment'] = treatment
    df['visit'] = y
    
    return df


# Kolaylık fonksiyonları (scikit-uplift tarzı)
def load_criteo(sample_size: Optional[int] = None) -> pd.DataFrame:
    """
    Criteo veri setini yükle (kısa isim)
    
    Parameters
    ----------
    sample_size : int, optional
        Örnek boyutu. None ise tam veri.
    
    Returns
    -------
    pd.DataFrame
    """
    if sample_size:
        path = "data/criteo_sample.parquet"
    else:
        path = "data/criteo_uplift_full.parquet"
    
    return load_criteo_sample(path)


if __name__ == "__main__":
    # Test kodu
    print("🧪 Testing data module...\n")
    
    # Toy dataset test
    print("1️⃣ Creating toy dataset...")
    df_toy = create_toy_dataset(n_samples=1000)
    print(f"✅ Created: {df_toy.shape}")
    
    # Split test
    print("\n2️⃣ Testing train/test split...")
    X_train, X_test, y_train, y_test, t_train, t_test = \
        train_test_split_uplift(df_toy, test_size=0.25)
    
    # Balance check
    check_treatment_balance(t_train, "Training Set")
    check_treatment_balance(t_test, "Test Set")
    
    # Baseline metrics
    calculate_baseline_metrics(y_train, t_train)
    
    print("\n✅ All tests passed!")