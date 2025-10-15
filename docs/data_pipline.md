# 📊 Veri Pipeline Kullanım Kılavuzu

Bu dokümantasyon, `src/data.py` modülünün nasıl kullanılacağını açıklar.

---

## 🚀 Hızlı Başlangıç

### 1. Veriyi İndir ve Hazırla

```bash
# Küçük örnek oluştur (öğrenme için)
python scripts/prepare_data.py

# Veya Python içinden:
python
>>> from scripts.prepare_data import download_criteo_dataset
>>> download_criteo_dataset(sample_size=10_000)
```

### 2. Veriyi Yükle

```python
from src.data import load_criteo_sample

# Veriyi yükle
df = load_criteo_sample("data/criteo_sample.parquet")
print(df.head())
```

### 3. Train/Test Split

```python
from src.data import train_test_split_uplift

# Split yap
X_train, X_test, y_train, y_test, t_train, t_test = \
    train_test_split_uplift(df, test_size=0.25, random_state=42)

print(f"Train size: {len(X_train)}")
print(f"Test size: {len(X_test)}")
```

### 4. Baseline Metrikleri Hesapla

```python
from src.data import calculate_baseline_metrics

metrics = calculate_baseline_metrics(y_train, t_train)
print(f"ATE: {metrics['ate']:.4f}")
print(f"Relative uplift: {metrics['relative_uplift']:.2%}")
```

---

## 📚 Fonksiyon Referansı

### `load_criteo_sample(path)`

Criteo uplift veri setini yükler.

**Parametreler:**
- `path` (str): Veri dosyasının yolu (parquet veya csv)

**Döndürür:**
- `pd.DataFrame`: Yüklenen veri

**Örnek:**
```python
df = load_criteo_sample("data/criteo_sample.parquet")
```

**Veri Seti Yapısı:**
```
Columns:
- f0, f1, ..., f11: Özellikler (anonimleştirilmiş)
- treatment: 1 = reklam gösterildi, 0 = gösterilmedi
- visit: 1 = siteyi ziyaret etti, 0 = etmedi
- conversion: 1 = satın aldı, 0 = almadı (varsa)
- exposure: 1 = reklamı gördü, 0 = görmedi (varsa)
```

---

### `get_features_target_treatment(df, features, target, treatment)`

DataFrame'den X, y, treatment'ı ayırır.

**Parametreler:**
- `df` (pd.DataFrame): Ham veri
- `features` (list, optional): Kullanılacak özellikler. None ise f0-f11 kullanılır
- `target` (str): Hedef değişken sütun adı (varsayılan: 'visit')
- `treatment` (str): Treatment sütun adı (varsayılan: 'treatment')

**Döndürür:**
- `X` (pd.DataFrame): Özellikler
- `y` (pd.Series): Hedef değişken
- `treatment` (pd.Series): Treatment göstergesi

**Örnek:**
```python
# Varsayılan kullanım
X, y, t = get_features_target_treatment(df)

# Özel feature seçimi
X, y, t = get_features_target_treatment(
    df, 
    features=['f0', 'f1', 'f2'],
    target='conversion'
)
```

---

### `train_test_split_uplift(df, test_size, random_state, stratify_treatment, ...)`

Uplift modeling için train/test split yapar.

**⚠️ ÖNEMLİ:** Treatment dengesi korunmalı!

**Parametreler:**
- `df` (pd.DataFrame): Tüm veri
- `test_size` (float): Test set oranı (0-1 arası, varsayılan: 0.25)
- `random_state` (int): Random seed (varsayılan: 42)
- `stratify_treatment` (bool): Treatment oranını koru (varsayılan: True)
- `features` (list, optional): Kullanılacak özellikler
- `target` (str): Hedef değişken sütun adı
- `treatment` (str): Treatment sütun adı

**Döndürür:**
- `X_train, X_test` (pd.DataFrame): Train ve test özellikleri
- `y_train, y_test` (pd.Series): Train ve test hedef değişkenleri
- `t_train, t_test` (pd.Series): Train ve test treatment göstergeleri

**Örnek:**
```python
X_train, X_test, y_train, y_test, t_train, t_test = \
    train_test_split_uplift(
        df, 
        test_size=0.25, 
        random_state=42,
        stratify_treatment=True  # MUTLAKA True olmalı!
    )

# Treatment dengesi kontrolü
print(f"Train treatment ratio: {t_train.mean():.2%}")
print(f"Test treatment ratio: {t_test.mean():.2%}")
```

**Neden Stratify?**
```python
# ❌ YANLIŞ: Stratify kullanmazsan
X_train, X_test, y_train, y_test, t_train, t_test = \
    train_test_split_uplift(df, stratify_treatment=False)
# Train: 45% treatment, Test: 55% treatment → Dengesiz!

# ✅ DOĞRU: Stratify kullan
X_train, X_test, y_train, y_test, t_train, t_test = \
    train_test_split_uplift(df, stratify_treatment=True)
# Train: 50% treatment, Test: 50% treatment → Dengeli!
```

---

### `check_treatment_balance(treatment, data_name)`

Treatment grubunun dengesini kontrol eder.

**Parametreler:**
- `treatment` (array-like): Treatment göstergesi (0 veya 1)
- `data_name` (str): Veri setinin adı (raporlama için)

**Döndürür:**
- `dict`: Treatment istatistikleri
  - `total`: Toplam örnek sayısı
  - `treatment`: Treatment grubu sayısı
  - `control`: Control grubu sayısı
  - `treatment_ratio`: Treatment oranı
  - `control_ratio`: Control oranı

**Örnek:**
```python
stats = check_treatment_balance(t_train, "Training Set")

# Çıktı:
# ============================================================
# 🎯 TREATMENT BALANCE: Training Set
# ============================================================
# Total samples:         7,500
# Treatment group:       3,750 (50.00%)
# Control group:         3,750 (50.00%)
# 
# ✅ Treatment groups are balanced
# ============================================================

print(f"Treatment ratio: {stats['treatment_ratio']:.2%}")
```

**İdeal Dengeler:**
- ✅ **30-70%**: İyi denge
- ⚠️ **20-80%**: Kabul edilebilir, dikkatli ol
- ❌ **<20% veya >80%**: Kötü denge, model yanılabilir

---

### `calculate_baseline_metrics(y, treatment)`

Baseline metriklerini hesaplar (ATE, conversion rates).

**Parametreler:**
- `y` (array-like): Outcome (0 veya 1)
- `treatment` (array-like): Treatment göstergesi (0 veya 1)

**Döndürür:**
- `dict`: Baseline metrikleri
  - `conversion_rate_treatment`: Treatment grubu conversion rate
  - `conversion_rate_control`: Control grubu conversion rate
  - `ate`: Average Treatment Effect (mutlak)
  - `relative_uplift`: Göreceli uplift (%)

**Örnek:**
```python
metrics = calculate_baseline_metrics(y_train, t_train)

print(f"CR (Treatment): {metrics['conversion_rate_treatment']:.2%}")
print(f"CR (Control):   {metrics['conversion_rate_control']:.2%}")
print(f"ATE:            {metrics['ate']:.4f}")
print(f"Relative:       {metrics['relative_uplift']:+.2%}")

# Çıktı:
# CR (Treatment): 4.50%
# CR (Control):   3.80%
# ATE:            0.0070
# Relative:       +18.42%
```

**Formüller:**
```
ATE = E[Y|T=1] - E[Y|T=0]
Relative Uplift = ATE / E[Y|T=0]
```

**Yorumlama:**
- **ATE > 0**: Treatment pozitif etki yapıyor ✅
- **ATE = 0**: Treatment etkisiz ➖
- **ATE < 0**: Treatment negatif etki yapıyor ⚠️

---

### `create_toy_dataset(n_samples, n_features, treatment_effect_size, ...)`

Test ve öğrenme için toy dataset oluşturur.

**Parametreler:**
- `n_samples` (int): Örnek sayısı (varsayılan: 1000)
- `n_features` (int): Özellik sayısı (varsayılan: 5)
- `treatment_effect_size` (float): Ortalama tedavi etkisi (varsayılan: 0.1)
- `noise_level` (float): Gürültü seviyesi (varsayılan: 0.3)
- `random_state` (int): Random seed (varsayılan: 42)

**Döndürür:**
- `pd.DataFrame`: Simüle edilmiş veri

**Örnek:**
```python
# Basit toy dataset
df = create_toy_dataset(n_samples=1000)

# Güçlü uplift ile
df = create_toy_dataset(
    n_samples=5000,
    n_features=10,
    treatment_effect_size=0.2,  # %20 uplift
    noise_level=0.1              # Az gürültü
)

# Heterogeneous uplift (bazı müşteriler pozitif, bazıları negatif)
# → Gerçek uplift modeling senaryosu
```

**Ne Zaman Kullan?**
- ✅ Hızlı test için
- ✅ Algoritma öğrenirken
- ✅ Debugging yaparken
- ❌ Production için (gerçek veri kullan!)

---

## 🎯 Tipik Workflow

### Tam Pipeline Örneği

```python
import pandas as pd
from src.data import (
    load_criteo_sample,
    train_test_split_uplift,
    check_treatment_balance,
    calculate_baseline_metrics
)

# 1. VERİ YÜKLEME
print("📥 Veri yükleniyor...")
df = load_criteo_sample("data/criteo_sample.parquet")
print(f"✅ {len(df):,} satır yüklendi")

# 2. TRAIN/TEST SPLIT
print("\n📊 Train/test split yapılıyor...")
X_train, X_test, y_train, y_test, t_train, t_test = \
    train_test_split_uplift(
        df, 
        test_size=0.25, 
        random_state=42,
        stratify_treatment=True
    )

# 3. TREATMENT BALANCE KONTROLÜ
print("\n⚖️  Treatment dengesi kontrol ediliyor...")
check_treatment_balance(t_train, "Training Set")
check_treatment_balance(t_test, "Test Set")

# 4. BASELINE METRİKLER
print("\n📈 Baseline metrikleri hesaplanıyor...")
train_metrics = calculate_baseline_metrics(y_train, t_train)
test_metrics = calculate_baseline_metrics(y_test, t_test)

# 5. KAYDET
print("\n💾 Kaydediliyor...")
import pickle

data = {
    'X_train': X_train,
    'X_test': X_test,
    'y_train': y_train,
    'y_test': y_test,
    't_train': t_train,
    't_test': t_test,
    'train_metrics': train_metrics,
    'test_metrics': test_metrics
}

with open('data/processed_data.pkl', 'wb') as f:
    pickle.dump(data, f)

print("✅ Pipeline tamamlandı!")
```

---

## 🔍 Debugging İpuçları

### Problem: Veri bulunamıyor

```python
FileNotFoundError: data/criteo_sample.parquet not found
```

**Çözüm:**
```bash
# Önce veriyi hazırla
python scripts/prepare_data.py
```

### Problem: Treatment dengesiz

```python
⚠️  WARNING: Treatment groups are imbalanced!
```

**Çözüm:**
```python
# Stratify kullan
X_train, X_test, y_train, y_test, t_train, t_test = \
    train_test_split_uplift(df, stratify_treatment=True)  # ← Bunu ekle
```

### Problem: ATE negatif

```python
ATE: -0.0234 (-14.5%)
⚠️  Treatment has NEGATIVE effect
```

**Yorumlama:**
- Bu NORMAL olabilir! Bazı kampanyalar negatif etki yapar.
- Örn: Agresif indirimler, marka değerini düşürüp uzun vadede zararlı olabilir
- Model işe yarar: Negatif uplift'li müşterileri EXCLUDE edersin

### Problem: Bellek yetersiz

```python
MemoryError: Unable to allocate array
```

**Çözüm:**
```python
# Küçük sample kullan
df = load_criteo_sample("data/criteo_sample.parquet")

# Veya daha küçük sample oluştur
from scripts.prepare_data import download_criteo_dataset
download_criteo_dataset(sample_size=10_000)  # 10K satır
```

---

## 📊 Veri Kalitesi Kontrolleri

### 1. Eksik Değer Kontrolü

```python
print(df.isnull().sum())
# Hepsi 0 olmalı!
```

### 2. Treatment Dengesi

```python
from src.data import check_treatment_balance
check_treatment_balance(df['treatment'], "Full Dataset")

# İdeal: 30-70% arası
# Kabul edilebilir: 20-80% arası
# Kötü: <20% veya >80%
```

### 3. Covariate Balance (RCT kontrolü)

```python
from scipy import stats

for col in ['f0', 'f1', 'f2']:  # Her feature için
    x_t = df[df['treatment']==1][col]
    x_c = df[df['treatment']==0][col]
    
    t_stat, p_value = stats.ttest_ind(x_t, x_c)
    print(f"{col}: p-value = {p_value:.4f}")
    
    # p > 0.05 olmalı (dengeli)
```

### 4. Outcome Dağılımı

```python
print("Conversion rates:")
print(f"Overall:    {df['visit'].mean():.2%}")
print(f"Treatment:  {df[df['treatment']==1]['visit'].mean():.2%}")
print(f"Control:    {df[df['treatment']==0]['visit'].mean():.2%}")

# Çok düşük (<1%) veya çok yüksek (>50%) ise soru işareti
```

---

## 🎓 Öğrenme Kaynakları

### Teorik Arka Plan

1. **Randomized Controlled Trials (RCT)**
   - Treatment rastgele atanmalı
   - Gruplar dengeli olmalı
   - Yoksa causal inference yapamayız!

2. **Average Treatment Effect (ATE)**
   ```
   ATE = E[Y(1)] - E[Y(0)]
       = E[Y|T=1] - E[Y|T=0]  (RCT altında)
   ```

3. **Conditional ATE (CATE)**
   ```
   τ(x) = E[Y(1) - Y(0) | X=x]
   ```
   → Heterogeneous effects: Her müşteri farklı etki alır!

### Pratik İpuçları

✅ **YAP:**
- Treatment dengesini kontrol et
- Stratified split kullan
- Baseline metrikleri hesapla
- Train ve test setlerini ayrı ayrı analiz et

❌ **YAPMA:**
- Treatment dengesiz olduğunda devam etme
- Random split kullanma (stratify kullan!)
- Test setine bakmadan model seç
- Toy dataset'le production kodu test etme

---

## 📚 İleri Konular

### Propensity Score Weighting

```python
# Treatment probability'yi tahmin et
from sklearn.linear_model import LogisticRegression

ps_model = LogisticRegression()
ps_model.fit(X_train, t_train)
propensity = ps_model.predict_proba(X_train)[:, 1]

# IPW (Inverse Propensity Weighting)
weights = np.where(t_train == 1, 1/propensity, 1/(1-propensity))
```

### Doubly Robust Estimation

```python
# Hem outcome modeli hem propensity score kullan
# → Daha robust tahmin
# Gün 5+'te göreceğiz
```

---

## 🤝 Katkıda Bulunma

Bug buldunuz mu? İyileştirme öneriniz var mı?

1. Issue açın: `github.com/sumeyraguclu/uplift-learn/issues`
2. Test ekleyin: `tests/test_data.py`
3. Pull request gönderin!

---

## 📝 Changelog

### v0.1.0 (Gün 1)
- ✅ `load_criteo_sample()` eklendi
- ✅ `train_test_split_uplift()` eklendi
- ✅ `check_treatment_balance()` eklendi
- ✅ `calculate_baseline_metrics()` eklendi
- ✅ `create_toy_dataset()` eklendi
- ✅ Unit testler yazıldı
- ✅ Dokümantasyon tamamlandı

---

**Son Güncelleme**: Gün 1  
**Yazar**: Sümeyra Güçlü  
**Lisans**: MIT