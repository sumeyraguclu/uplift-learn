# 🎯 Genel Bakış

**Uplift-Learn**, hedefli pazarlama kampanyaları için uplift modellerini oluşturmak, değerlendirmek ve dağıtmak için framework'tür. Gerçek dünya perakende pazarlama zorluklarından (X5 RetailHero veri seti) doğan bu proje, hangi müşterilerin pazarlama müdahalelerine olumlu yanıt vereceğini belirlemeyi gösterir.

### Uplift Modelleme Nedir?

Uplift modelleme kritik soruyu yanıtlar: *"Kampanyamızda kimi hedeflemeliyiz?"*

Dönüşüm olasılığını tahmin eden geleneksel tahmine dayalı modellerden farklı olarak, uplift modelleri müdahalenin **artımlı etkisini** tahmin eder:

```
CATE (Koşullu Ortalama Tedavi Etkisi) = P(dönüşüm|müdahale) - P(dönüşüm|kontrol)
```

**Ana fikir**: Zaten dönüşüm yapacak müşterilere (veya daha kötüsü, kampanyanızdan olumsuz etkilenecek müşterilere) bütçe harcamayın.

---

## ✨ Özellikler

### 🧠 Temel Modelleme
- **Çoklu Uplift Modelleri**: T-Learner, S-Learner, X-Learner, R-Learner implementasyonları
- **Model Karşılaştırma**: Kapsamlı model değerlendirme ve seçim sistemi
- **Özellik Mühendisliği**: RFM segmentasyonu, davranışsal özellikler ve otomatik ön işleme
- **Model Kalibrasyonu**: Güvenilir olasılık tahminleri için izotonik regresyon
- **Değerlendirme Metrikleri**: Qini eğrileri, Uplift@k, güven aralıklı ATE, tedavi dengesi kontrolleri

### 🎨 Kampanya Optimizasyonu
- **Çoklu Stratejiler**: Açgözlü optimizasyon, ROI eşikleri, top-k seçimi
- **Bütçe Kısıtlamaları**: Bütçe limitleri içinde ROI'yi maksimize edin
- **A/B Test Tasarımı**: İstatistiksel güç hesaplamaları, tedavi ataması, Meta Ads entegrasyonu
- **ROI Projeksiyonları**: Beklenen gelir, kâr ve maliyet analizi

### 📊 Görselleştirme & Analiz
- Qini kümülatif kazanç eğrileri
- CATE dağılım analizi
- Uplift@k performans grafikleri
- Kalibrasyon diagnostikleri
- Segment düzeyinde içgörüler

### 🏭
- **Merkezi Konfigürasyon**: YAML tabanlı yapılandırma yönetimi
- **Modüler Mimari**: Endişelerin temiz ayrımı (veri, model, metrikler, optimizasyon)
- **Kapsamlı Loglama**: Detaylı yürütme logları ve metrik takibi
- **Model Kalıcılığı**: Eğitilmiş modelleri ve kalibratörleri kaydet/yükle
- **Meta Ads Entegrasyonu**: Facebook/Instagram kampanyaları için dışa aktarıma hazır kitle listeleri

---

## 📦 Kurulum

### Ön Gereksinimler
- Python 3.11+
- pip veya conda

### Kaynaktan Kurulum

```bash
# Repository'yi klonlayın
git clone https://github.com/sumeyraguclu/uplift-learn.git
cd uplift-learn

# Virtual environment oluşturun (önerilir)
python -m venv venv
source venv/bin/activate  # Windows'ta: venv\Scripts\activate

# Bağımlılıkları yükleyin
pip install -r requirements.txt

# Paketi geliştirme modunda yükleyin
pip install -e .
```

### Bağımlılıklar
```
numpy==1.26.2
pandas==2.1.4
scipy==1.11.4
scikit-learn==1.3.2
xgboost==2.0.3
scikit-uplift==0.5.1
matplotlib==3.8.2
seaborn==0.13.0
plotly==5.18.0
jupyter==1.0.0
ipykernel==6.27.1
notebook==7.0.6
ipywidgets==8.1.1
tqdm==4.66.1
python-dotenv==1.0.0
```

---

## 🚀 Hızlı Başlangıç

### 1. T-Learner Modelini Eğitin

```python
from src.model import TLearner
from src.data import load_criteo_sample
import pickle

# Ön işlenmiş veriyi yükleyin
with open('data/x5_rfm_processed.pkl', 'rb') as f:
    data = pickle.load(f)

df = data['data']
X = df[feature_cols]
y = df['target']
treatment = df['treatment']

# Modeli eğitin
model = TLearner(random_state=42)
metrics = model.fit(X, y, treatment, test_size=0.2, verbose=True)

# CATE tahmin edin
predictions = model.predict_cate(X)

print(f"Ortalama CATE: {predictions['cate'].mean():.4f}")
print(f"Model AUC (Kontrol): {metrics['auc_0']:.4f}")
print(f"Model AUC (Müdahale): {metrics['auc_1']:.4f}")

# Modeli kaydedin
model.save('models/tlearner_model.pkl')
```

### 2. Model Performansını Değerlendirin

```python
from src.metrics import evaluate_uplift_model

# Kapsamlı değerlendirme
metrics = evaluate_uplift_model(
    y_true=y_test,
    uplift=cate_pred,
    treatment=treatment_test,
    X=X_test,
    k_list=[0.1, 0.2, 0.3, 0.5]
)

print(f"Qini AUC: {metrics['qini_auc']:.4f}")
print(f"ATE: {metrics['ate']['ate']*100:.2f}%")
print(f"Tedavi Dengesi: {metrics['balance']['status']}")

# Farklı yüzdeliklerde uplift
for k, v in metrics['uplift_at_k'].items():
    print(f"{k}: {v:+.2f}%")
```

### 3. Tahminleri Kalibre Edin

```python
from src.calibration import calibrate_cate

# Ham CATE tahminlerini kalibre edin
calibrated_df, calibrator = calibrate_cate(
    predictions_df=pred_df,
    outcomes_df=df,
    save_calibrator=True,
    calibrator_path='models/calibrator.pkl',
    verbose=True
)

print(f"Ham CATE ortalaması: {calibrated_df['cate'].mean():.4f}")
print(f"Kalibre CATE ortalaması: {calibrated_df['cate_calibrated'].mean():.4f}")
```

### 4. Kampanyayı Optimize Edin

```python
from src.optimize import greedy_optimizer, compare_strategies

# Kampanya parametreleri
campaign:
  margin: 50.0              # Dönüşüm başına gelir ($)
  contact_cost: 0.50        # İletişim kurulan müşteri başına maliyet ($)
  budget: 10000.0           # Toplam kampanya bütçesi ($)
  min_roi: 0.0              # Minimum kabul edilebilir ROI
  top_k_default: 0.30       # Hedeflenecek varsayılan üst %

# Açgözlü optimizasyon (bütçe içinde kârı maksimize et)
result = greedy_optimizer(
    uplift=calibrated_df['cate_calibrated'].values,
    margin=MARGIN,
    contact_cost=CONTACT_COST,
    budget=BUDGET
)

print(f"Seçilen müşteriler: {result['n_selected']:,}")
print(f"Beklenen kâr: ${result['expected_profit']:,.2f}")
print(f"ROI: {result['roi_pct']:.1f}%")

# Birden fazla stratejiyi karşılaştırın
comparison = compare_strategies(
    uplift=calibrated_df['cate_calibrated'].values,
    margin=MARGIN,
    contact_cost=CONTACT_COST,
    budget=BUDGET,
    k_values=[0.1, 0.2, 0.3],
    roi_thresholds=[0.0, 0.5, 1.0]
)

print(comparison)
```

---

## 📖 Dokümantasyon

### Proje Yapısı

```
uplift-learn/
│
├── src/                          # Ana paket
│   ├── __init__.py
│   ├── config.py                 # Konfigürasyon yönetimi
│   ├── data.py                   # Veri yükleme yardımcıları
│   ├── model.py                  # T-Learner uygulaması
│   ├── metrics.py                # Uplift metrikleri (Qini, ATE, vb.)
│   ├── calibration.py            # CATE kalibrasyonu
│   └── optimize.py               # Kampanya optimizasyonu
│
├── scripts/                      # Uçtan uca pipeline scriptleri
│   ├── 1_check_x5_compatibility.py    # Veri uyumluluk kontrolü
│   ├── 2_explore_x5_detailed.py       # Veri keşfi
│   ├── 3_process_x5_rfm.py            # RFM segmentasyonu
│   ├── 5_train_tlearner.py            # T-Learner eğitimi
│   ├── 6_train_slearner.py            # S-Learner eğitimi
│   ├── 7_train_xlearner.py            # X-Learner eğitimi
│   ├── 8_train_rlearner.py            # R-Learner eğitimi
│   ├── 9_evaluate_uplift_metrics.py   # Model değerlendirmesi
│   ├── 10_campaign_planning.py        # Kampanya optimizasyonu
│   ├── 11_ab_test_meta.py             # Meta Ads A/B test
│   ├── 12_prepare_cate.py             # CATE kalibrasyonu
│   ├── 13_optimization_engine_meta.py # Gelişmiş optimizasyon
│   ├── 14_comprehensive_model_test.py # Kapsamlı model testi
│   └── compare_models.py              # Model karşılaştırması
│
├── examples/                     # Kullanım örnekleri
│   ├── t_learner_usage.py
│   ├── metrics_usage.py
│   ├── optimize_usage.py
│   ├── calibration_usage.py
│   └── config_usage.py
│
├── docs/                         # Dokümantasyon
│   ├── theory.md                 # Teorik arka plan
│   ├── math.md                   # Matematiksel formüller
│   ├── references.md             # Kaynaklar
│   └── MIGRATION_USAGE.md        # Migration kılavuzu
│
├── data/                         # Veri dizini
├── models/                       # Kaydedilen modeller
├── results/                      # Çıktı dosyaları
├── plots/                        # Görselleştirmeler
├── exports/                      # Harici çıktılar
├── logs/                         # Çalıştırma logları
│
├── config.yaml                   # Konfigürasyon dosyası
├── requirements.txt              # Python bağımlılıkları
├── setup.py                      # Paket kurulumu
└── README.md                     # Bu dosya
```

### Ana Modüller

#### `src.model.TLearner`
Production-grade T-Learner implementasyonu:
- Müdahale ve kontrol grupları için ayrı modeller eğitir
- CATE = P(Y|T=1,X) - P(Y|T=0,X) tahmin eder
- XGBoost tabanlı, özelleştirilebilir estimator'lar
- Katmanlandırmalı yerleşik train/test ayrımı
- Model kalıcılığı ve yükleme

**Not:** S-Learner, X-Learner ve R-Learner modelleri `scripts/` altında implementasyonu bulunmaktadır ve `src/` modüllerini kullanmaktadır.

#### `src.metrics`
Kapsamlı uplift değerlendirme metrikleri:
- **Qini AUC**: Qini eğrisi altında kalan alan (sıralama metriği)
- **Uplift@k**: Müşterilerin ilk k%'sindeki uplift
- **ATE**: Güven aralıklı ortalama tedavi etkisi
- **Tedavi Dengesi**: Kovaryat denge kontrolleri (SMD)

#### `src.calibration.CATECalibrator`
Ham model tahminlerini kalibre eder:
- Olasılık kalibrasyonu için izotonik regresyon
- Müdahale ve kontrol için ayrı kalibrasyon
- Metrik takibi (MAE iyileştirmesi)
- Görselleştirme araçları

#### `src.optimize`
Kampanya optimizasyon stratejileri:
- **Açgözlü Optimizasyon**: Bütçe içinde kârı maksimize et
- **ROI Eşiği**: ROI eşiğinin üzerindeki müşterileri seç
- **Top-k**: Uplift'e göre ilk k%'yi hedefle
- **Kısıtlı Optimizasyon**: Birden fazla kısıt (bütçe, ROI, maks müşteri)
- **Strateji Karşılaştırması**: Birden fazla yaklaşımı değerlendir

#### `src.config`
Merkezi konfigürasyon:
- YAML tabanlı ayarlar
- Ortama özgü geçersiz kılmalar (dev/prod)
- Yol yönetimi
- Kampanya parametreleri

---

## 📊 Örnekler

### Uçtan Uca Pipeline

```bash
# 1. Veri hazırlama
python scripts/3_process_x5_rfm.py

# 2. Tüm modelleri eğitin
python scripts/5_train_tlearner.py   # T-Learner
python scripts/6_train_slearner.py   # S-Learner
python scripts/7_train_xlearner.py   # X-Learner
python scripts/8_train_rlearner.py   # R-Learner

# 3. Model karşılaştırması
python scripts/compare_models.py

# 4. Model değerlendirmesi
python scripts/9_evaluate_uplift_metrics.py

# 5. CATE kalibrasyonu
python scripts/12_prepare_cate.py

# 6. Kampanya planlaması
python scripts/10_campaign_planning.py

# 7. Meta Ads A/B testi
python scripts/11_ab_test_meta.py
```

### Özel Analiz

```python
from src.config import get_config
from src.model import TLearner
from src.metrics import qini_auc_score, uplift_at_k_multiple
from src.optimize import optimize_with_constraints

# Konfigürasyonu yükle
config = get_config()

# Özel parametrelerle model eğit
model = TLearner(random_state=42)
model.fit(X_train, y_train, treatment_train)

# Tahmin et ve değerlendir
predictions = model.predict_cate(X_test)
cate = predictions['cate']

qini = qini_auc_score(y_test, cate, treatment_test)
uplifts = uplift_at_k_multiple(y_test, cate, treatment_test, [0.1, 0.2, 0.3])

print(f"Qini AUC: {qini:.4f}")
for k, v in uplifts.items():
    print(f"{k}: {v:.2f}%")

# Özel kısıtlarla optimize et
result = optimize_with_constraints(
    uplift=cate,
    margin=config.campaign.margin,
    contact_cost=config.campaign.contact_cost,
    budget=15000,
    min_roi=0.5,
    max_customers=5000
)

print(f"Seçilen: {result['n_selected']:,} müşteri")
print(f"Beklenen ROI: {result['roi_pct']:.1f}%")
```

---

## 🧪 Test Etme

```bash
# Tüm testleri çalıştır
pytest tests/

# Coverage ile çalıştır
pytest --cov=src tests/

# Belirli test dosyasını çalıştır
pytest tests/test_model.py -v
```

---

## 📈 Performans Kıyaslamaları

X5 RetailHero veri setine dayalı (gerçek perakende verisi - 200,039 müşteri):

### Model Karşılaştırması

| Model | Qini AUC | Uplift@10% | Uplift@20% | Uplift@30% |
|-------|----------|------------|------------|------------|
| **T-Learner** | 0.0727 | 26.64% | 20.15% | 16.39% |
| **X-Learner** | 0.0443 | 15.25% | 11.17% | 9.66% |
| **S-Learner** | 0.0333 | 11.31% | 8.52% | 7.03% |

**En İyi Model:** T-Learner (Qini AUC: 0.0727)

### Ortalama Tedavi Etkisi (ATE)
- **ATE**: +3.32% | Güvenilir pozitif etki
- **Tedavi Dengesi**: Dengeli gruplar (randomized experimental design)

---

## 🔬 Metodoloji

### Uplift Modelleri

Projede 4 farklı uplift modeli implementasyonu bulunmaktadır:

#### 1. T-Learner (Two-Model Approach)
İki ayrı model eğitir:
1. **Model 0 (Kontrol)**: P(Y=1|X, T=0) tahmin eder
2. **Model 1 (Müdahale)**: P(Y=1|X, T=1) tahmin eder
3. **CATE**: τ(X) = μ₁(X) - μ₀(X)

**Avantajları:** Gruba özgü kalıplar, az yanlılık, yorumlanabilir

#### 2. S-Learner (Single-Model Approach)
Tek model ile treatment'ı feature olarak kullanır:
- Treatment'ı ek özellik olarak ekleyerek tahmin yapar
- **Avantajları:** Daha az model, hızlı eğitim

#### 3. X-Learner (Cross-Learner Approach)
İki aşamalı cross-fit yaklaşımı:
- İlk aşamada T-Learner benzeri modeller
- İkinci aşamada treatment effect modeli
- **Avantajları:** Heterojen efektleri iyi yakalar

#### 4. R-Learner (Robinson Transformation)
Residualization ile confounding kaldırır:
- Orthogonalization tekniği
- **Avantajları:** Confounding bias azaltma

### Kalibrasyon

Ham model tahminleri genellikle güvenilir olasılık tahminleri için kalibrasyona ihtiyaç duyar:

- **Yöntem**: İzotonik regresyon
- Müdahale ve kontrol grupları için **ayrı kalibrasyon**
- **Doğrulama**: Çapraz doğrulanmış MAE iyileştirmesi
- **Sonuç**: İş kararları için daha güvenilir CATE tahminleri

### Optimizasyon

Kampanya optimizasyonu iş değerini maksimize eder:

```
max Σᵢ (CATEᵢ × margin - iletişim_maliyeti) × xᵢ

kısıtlar:
- Σᵢ iletişim_maliyeti × xᵢ ≤ bütçe
- CATEᵢ × margin ≥ iletişim_maliyeti × (1 + min_roi)
- Σᵢ xᵢ ≤ maks_müşteri
```

xᵢ ∈ {0,1}, i müşterisinin hedeflenip hedeflenmediğini gösterir.

---

## 🎯 Kullanım Alanları

### 1. **E-posta Pazarlama Kampanyaları**
Promosyon e-postalarına olumlu yanıt verecek müşterileri hedefleyin, zaten satın alacak olanlardan veya mesajdan rahatsız olacaklardan kaçının.

### 2. **Dijital Reklamcılık (Meta, Google)**
Maksimum ROAS için optimize edilmiş müşteri listelerini reklam platformlarına yükleyin. Kullanıma hazır Meta Ads entegrasyonu içerir.

### 3. **Elde Tutma Kampanyaları**
Hedefli müdahalelerle (indirimler, kişiselleştirilmiş teklifler) elde tutulabilecek riskli müşterileri belirleyin.

### 4. **Yukarı Satış/Çapraz Satış**
Ek ürünleri satın alma olasılığı en yüksek müşterileri bulun.

### 5. **A/B Test Planlama**
Müdahale/kontrol ataması ve başarı metrikleriyle istatistiksel olarak güçlü deneyler tasarlayın.

---

## 🛠️ Konfigürasyon

Özelleştirmek için `config.yaml` dosyasını düzenleyin:


# Kampanya ekonomisi
campaign:
  margin: 50.0              # Dönüşüm başına gelir ($)
  contact_cost: 0.50        # İletişim kurulan müşteri başına maliyet ($)
  budget: 10000.0           # Toplam kampanya bütçesi ($)
  min_roi: 0.0              # Minimum kabul edilebilir ROI
  top_k_default: 0.30       # Hedeflenecek varsayılan üst %

# Model parametreleri
model:
  random_state: 42
  test_size: 0.20
  xgboost:
    max_depth: 5
    n_estimators: 100
    learning_rate: 0.1

# Değerlendirme metrikleri
metrics:
  qini_bins: 100
  uplift_k_values: [0.1, 0.2, 0.3, 0.5]
  confidence_level: 0.95

# Kalibrasyon
calibration:
  method: "isotonic"
  cv_folds: 5
  min_samples_leaf: 10




## 🙏 Teşekkürler

- **X5 Retail Group** - RetailHero veri seti için
- **scikit-uplift** - Metrik uygulamasına ilham verdiği için
- **Radcliffe (2007)** - Qini eğrisi metodolojisi için
- Açık kaynak topluluğu mükemmel araçlar için (scikit-learn, XGBoost, pandas)

---


### Versiyon 0.2.0 (Tamamlandı ✅)
- [x] S-Learner, X-Learner ve R-Learner uygulamaları
- [x] Model karşılaştırma sistemi
- [x] Kapsamlı metrik sistemi
- [x] Kalibrasyon modülü
