# 🔍 MODEL KARŞILAŞTIRMA RAPORU - FİNAL ANALİZ

## ⚠️ ÖNEMLİ: İKİ FARKLI SONUÇ DOSYASI BULUNDU

Projede **iki farklı model karşılaştırma dosyası** var ve sonuçlar **tamamen çelişiyor**:

---

## 📊 DOSYA 1: `exports/model_comparison.csv`

### Sonuçlar:
```
Model        Qini AUC    Uplift@10    Uplift@20    Rank
--------     --------    ----------   ----------   ----
S-Learner    0.083012    0.071000     0.057073     1️⃣ 🏆
X-Learner    0.079102    0.065210     0.053646     2️⃣
R-Learner    0.077158    0.073533     0.049361     3️⃣
T-Learner    0.068943    0.051349     0.044065     4️⃣
```

**En İyi:** S-Learner (Qini AUC: 0.083012)

---

## 📊 DOSYA 2: `results/model_comparison.csv`

### Sonuçlar:
```
Model        Qini AUC    Uplift@10    Uplift@20    Rank
--------     --------    ----------   ----------   ----
T-Learner    0.072672    26.64%       20.15%       1️⃣ 🏆
X-Learner    0.044292    15.25%       11.17%       2️⃣
S-Learner    0.033258    11.31%       8.52%        3️⃣
```

**En İyi:** T-Learner (Qini AUC: 0.072672)

⚠️ **Not:** R-Learner bu dosyada yok!

---

## 🔍 FARK ANALİZİ

### Qini AUC Karşılaştırması:

| Model      | Exports Dosyası | Results Dosyası | Fark        | % Değişim |
|------------|----------------|-----------------|-------------|-----------|
| **T-Learner** | 0.068943     | **0.072672**     | **+0.003729** | **+5.4%** ✅ |
| **X-Learner** | 0.079102     | **0.044292**     | -0.034810   | -44.0% ❌ |
| **S-Learner** | 0.083012     | **0.033258**     | -0.049754   | -59.9% ❌ |

### Uplift@10 Karşılaştırması:

| Model      | Exports (decimal) | Results (percentage) | Notlar |
|------------|------------------|---------------------|--------|
| T-Learner  | 0.051349 (5.13%) | 26.64%              | ⚠️ Çok farklı! |
| X-Learner  | 0.065210 (6.52%) | 15.25%              | ⚠️ Farklı! |
| S-Learner  | 0.071000 (7.10%) | 11.31%              | ⚠️ Farklı! |

---

## 🤔 OLASI NEDENLER

1. **Farklı Veri Setleri:**
   - Exports dosyası: Farklı bir veri subset'i üzerinde değerlendirilmiş olabilir
   - Results dosyası: Tam veri seti (200,039 samples) kullanılmış

2. **Farklı Metrik Hesaplama:**
   - Uplift@10 değerleri tamamen farklı (26.64% vs 5.13%)
   - Bu, farklı bir hesaplama metodolojisi olduğunu gösteriyor

3. **Farklı Değerlendirme Zamanları:**
   - Exports dosyası: Eski bir değerlendirme (R-Learner dahil)
   - Results dosyası: Yeni bir değerlendirme (sadece T, S, X modelleri)

---

## ✅ HANGİSİ DOĞRU?

### `results/model_comparison.csv` daha güncel görünüyor çünkü:

1. ✅ **Daha fazla sample:** 200,039 samples
2. ✅ **Güncel script:** `compare_models.py` tarafından oluşturuluyor
3. ✅ **Standardize metodoloji:** `src.metrics.evaluate_uplift_model` kullanıyor
4. ✅ **Tüm modeller aynı veri üzerinde:** Hepsi 200,039 sample

### Ancak bazı endişeler var:

⚠️ **Uplift@10 değerleri çok yüksek:**
   - T-Learner: 26.64% → Bu gerçekçi mi?
   - Bu değerler ya yanlış hesaplanmış ya da farklı bir metodoloji kullanılmış olabilir

⚠️ **S-Learner'ın performansı çok düşük:**
   - Exports'ta: 0.083 (en iyi)
   - Results'ta: 0.033 (en kötü)
   - %60 düşüş çok fazla!

---

## 🎯 SONUÇ VE TAVSİYE

### Mevcut Durum:

**`results/model_comparison.csv`'ye göre:**
- ✅ **T-Learner en iyi model** (Qini AUC: 0.072672)
- ✅ Uplift@10'da çok yüksek performans (26.64%)
- ⚠️ Ancak bu sonuçlar şüpheli - çok yüksek değerler

### Önerilen Aksiyon:

1. **Yeniden değerlendirme yap:**
   ```bash
   python scripts/compare_models.py
   ```

2. **Manuel kontrol:**
   - Prediction dosyalarını kontrol et
   - Metrik hesaplama fonksiyonlarını doğrula
   - Gerçek veri üzerinde cross-check yap

3. **Hangisini kullanmalı:**
   - Eğer **`results/model_comparison.csv`** güncel ve doğru hesaplanmışsa → **T-Learner en iyi**
   - Eğer **`exports/model_comparison.csv`** daha güvenilirse → **S-Learner en iyi**

### Final Karar için Gereken:

1. ✅ Her iki dosyanın nasıl oluşturulduğunu anla
2. ✅ Prediction dosyalarının kalitesini kontrol et
3. ✅ Metrik hesaplama fonksiyonlarını doğrula
4. ✅ Gerçek veri üzerinde manuel test yap

---

## 📝 NOTLAR

- **Uplift@10 değerleri:** Results dosyasındaki değerler çok yüksek (26.64%), bu muhtemelen percentage formatında ve belki de farklı bir hesaplama yöntemi kullanılmış
- **R-Learner:** Results dosyasında yok, sadece T, S, X modelleri var
- **Sample sayısı:** Her iki dosyada da aynı (200,039), bu iyi bir işaret

---

**Son Güncelleme:** Analiz tarihi  
**Durum:** ⚠️ Çelişkili sonuçlar - doğrulama gerekiyor

