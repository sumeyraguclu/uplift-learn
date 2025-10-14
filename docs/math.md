# 🧮 Matematiksel Detaylar

> Bu dosya, projede kullanılan tüm formüllerin detaylı açıklamasını içerir.

---

## 📐 1. Temel Notasyon

| Sembol | Açıklama | Örnek Değer |
|--------|----------|-------------|
| `Y` | Outcome (sonuç değişkeni) | 0 veya 1 |
| `T` | Treatment (tedavi göstergesi) | 0 veya 1 |
| `X` | Features (özellikler) | [f0, f1, ..., f11] |
| `τ(x)` | Uplift (bireysel tedavi etkisi) | -0.1 ile +0.5 arası |
| `n` | Toplam örnek sayısı | 10,000 |
| `n_t` | Treatment grubundaki sayı | 8,470 |
| `n_c` | Control grubundaki sayı | 1,530 |

---

## 🎯 2. Uplift Tanımı

### 2.1 Potansiyel Sonuçlar Çerçevesi (Rubin Causal Model)

Her birey için **iki potansiyel sonuç** vardır:

```
Y(1): Treatment alırsa olan
Y(0): Treatment almazsa olan
```

**Bireysel Tedavi Etkisi (ITE)**:
```
τᵢ = Y(1)ᵢ - Y(0)ᵢ
```

**Problem**: Aynı anda hem Y(1) hem Y(0) gözlemleyemeyiz!

### 2.2 Conditional Average Treatment Effect (CATE)

```
τ(x) = E[Y(1) - Y(0) | X = x]
     = E[Y(1) | X = x] - E[Y(0) | X = x]
```

**Randomization altında**:
```
τ(x) = E[Y | T=1, X=x] - E[Y | T=0, X=x]
     = P(Y=1 | T=1, X=x) - P(Y=1 | T=0, X=x)
```

---

## 🤖 3. T-Learner

### 3.1 Algoritma

**Adım 1**: Veriyi ayır
```
D₁ = {(Xᵢ, Yᵢ) : Tᵢ = 1}  # Treatment grubu
D₀ = {(Xᵢ, Yᵢ) : Tᵢ = 0}  # Control grubu
```

**Adım 2**: İki model eğit
```
μ₁(x) = P(Y=1 | T=1, X=x)  ← Train on D₁
μ₀(x) = P(Y=1 | T=0, X=x)  ← Train on D₀
```

**Adım 3**: Uplift tahmin et
```
τ̂(x) = μ₁(x) - μ₀(x)
```

### 3.2 Matematiksel Gerekçe

**Beklenti**:
```
E[τ̂(x)] = E[μ₁(x) - μ₀(x)]
        = E[Y | T=1, X=x] - E[Y | T=0, X=x]
        = τ(x)  ✓ (Unbiased)
```

**Varyans**:
```
Var[τ̂(x)] = Var[μ₁(x)] + Var[μ₀(x)]
```
→ İki modelin hatası toplanır (dezavantaj)

---

## 📊 4. Metrikler

### 4.1 Qini Coefficient

**Tanım**: Kümülatif kazanç eğrisi

**Adım 1**: Uplift'e göre sırala (azalan)
```
π = argsort(-τ̂)  # Permutasyon
```

**Adım 2**: Kümülatif kazançları hesapla
```
Qini(k) = Σᵢ₌₁ᵏ [Yπ(i) × Tπ(i) / nₜ - Yπ(i) × (1-Tπ(i)) / nᴄ]
```

**Alternatif Formül** (daha verimli):
```
Qini(k) = (Yₜ(k) / nₜ - Yᴄ(k) / nᴄ) × k

where:
- Yₜ(k) = Σᵢ₌₁ᵏ Yπ(i) × 1[Tπ(i)=1]  # Treatment grubundaki başarılar
- Yᴄ(k) = Σᵢ₌₁ᵏ Yπ(i) × 1[Tπ(i)=0]  # Control grubundaki başarılar
```

**Qini AUC**:
```
QINI_AUC = (1/n) × Σₖ₌₁ⁿ Qini(k)
```

### 4.2 Uplift@k

**Tanım**: İlk k% müşteride ortalama uplift

```
Uplift@k = (Yₜ(⌊k×n⌋) / nₜ(⌊k×n⌋)) - (Yᴄ(⌊k×n⌋) / nᴄ(⌊k×n⌋))

where:
- nₜ(k) = Σᵢ₌₁ᵏ 1[Tπ(i)=1]  # İlk k'da treatment sayısı
- nᴄ(k) = Σᵢ₌₁ᵏ 1[Tπ(i)=0]  # İlk k'da control sayısı
```

**Örnek**: Uplift@30
```
İlk %30'u hedeflediğimizde:
- Treatment grubunda: 250/1000 = 25% dönüşüm
- Control grubunda: 200/1000 = 20% dönüşüm
- Uplift@30 = 0.25 - 0.20 = 0.05 = +5%
```

---

## 💰 5. Optimizasyon

### 5.1 Kâr Fonksiyonu

**Bireysel Beklenen Kâr**:
```
Profit(i) = τ(xᵢ) × Margin - Cost

where:
- Margin: Müşteri başı marj (TL)
- Cost: Temas maliyeti (TL)
```

**Toplam Kâr**:
```
Total_Profit = Σᵢ∈S [τ(xᵢ) × Margin - Cost]

where S = Seçilen müşteri kümesi
```

### 5.2 Knapsack Problemi

**Formülasyon**:
```
maximize:   Σᵢ vᵢ × xᵢ
subject to: Σᵢ wᵢ × xᵢ ≤ B
            xᵢ ∈ {0, 1}

where:
- vᵢ = τ(xᵢ) × Margin - Cost  # Değer (kâr)
- wᵢ = Cost                    # Ağırlık (maliyet)
- B = Budget                   # Kapasite (bütçe)
- xᵢ = 1 if seçildi, 0 otherwise
```

**Greedy Çözüm** (sabit maliyette optimal):
```
1. Kârı sırala: v₁ ≥ v₂ ≥ ... ≥ vₙ
2. İlk k = ⌊B/Cost⌋ tanesini seç
```

**Oran-bazlı Greedy** (değişken maliyette):
```
1. ratio_i = vᵢ / wᵢ hesapla
2. Oranı sırala: ratio₁ ≥ ratio₂ ≥ ...
3. Bütçe bitene kadar ekle
```

---

## 📈 6. İstatistiksel Testler

### 6.1 T-Test (Covariate Balance)

**Hipotez**:
```
H₀: μₜ = μᴄ  (Gruplar arasında fark yok)
H₁: μₜ ≠ μᴄ  (Fark var)
```

**Test İstatistiği**:
```
t = (x̄ₜ - x̄ᴄ) / √(s²ₜ/nₜ + s²ᴄ/nᴄ)

where:
- x̄ₜ, x̄ᴄ: Grup ortalamaları
- s²ₜ, s²ᴄ: Grup varyansları
- nₜ, nᴄ: Grup büyüklükleri
```

**Karar**:
```
if p-value > 0.05:
    "Gruplar dengeli" ✓
else:
    "Gruplar dengesiz" ✗
```

### 6.2 ATE İstatistiksel Anlamlılık

**ATE**:
```
ATE = ȳₜ - ȳᴄ

where:
- ȳₜ = (1/nₜ) Σᵢ:Tᵢ=1 Yᵢ
- ȳᴄ = (1/nᴄ) Σᵢ:Tᵢ=0 Yᵢ
```

**Standart Hata**:
```
SE(ATE) = √(s²ₜ/nₜ + s²ᴄ/nᴄ)
```

**%95 Güven Aralığı**:
```
CI = ATE ± 1.96 × SE(ATE)
```

---

## 🎓 7. İleri Konular (Gün 5+)

### 7.1 Propensity Score

```
e(x) = P(T=1 | X=x)
```

**IPW (Inverse Propensity Weighting)**:
```
τ̂ᴵᴾᵂ(x) = E[Y×T/e(x) - Y×(1-T)/(1-e(x)) | X=x]
```

### 7.2 Doubly Robust Estimator

```
τ̂ᴰᴿ(x) = μ₁(x) - μ₀(x) 
        + T/e(x) × [Y - μ₁(x)]
        - (1-T)/(1-e(x)) × [Y - μ₀(x)]
```

---

## 📚 Referanslar

1. **Rubin (1974)**: "Estimating causal effects of treatments"
2. **Künzel et al. (2019)**: "Metalearners for estimating heterogeneous treatment effects"
3. **Radcliffe & Surry (2011)**: "Real-world uplift modelling with significance-based uplift trees"

---

**Son Güncelleme**: (Gün 1)