# 📚 Öğrenme Günlüğü

> Bu dosya, her gün öğrendiğim kavramları ve karşılaştığım sorunları takip etmek için.

---

## 🗓️ Gün 1: Veri Keşfi (Tarih: ______)

### ✅ Tamamlanan Görevler
- [x] Proje yapısı oluşturuldu
- [x] README.md yazıldı
- [x] docs/theory.md okundu
- [x] Criteo veri seti indirildi (10k örnek)
- [x] 01_data_exploration.ipynb tamamlandı
- [x] Github'a ilk commit

### 📊 Öğrendiğim Kavramlar

#### 1. Uplift Modeling Temelleri
- **Treatment**: Kupon/reklam gösterme (T=1) vs. göstermeme (T=0)
- **Outcome**: Satın alma, ziyaret, dönüşüm (Y=0 veya Y=1)
- **ATE (Average Treatment Effect)**: Ortalama tedavi etkisi
  ```
  ATE = Mean(Y|T=1) - Mean(Y|T=0)
  Benim verimde: ATE = _____%
  ```

#### 2. 4 Müşteri Tipi
- **Persuadables**: Kupon ile alır, kupon olmadan almaz → HEDEFİMİZ!
- **Sure Things**: Zaten alacak → Kupon gereksiz, maliyet boşa
- **Lost Causes**: Hiç almaz → Kupon gereksiz
- **Sleeping Dogs**: Kupon gösterince ALMAZ! → Negatif uplift

#### 3. Randomization (A/B Test)
- Rastgele atama → Treatment ve Control grupları dengeli olmalı
- **Covariate Balance**: Gruplar arasında özellik dağılımı benzer mi?
- T-test ile kontrol: p-value > 0.05 ise dengeli

### 📈 Veri Özeti
```
Toplam Satır: ______
Toplam Sütun: ______
Feature Sayısı: 12 (f0-f11)

Treatment Dağılımı:
- Control (T=0): _____% 
- Treatment (T=1): _____%

Dönüşüm Oranları:
- Genel: _____%
- Control: _____%
- Treatment: _____%
- ATE: +_____%

Covariate Balance: _____/6 özellik dengeli
```

### 🤔 Kafama Takılan Sorular

1. **p-value ne anlama geliyor?**
   - Soru: p-value = 0.03 ise ne demek?
   - Cevap: (Araştır)

2. **ATE küçük olabilir mi?**
   - Soru: ATE = 0.3% çok küçük, bu iyi mi?
   - Cevap: (Araştır)

3. **Covariate balance bozuksa ne olur?**
   - Soru: Gruplar dengesizse sonuçlara güvenilir mi?
   - Cevap: (Araştır)

4. **Feature'lar anonim, gerçek anlamları ne?**
   - Soru: f0, f1 ne anlama geliyor?
   - Cevap: Criteo gizlilik için anonimleştirmiş, tahmin edemeyiz

### 🐛 Karşılaştığım Sorunlar

1. **Veri indirme yavaş**
   - Sorun: 100MB veri 5 dakika sürdü
   - Çözüm: Parquet formatı kullanarak hızlandırdım

2. **Jupyter hataları**
   - Sorun: Kernel restart gerekti
   - Çözüm: (Notunu buraya yaz)

### 📚 Okuduğum/İzlediğim Kaynaklar
- [ ] docs/theory.md → T-Learner bölümü
- [ ] Video: Causal ML Crash Course (15 dk)
- [ ] scikit-uplift repo incelemesi

### 🎯 Yarın Hedeflerim (Gün 2)
- [ ] T-Learner sınıfını sıfırdan kodlamak
- [ ] XGBoost ile model eğitmek
- [ ] İlk uplift tahminlerini yapmak
- [ ] Uplift dağılımını görselleştirmek
- [ ] scikit-uplift ile karşılaştırmak

---

## 🗓️ Gün 2: T-Learner İmplementasyonu (Tarih: ______)

### ✅ Tamamlanan Görevler
- [ ] ...

### 📊 Öğrendiğim Kavramlar
- [ ] ...

### 🤔 Kafama Takılan Sorular
1. ...

### 🎯 Yarın Hedeflerim (Gün 3)
- [ ] ...

---

## 🗓️ Gün 3: Metrik Değerlendirme (Tarih: ______)

(Daha sonra doldurulacak)

---

## 🗓️ Gün 4: Optimizasyon (Tarih: ______)

(Daha sonra doldurulacak)

---

## 🗓️ Gün 5: Pipeline (Tarih: ______)

(Daha sonra doldurulacak)

---

## 📊 Genel İstatistikler

**Toplam Öğrenme Günü**: 1/7  
**Tamamlanan Notebook**: 1/5  
**GitHub Commit**: 1  
**Kod Satırı**: ~0 (henüz sadece veri analizi)

---

## 🎓 Önemli Notlar

### Formüller
```
Uplift = P(Y=1|T=1,X) - P(Y=1|T=0,X)

ATE = E[Y|T=1] - E[Y|T=0]

Qini(k) = (sonra eklenecek)
```

### Python Snippet'leri
```python
# Uplift hesaplama (T-Learner)
p1 = model_treatment.predict_proba(X)[:, 1]
p0 = model_control.predict_proba(X)[:, 1]
uplift = p1 - p0
```

---

**Son Güncelleme**: ______