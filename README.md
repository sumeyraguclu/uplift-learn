# uplift-learn
"Learning uplift modeling from scratch"
# 🎯 Uplift Learn

**Sıfırdan Uplift Modeling Öğrenme Projesi**

Bu proje, uplift modeling'i temellerinden öğrenmek ve production-ready bir sistem geliştirmek için oluşturulmuştur.

---

## 🎓 Ne Öğreneceğiz?

1. **Causal Inference (Nedensellik)**: Korelasyon ≠ Nedensellik
2. **T-Learner**: En basit uplift modeli
3. **Uplift Metrikleri**: Qini curve, Uplift@k
4. **Optimizasyon**: Greedy ve OR-Tools
5. **Production Pipeline**: Veri → Model → Plan

---

## 📊 Problem Tanımı

**Senaryo**: Bir e-ticaret şirketi müşterilere indirim kuponu göndermek istiyor.

**Sorular**:
- ❓ Hangi müşterilere kupon göndermeliyiz?
- ❓ Kupon göndermek gerçekten satışı artırır mı?
- ❓ Zaten alacak müşterilere gereksiz kupon göndermiyor muyuz?

**Çözüm**: Uplift Modeling ile "kuponun net etkisini" ölçeriz.

---

## 🗂️ Proje Yapısı

```
uplift-learn/
├── notebooks/           # Jupyter notebook'lar (öğrenme)
│   ├── 01_data_exploration.ipynb
│   ├── 02_t_learner_basics.ipynb
│   ├── 03_uplift_metrics.ipynb
│   ├── 04_optimization_intro.ipynb
│   └── 05_full_pipeline.ipynb
│
├── src/                 # Python modülleri
│   ├── data.py         # Veri yükleme
│   ├── model.py        # T-Learner
│   ├── metrics.py      # Metrikler
│   └── optimize.py     # Optimizasyon
│
├── data/               # Veri dosyaları (gitignore)
│   └── criteo_sample.parquet
│
├── docs/               # Dokümantasyon
│   ├── theory.md       # Teorik açıklamalar
│   ├── math.md         # Matematik detayları
│   └── references.md   # Kaynaklar
│
└── requirements.txt    # Bağımlılıklar
```

---

## 🚀 Hızlı Başlangıç

### 1. Kurulum

```bash
# Repo'yu klonla
git clone https://github.com/KULLANICI_ADIN/uplift-learn.git
cd uplift-learn

# Virtual environment oluştur
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Bağımlılıkları yükle
pip install -r requirements.txt

# Jupyter başlat
jupyter notebook
```

### 2. İlk Notebook

`notebooks/01_data_exploration.ipynb` dosyasını aç ve çalıştır.

---

## 📚 Öğrenme Kaynakları

### Başlangıç
- [Causal Inference Mixtape](https://mixtape.scunning.com/) - Nedensellik temelleri
- [PyData Uplift Talk](https://www.youtube.com/watch?v=fkXIxRsRj3E) - 30 dk video

### İleri Seviye
- [scikit-uplift Docs](https://www.uplift-modeling.com/en/latest/)
- [Uber CausalML Paper](https://arxiv.org/abs/1910.12043)

### Akademik
- Radcliffe & Surry (2007) - Uplift modeling temelleri
- Gutierrez & Gérardy (2017) - Literature review

---

## 📈 İlerleme

- [x] **Gün 1**: Veri keşfi
- [ ] **Gün 2**: T-Learner implementasyonu
- [ ] **Gün 3**: Metrik hesaplama
- [ ] **Gün 4**: Optimizasyon
- [ ] **Gün 5**: Pipeline
- [ ] **Gün 6-7**: Production kod

---

## 🤝 Katkıda Bulunma

Bu bir öğrenme projesidir. Hatalar ve iyileştirmeler beklenir!

---

## 📄 Lisans

MIT License

---

## 📧 İletişim

Sorular için Issue açın veya Pull Request gönderin.
