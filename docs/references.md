# 📚 Kaynaklar ve Referanslar

> Bu dosya, proje boyunca kullanılan tüm kaynakların listesidir.

---

## 📖 1. Kitaplar

### Causal Inference

**⭐ Önerilen**:
1. **Causal Inference: The Mixtape** - Scott Cunningham
   - URL: https://mixtape.scunning.com/
   - Ücretsiz online
   - Seviye: Başlangıç-Orta
   - Not: En iyi nedensellik kaynağı, örneklerle dolu

2. **Causal Inference: What If** - Hernán & Robins
   - URL: https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/
   - Ücretsiz PDF
   - Seviye: Orta-İleri
   - Not: Daha teknik, epidemiyoloji odaklı

### Machine Learning

3. **The Elements of Statistical Learning** - Hastie, Tibshirani, Friedman
   - Bölüm 10: Boosting
   - Seviye: İleri
   - Not: XGBoost teorisi için

---

## 📄 2. Akademik Makaleler

### Uplift Modeling Temelleri

```bibtex
@article{radcliffe2007using,
  title={Using control groups to target on predicted lift: Building and assessing uplift model},
  author={Radcliffe, Nicholas J and Surry, Patrick D},
  journal={Direct Marketing Analytics Journal},
  pages={14--21},
  year={2007}
}
```
**Özet**: İlk uplift modeling makalesi, Qini metriğini tanıtıyor.  
**PDF**: [ResearchGate](https://www.researchgate.net/publication/242539235)

---

```bibtex
@inproceedings{gutierrez2017causal,
  title={Causal inference and uplift modelling: A review of the literature},
  author={Gutierrez, Pierre and G{\'e}rardy, Jean-Yves},
  booktitle={International Conference on Predictive Applications and APIs},
  pages={1--13},
  year={2017}
}
```
**Özet**: Uplift modeling'in kapsamlı literatür taraması.  
**PDF**: [PMLR](http://proceedings.mlr.press/v67/gutierrez17a.html)

---

### Meta-Learners

```bibtex
@article{kunzel2019metalearners,
  title={Metalearners for estimating heterogeneous treatment effects using machine learning},
  author={K{\"u}nzel, S{\"o}ren R and Sekhon, Jasjeet S and Bickel, Peter J and Yu, Bin},
  journal={Proceedings of the National Academy of Sciences},
  volume={116},
  number={10},
  pages={4156--4165},
  year={2019}
}
```
**Özet**: T-Learner, S-Learner, X-Learner karşılaştırması.  
**PDF**: [PNAS](https://www.pnas.org/doi/10.1073/pnas.1804597116)

---

### Causal Forests

```bibtex
@article{athey2019estimating,
  title={Estimating treatment effects with causal forests: An application},
  author={Athey, Susan and Wager, Stefan},
  journal={Observational Studies},
  volume={5},
  number={2},
  pages={37--51},
  year={2019}
}
```
**Özet**: Causal forests ile heterogeneous treatment effects.  
**PDF**: [arXiv](https://arxiv.org/abs/1902.07409)

---

## 🎥 3. Video Kaynaklar

### Başlangıç Seviyesi

1. **Causal Inference Crash Course** - Brady Neal
   - URL: https://www.youtube.com/playlist?list=PLoazKTcS0Rzb6bb9L508cyJ1z-U9iWkA0
   - Süre: 10 bölüm × 15-30 dk
   - Seviye: Başlangıç
   - Not: Potansiyel outcomes, DAG, backdoor criterion

2. **Uplift Modeling with Python** - PyData Talk
   - URL: https://www.youtube.com/watch?v=fkXIxRsRj3E
   - Süre: 30 dk
   - Seviye: Başlangıç-Orta
   - Not: Pratik örneklerle uplift

### İleri Seviye

3. **Causal ML at Uber** - Uber Engineering
   - URL: https://www.youtube.com/watch?v=4J0KiKXJEg0
   - Süre: 45 dk
   - Seviye: İleri
   - Not: Production uygulaması

---

## 🌐 4. Online Kaynaklar

### Dokümantasyon

1. **scikit-uplift Documentation**
   - URL: https://www.uplift-modeling.com/en/latest/
   - İçerik: API referansı, örnekler, tutorials
   - Not: Bizim referans kaynağımız

2. **XGBoost Documentation**
   - URL: https://xgboost.readthedocs.io/
   - İçerik: Parametre ayarları, örnekler
   - Not: Model eğitimi için gerekli

3. **scikit-learn User Guide**
   - URL: https://scikit-learn.org/stable/user_guide.html
   - Bölüm: Model Selection, Metrics
   - Not: Cross-validation, AUC hesaplama

### Blog Yazıları

4. **Uplift Modeling - Towards Data Science**
   - URL: https://towardsdatascience.com/a-quick-introduction-to-uplift-modeling-b10a78a3ec9c
   - Yazar: Robert Yi
   - Not: Görsel anlatım, Python örnekleri

5. **Causal Inference for The Brave and True**
   - URL: https://matheusfacure.github.io/python-causality-handbook/
   - Yazar: Matheus Facure
   - Not: Python ile causal inference, ücretsiz online kitap

---

## 📊 5. Veri Setleri

### Criteo Uplift Dataset

```bibtex
@misc{criteo2016uplift,
  title={Criteo Uplift Modeling Dataset},
  author={Criteo AI Lab},
  year={2016},
  howpublished={\url{https://ailab.criteo.com/criteo-uplift-prediction-dataset/}},
  note={13.9M observations, 12 features, treatment/control split}
}
```

**Detaylar**:
- Boyut: 13.9M satır
- Features: 12 (f0-f11, anonimleştirilmiş)
- Treatment: Reklam gösterildi mi?
- Outcome: Web sitesi ziyareti (visit)
- Format: CSV.GZ
- Lisans: Creative Commons

**İndirme**:
```bash
wget https://huggingface.co/datasets/criteo/criteo-uplift/resolve/main/criteo-research-uplift-v2.1.csv.gz
```

### Diğer Veri Setleri

1. **Hillstrom Email Marketing**
   - URL: https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html
   - Boyut: 64k satır
   - Not: E-mail kampanya verisi

2. **RetailHero (X5 Retail)**
   - URL: https://ods.ai/competitions/x5-retailhero-uplift-modeling
   - Boyut: 2M satır
   - Not: Perakende kampanya verisi

---

## 🛠️ 6. Araçlar ve Kütüphaneler

### Python Kütüphaneleri

```python
# Core
numpy==1.26.2
pandas==2.1.4
scipy==1.11.4

# ML
scikit-learn==1.3.2
xgboost==2.0.3

# Uplift
scikit-uplift==0.5.1

# Visualization
matplotlib==3.8.2
seaborn==0.13.0

# Optimization
ortools==9.8.3296
```

### Alternatif Kütüphaneler

1. **CausalML** (Uber)
   - URL: https://github.com/uber/causalml
   - İçerik: T/S/X-Learner, Causal Forests, TMLE
   - Not: Production-ready, Uber'in kütüphanesi

2. **EconML** (Microsoft)
   - URL: https://github.com/microsoft/EconML
   - İçerik: Double ML, DRLearner, Causal Forests
   - Not: Ekonometri odaklı, çok güçlü

3. **DoWhy** (Microsoft)
   - URL: https://github.com/py-why/dowhy
   - İçerik: Causal inference framework
   - Not: DAG-based, identification

---

## 📝 7. Tutorials ve Workshops

### Interactive Tutorials

1. **scikit-uplift Tutorials**
   - URL: https://www.uplift-modeling.com/en/latest/tutorials/
   - İçerik: RetailHero, Criteo örnekleri
   - Format: Jupyter Notebook

2. **CausalML Tutorials**
   - URL: https://causalml.readthedocs.io/en/latest/examples.html
   - İçerik: Meta-learner karşılaştırmaları
   - Format: Jupyter Notebook

### Coursera / edX

3. **A Crash Course in Causality** - University of Pennsylvania
   - Platform: Coursera
   - Süre: 5 hafta
   - Seviye: Başlangıç
   - Sertifika: Var (ücretli)

4. **Causal Diagrams** - Johns Hopkins
   - Platform: Coursera
   - Süre: 4 hafta
   - Seviye: Başlangıç
   - Not: DAG ve backdoor criterion

---

## 🏢 8. Endüstri Uygulamaları

### Case Studies

1. **Uber: Causal ML Platform**
   - Blog: https://eng.uber.com/causal-inference-at-uber/
   - İçerik: Production pipeline, A/B testing at scale
   - Yıl: 2019

2. **Booking.com: Uplift Modeling**
   - Konferans: KDD 2018
   - İçerik: Travel industry uplift modeling
   - PDF: [ACM Digital Library](https://dl.acm.org/doi/10.1145/3219819.3219959)

3. **Wayfair: Heterogeneous Treatment Effects**
   - Blog: https://tech.wayfair.com/
   - İçerik: E-commerce personalization
   - Yıl: 2020

---

## 📊 9. Konferans Sunumları

### KDD (Knowledge Discovery and Data Mining)

1. **"Large Scale Uplift Modeling"** - KDD 2015
   - Sunan: Pierre Gutierrez (Criteo)
   - Slides: [SlideShare](https://www.slideshare.net/)

2. **"Causal Inference and Uplift"** - KDD 2018
   - Tutorial
   - Speakers: S. Athey, G. Imbens

### PyData

3. **"Uplift Modeling with scikit-uplift"** - PyData 2019
   - Video: YouTube
   - Speaker: Maksim Shevchenko

---

## 🎓 10. Akademik Dersler

### Stanford

1. **STATS 361: Causal Inference**
   - Instructor: Stefan Wager
   - Materials: http://web.stanford.edu/~swager/stats361.html
   - Not: Lecture notes ve assignments

### MIT

2. **6.S897: Machine Learning for Healthcare**
   - Bölüm: Causal Inference
   - Materials: MIT OpenCourseWare

---

## 🔧 11. GitHub Repositories

### Öğrenme Kaynakları

1. **awesome-causality**
   - URL: https://github.com/rguo12/awesome-causality-algorithms
   - İçerik: Curated list of causal inference resources

2. **CausalInference**
   - URL: https://github.com/laurencium/causalinference
   - İçerik: Python implementations of causal methods

### Örnek Projeler

3. **uplift-modeling-examples**
   - URL: https://github.com/Minyus/uplift_modeling_examples
   - İçerik: Multiple datasets, different methods

---

## 📖 12. Glossary (Terimler Sözlüğü)

| Terim | İngilizce | Açıklama |
|-------|-----------|----------|
| Uplift | Uplift | Tedavi etkisi (p1 - p0) |
| ATE | Average Treatment Effect | Ortalama tedavi etkisi |
| CATE | Conditional ATE | Koşullu ortalama etki |
| ITE | Individual Treatment Effect | Bireysel tedavi etkisi |
| Qini | Qini Coefficient | Kümülatif kazanç metriği |
| AUUC | Area Under Uplift Curve | Uplift eğrisi altındaki alan |
| IPW | Inverse Propensity Weighting | Ters eğilim ağırlıklandırma |
| TMLE | Targeted Maximum Likelihood | Hedeflenmiş maksimum olabilirlik |
| DAG | Directed Acyclic Graph | Yönlü döngüsüz grafik |
| RCT | Randomized Controlled Trial | Rastgele kontrollü deney |
| SUTVA | Stable Unit Treatment Value | Kararlı birim tedavi değeri |

---

## 📅 13. Güncelleme Geçmişi

| Tarih | Eklenen Kaynaklar | Notlar |
|-------|-------------------|--------|
| Gün 1 | Temel kaynaklar eklendi | İlk versiyon |
| Gün 2 | T-Learner makaleleri | Meta-learners |
| Gün 3 | Metrik makaleleri | Qini, AUUC |
| ... | ... | ... |

---

## 🎯 14. Önerilen Okuma Sırası

### Hafta 1: Temel Kavramlar
1. ✅ Causal Inference Mixtape - Bölüm 1-2
2. ✅ Radcliffe & Surry (2007) - Uplift temel makale
3. ✅ scikit-uplift docs - Quickstart

### Hafta 2: Methodlar
1. Künzel et al. (2019) - Meta-learners
2. Gutierrez & Gérardy (2017) - Literature review
3. CausalML tutorials

### Hafta 3: İleri Konular
1. Athey & Wager (2019) - Causal Forests
2. Propensity score papers
3. Production case studies (Uber, Booking.com)

---

## 📧 15. İletişim ve Topluluk

### Forums

1. **Cross Validated (StackExchange)**
   - Tag: [causal-inference]
   - URL: https://stats.stackexchange.com/

2. **r/CausalInference (Reddit)**
   - URL: https://www.reddit.com/r/CausalInference/

### Slack/Discord

3. **Causal Inference Community**
   - Platform: Slack
   - Join: [Link]

---

## 📝 16. Citation Template

Eğer bu projeyi bir makalede kullanacaksan:

```bibtex
@misc{upliftlearn2025,
  title={Uplift-Learn: A Hands-on Learning Project for Uplift Modeling},
  author={[Your Name]},
  year={2025},
  howpublished={\url{https://github.com/[username]/uplift-learn}},
  note={Educational implementation of T-Learner and optimization methods}
}
```

---

**Son Güncelleme**: Gün 1  
**Toplam Kaynak**: 30+  
**Kategoriler**: 16

---

## ✅ Kullanım Notu

Bu dosyayı şu şekilde kullan:
1. Her gün yeni öğrendiğin kaynak ekle
2. Okuduğun makaleleri işaretle
3. Raporunda bu kaynakları cite et
4. "Hangi kaynak ne zaman kullanıldı" notunu tut

**Örnek**:
```markdown
## Gün 2'de Kullanılan Kaynaklar
- [x] Künzel et al. (2019) - T-Learner section
- [x] scikit-uplift source code - TwoModels class
- [ ] Causal Inference Mixtape - Chapter 5
```