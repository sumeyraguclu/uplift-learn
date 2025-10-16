"""
X5 RetailHero - scikit-uplift Uyumluluk Kontrolü
Veriyi tanı ve uplift modelling için uygun mu kontrol et
"""
from sklift.datasets import fetch_x5
import pandas as pd
import numpy as np

def analyze_x5_structure(dataset):
    """X5 veri yapısını detaylı incele"""
    print("="*80)
    print("🔍 X5 RETAILHERO VERİ YAPISI ANALİZİ")
    print("="*80)
    
    # 1. ANA TABLO YAPISI
    print("\n📦 1. ANA TABLOLAR:")
    print("-" * 60)
    for key, value in dataset.data.items():
        if isinstance(value, pd.DataFrame):
            print(f"\n   {key.upper()}:")
            print(f"      Satır: {len(value):,}")
            print(f"      Sütun: {len(value.columns)}")
            print(f"      Sütunlar: {list(value.columns)}")
            print(f"      Memory: {value.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    # 2. TREATMENT VE TARGET
    print("\n🎯 2. TREATMENT & TARGET:")
    print("-" * 60)
    
    if hasattr(dataset, 'treatment'):
        treatment = dataset.treatment
        print(f"\n   TREATMENT:")
        print(f"      Tip: {type(treatment)}")
        print(f"      Boyut: {len(treatment):,}")
        if isinstance(treatment, pd.Series):
            print(f"      Dağılım:\n{treatment.value_counts()}")
            print(f"      Treatment rate: {treatment.mean():.2%}")
        else:
            unique, counts = np.unique(treatment, return_counts=True)
            for val, count in zip(unique, counts):
                print(f"         {val}: {count:,} ({count/len(treatment):.2%})")
    
    if hasattr(dataset, 'target'):
        target = dataset.target
        print(f"\n   TARGET:")
        print(f"      Tip: {type(target)}")
        print(f"      Boyut: {len(target):,}")
        if isinstance(target, pd.Series):
            print(f"      Dağılım:\n{target.value_counts()}")
            print(f"      Response rate: {target.mean():.2%}")
        else:
            unique, counts = np.unique(target, return_counts=True)
            for val, count in zip(unique, counts):
                print(f"         {val}: {count:,} ({count/len(target):.2%})")

def check_sklift_requirements(dataset):
    """scikit-uplift için gerekli formatı kontrol et"""
    print("\n" + "="*80)
    print("✅ SCIKIT-UPLIFT UYUMLULUK KONTROLÜ")
    print("="*80)
    
    requirements = {
        'X (Features)': False,
        'y (Target)': False,
        'treatment': False,
        'Binary target': False,
        'Binary treatment': False,
        'Sufficient data': False
    }
    
    # 1. Features kontrolü
    if 'train' in dataset.data:
        train_df = dataset.data['train']
        print(f"\n📊 1. FEATURES (X):")
        print(f"   ✅ Train subset var: {len(train_df):,} satır")
        requirements['X (Features)'] = True
        
        # Sütunları göster
        print(f"   Sütunlar: {list(train_df.columns)}")
    
    # 2. Target kontrolü
    if hasattr(dataset, 'target'):
        target = dataset.target
        print(f"\n🎯 2. TARGET (y):")
        print(f"   ✅ Target var: {len(target):,} değer")
        requirements['y (Target)'] = True
        
        # Binary mi?
        unique_vals = np.unique(target)
        print(f"   Unique değerler: {unique_vals}")
        if len(unique_vals) == 2 and set(unique_vals) == {0, 1}:
            print(f"   ✅ Binary (0/1) format")
            requirements['Binary target'] = True
        else:
            print(f"   ⚠️  Binary DEĞİL!")
    
    # 3. Treatment kontrolü
    if hasattr(dataset, 'treatment'):
        treatment = dataset.treatment
        print(f"\n💊 3. TREATMENT:")
        print(f"   ✅ Treatment var: {len(treatment):,} değer")
        requirements['treatment'] = True
        
        # Binary mi?
        unique_vals = np.unique(treatment)
        print(f"   Unique değerler: {unique_vals}")
        if len(unique_vals) == 2 and set(unique_vals) == {0, 1}:
            print(f"   ✅ Binary (0/1) format")
            requirements['Binary treatment'] = True
        else:
            print(f"   ⚠️  Binary DEĞİL!")
    
    # 4. Yeterli veri var mı?
    if hasattr(dataset, 'target') and hasattr(dataset, 'treatment'):
        total = len(dataset.target)
        treatment_count = np.sum(dataset.treatment == 1)
        control_count = np.sum(dataset.treatment == 0)
        
        print(f"\n📈 4. VERİ YETERLİLİĞİ:")
        print(f"   Toplam: {total:,}")
        print(f"   Treatment: {treatment_count:,} ({treatment_count/total:.1%})")
        print(f"   Control: {control_count:,} ({control_count/total:.1%})")
        
        if total >= 1000 and treatment_count >= 500 and control_count >= 500:
            print(f"   ✅ Yeterli veri var!")
            requirements['Sufficient data'] = True
        else:
            print(f"   ⚠️  Veri miktarı az olabilir")
    
    # ÖZET
    print("\n" + "="*80)
    print("📋 UYUMLULUK ÖZET")
    print("="*80)
    
    for req, status in requirements.items():
        icon = "✅" if status else "❌"
        print(f"   {icon} {req}")
    
    all_ok = all(requirements.values())
    
    if all_ok:
        print("\n🎉 VERİ SETİ SCIKIT-UPLIFT İÇİN UYGUN!")
    else:
        print("\n⚠️  Bazı gereksinimler eksik")
    
    return requirements, all_ok

def compare_with_criteo():
    """X5'i Criteo ile karşılaştır"""
    print("\n" + "="*80)
    print("🔄 X5 vs CRITEO KARŞILAŞTIRMASI")
    print("="*80)
    
    print("""
    CRITEO UPLIFT DATASET:
    ----------------------
    • Satır: ~13M (full), tipik kullanım 100K-1M
    • Features: 12 (f0-f11, anonim)
    • Treatment: Binary (0/1) - reklam gösterildi mi?
    • Target: Binary (0/1) - web sitesini ziyaret etti mi?
    • Use case: Display advertising
    • Format: X, y, treatment ayrı
    
    X5 RETAILHERO DATASET:
    ---------------------
    • Satır: 200K (train)
    • Features: ??? (henüz feature engineering yapmadık)
    • Transaction history: 45M+ satır (purchases tablosu)
    • Treatment: Binary (0/1) - kampanya gönderildi mi?
    • Target: Binary (0/1) - satın alma yaptı mı?
    • Use case: Retail marketing
    • Format: Dictionary (clients, purchases, train)
    
    ÖNEMLİ FARKLAR:
    --------------
    1. ❗ Criteo: Hazır features (f0-f11)
       X5: Transaction history var, feature engineering gerekli!
    
    2. ❗ Criteo: Flat format (her satır bir observation)
       X5: Relational format (3 tablo: clients, purchases, train)
    
    3. ✅ İKİSİ DE: Binary treatment/target
    4. ✅ İKİSİ DE: scikit-uplift ile uyumlu
    
    SONUÇ:
    ------
    X5 kullanmak için ÖNCE feature engineering yapmalıyız!
    RFM gibi metrikler hesaplayıp flat format'a çevirmeliyiz.
    """)

def show_sample_data(dataset):
    """Örnek veriyi göster"""
    print("\n" + "="*80)
    print("📊 ÖRNEK VERİLER")
    print("="*80)
    
    # Clients
    print("\n1. CLIENTS (ilk 3 satır):")
    print("-" * 60)
    print(dataset.data['clients'].head(3))
    
    # Purchases
    print("\n2. PURCHASES (ilk 3 satır):")
    print("-" * 60)
    print(dataset.data['purchases'].head(3))
    
    # Train
    print("\n3. TRAIN (ilk 3 satır):")
    print("-" * 60)
    print(dataset.data['train'].head(3))
    
    # Treatment & Target
    print("\n4. TREATMENT & TARGET (ilk 10 değer):")
    print("-" * 60)
    if hasattr(dataset, 'treatment') and hasattr(dataset, 'target'):
        sample_df = pd.DataFrame({
            'treatment': dataset.treatment[:10],
            'target': dataset.target[:10]
        })
        print(sample_df)

def main():
    """Ana analiz"""
    print("="*80)
    print("🚀 X5 RETAILHERO - VERİ TANıMA VE UYUMLULUK ANALİZİ")
    print("   scikit-uplift formatına uygun mu kontrol ediyoruz")
    print("="*80)
    
    # 1. Dataset yükle
    print("\n⏳ X5 RetailHero yükleniyor...")
    dataset = fetch_x5()
    print("✅ Yüklendi!\n")
    
    # 2. Veri yapısını analiz et
    analyze_x5_structure(dataset)
    
    # 3. scikit-uplift uyumluluğunu kontrol et
    requirements, all_ok = check_sklift_requirements(dataset)
    
    # 4. Criteo ile karşılaştır
    compare_with_criteo()
    
    # 5. Örnek veriyi göster
    show_sample_data(dataset)
    
    # 6. SONUÇ VE ÖNERİLER
    print("\n" + "="*80)
    print("💡 SONUÇ VE ÖNERİLER")
    print("="*80)
    
    if all_ok:
        print("""
✅ X5 RetailHero dataset'i scikit-uplift ile UYUMLU!

ANCAK ÖNEMLİ NOT:
-----------------
X5'in transaction history formatı var (purchases tablosu).
scikit-uplift modelleri FLAT FORMAT bekler (her satır = bir müşteri).

YAPMAMIZ GEREKENLER:
-------------------
1. ✅ purchases tablosundan FEATURE ENGINEERING:
   • RFM metrikleri (Recency, Frequency, Monetary)
   • Ortalama sepet büyüklüğü
   • En çok alınan ürünler
   • Alışveriş zamanı özellikleri
   • vs.

2. ✅ clients + train tablolarını birleştir
3. ✅ treatment + target ekle
4. ✅ Final flat format oluştur

SONRAKI ADIM:
------------
Feature engineering script'i çalıştır:
→ python scripts/create_features_x5.py
        """)
    else:
        print("\n⚠️  Bazı eksiklikler var, detayları yukarıda görebilirsin")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()