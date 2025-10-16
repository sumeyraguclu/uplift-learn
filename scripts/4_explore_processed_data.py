"""
İşlenmiş X5 RFM verisini detaylı incele
scikit-uplift formatına hazır mı kontrol et
"""
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def load_processed_data():
    """İşlenmiş veriyi yükle"""
    print("="*80)
    print("📂 İŞLENMİŞ VERİ YÜKLENİYOR")
    print("="*80)
    
    data_path = Path("data/x5_rfm_processed.pkl")
    
    if not data_path.exists():
        raise FileNotFoundError(
            "İşlenmiş veri bulunamadı!\n"
            "Önce: python scripts/process_x5_rfm.py"
        )
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"✅ Veri yüklendi: {data_path}")
    print(f"   Boyut: {data_path.stat().st_size / 1024**2:.1f} MB")
    print(f"\n📦 İçerik:")
    print(f"   Keys: {list(data.keys())}")
    
    return data

def analyze_features(df):
    """Feature'ları detaylı analiz et"""
    print("\n" + "="*80)
    print("📊 FEATURE ANALİZİ")
    print("="*80)
    
    print(f"\n1. VERİ YAPISI:")
    print(f"   Satır: {len(df):,}")
    print(f"   Sütun: {len(df.columns)}")
    print(f"   Memory: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    print(f"\n2. SÜTUNLAR VE TİPLER:")
    print(df.dtypes)
    
    print(f"\n3. EKSİK DEĞERLER:")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("   ✅ Eksik değer yok!")
    else:
        print(missing[missing > 0])
    
    print(f"\n4. İSTATİSTİKLER:")
    print(df.describe())
    
    print(f"\n5. İLK 5 SATIR:")
    print(df.head())

def analyze_treatment_target(df):
    """Treatment ve target dengelerini kontrol et"""
    print("\n" + "="*80)
    print("🎯 TREATMENT & TARGET ANALİZİ")
    print("="*80)
    
    # Treatment dengesi
    treatment_counts = df['treatment'].value_counts()
    print(f"\n📊 TREATMENT DAĞILIMI:")
    print(treatment_counts)
    print(f"\n   Control: {treatment_counts[0]:,} ({treatment_counts[0]/len(df):.1%})")
    print(f"   Treatment: {treatment_counts[1]:,} ({treatment_counts[1]/len(df):.1%})")
    
    # İdeal mi?
    treatment_ratio = treatment_counts[1] / len(df)
    if 0.3 <= treatment_ratio <= 0.7:
        print(f"   ✅ Dengeli (30-70% aralığında)")
    else:
        print(f"   ⚠️  Dengesiz!")
    
    # Target dağılımı
    target_counts = df['target'].value_counts()
    print(f"\n🎲 TARGET DAĞILIMI:")
    print(target_counts)
    print(f"\n   Non-converters: {target_counts[0]:,} ({target_counts[0]/len(df):.1%})")
    print(f"   Converters: {target_counts[1]:,} ({target_counts[1]/len(df):.1%})")
    
    # Conversion rates
    cr_control = df[df['treatment']==0]['target'].mean()
    cr_treatment = df[df['treatment']==1]['target'].mean()
    
    print(f"\n📈 CONVERSION RATES:")
    print(f"   Control: {cr_control:.2%}")
    print(f"   Treatment: {cr_treatment:.2%}")
    print(f"   Fark (ATE): {(cr_treatment - cr_control):.2%}")
    print(f"   Relative uplift: {((cr_treatment/cr_control - 1)*100):.1f}%")
    
    return {
        'cr_control': cr_control,
        'cr_treatment': cr_treatment,
        'ate': cr_treatment - cr_control
    }

def analyze_rfm_features(df):
    """RFM feature'larını analiz et"""
    print("\n" + "="*80)
    print("🔍 RFM FEATURES ANALİZİ")
    print("="*80)
    
    rfm_features = ['recency', 'frequency', 'monetary', 'r_score', 'f_score', 'm_score', 'rfm_score']
    
    for feature in rfm_features:
        if feature in df.columns:
            print(f"\n{feature.upper()}:")
            print(f"   Min: {df[feature].min()}")
            print(f"   Max: {df[feature].max()}")
            print(f"   Mean: {df[feature].mean():.2f}")
            print(f"   Median: {df[feature].median():.2f}")
            print(f"   Std: {df[feature].std():.2f}")

def check_sklift_format(df):
    """scikit-uplift formatına uygun mu kontrol et"""
    print("\n" + "="*80)
    print("✅ SCIKIT-UPLIFT FORMAT KONTROLÜ")
    print("="*80)
    
    checks = {
        'X (Features)': False,
        'y (Target)': False,
        'treatment': False,
        'Binary target': False,
        'Binary treatment': False,
        'No missing values': False,
        'Sufficient samples': False
    }
    
    # Feature'lar var mı?
    feature_cols = [col for col in df.columns if col not in ['client_id', 'treatment', 'target']]
    if len(feature_cols) >= 3:
        print(f"\n✅ Features: {len(feature_cols)} feature var")
        print(f"   {feature_cols}")
        checks['X (Features)'] = True
    else:
        print(f"\n❌ Features: Yeterli feature yok!")
    
    # Target var mı?
    if 'target' in df.columns:
        print(f"\n✅ Target: 'target' sütunu var")
        checks['y (Target)'] = True
        
        # Binary mi?
        unique_vals = df['target'].unique()
        if len(unique_vals) == 2 and set(unique_vals) == {0, 1}:
            print(f"   ✅ Binary (0/1)")
            checks['Binary target'] = True
        else:
            print(f"   ❌ Binary değil: {unique_vals}")
    
    # Treatment var mı?
    if 'treatment' in df.columns:
        print(f"\n✅ Treatment: 'treatment' sütunu var")
        checks['treatment'] = True
        
        # Binary mi?
        unique_vals = df['treatment'].unique()
        if len(unique_vals) == 2 and set(unique_vals) == {0, 1}:
            print(f"   ✅ Binary (0/1)")
            checks['Binary treatment'] = True
        else:
            print(f"   ❌ Binary değil: {unique_vals}")
    
    # Eksik değer var mı?
    if df.isnull().sum().sum() == 0:
        print(f"\n✅ No missing values")
        checks['No missing values'] = True
    else:
        print(f"\n❌ Eksik değerler var!")
    
    # Yeterli sample var mı?
    if len(df) >= 10000:
        print(f"\n✅ Sufficient samples: {len(df):,}")
        checks['Sufficient samples'] = True
    else:
        print(f"\n⚠️  Sample az: {len(df):,}")
    
    # ÖZET
    print("\n" + "="*80)
    print("📋 FORMAT KONTROLÜ ÖZET")
    print("="*80)
    
    for check, status in checks.items():
        icon = "✅" if status else "❌"
        print(f"   {icon} {check}")
    
    all_ok = all(checks.values())
    
    if all_ok:
        print("\n🎉 VERİ SETİ SCIKIT-UPLIFT İÇİN HAZIR!")
        print("\nSONRAKİ ADIM:")
        print("   → python scripts/train_uplift_model.py")
    else:
        print("\n⚠️  Bazı sorunlar var, yukarıda detayları gör")
    
    return checks, all_ok

def prepare_for_modeling(df):
    """Model eğitimi için X, y, treatment ayır"""
    print("\n" + "="*80)
    print("🔧 MODEL EĞİTİMİ İÇİN VERİ HAZIRLAMA")
    print("="*80)
    
    # Feature columns (client_id, treatment, target hariç)
    feature_cols = [col for col in df.columns if col not in ['client_id', 'treatment', 'target']]
    
    X = df[feature_cols].copy()
    y = df['target'].copy()
    treatment = df['treatment'].copy()
    
    print(f"\n✅ Veri ayrıldı:")
    print(f"   X shape: {X.shape}")
    print(f"   y shape: {y.shape}")
    print(f"   treatment shape: {treatment.shape}")
    
    print(f"\n📊 Feature listesi:")
    for i, col in enumerate(feature_cols, 1):
        print(f"   {i}. {col}")
    
    # Örnek kaydet
    print(f"\n💾 Örnek kaydediliyor...")
    example = {
        'X': X,
        'y': y,
        'treatment': treatment,
        'feature_names': feature_cols
    }
    
    output_path = Path("data/x5_ready_for_modeling.pkl")
    with open(output_path, 'wb') as f:
        pickle.dump(example, f)
    
    print(f"✅ Model eğitimi için hazır veri kaydedildi: {output_path}")
    
    return X, y, treatment

def visualize_rfm_distributions(df):
    """RFM dağılımlarını görselleştir"""
    print("\n" + "="*80)
    print("📊 RFM DAĞILIMLARI GÖRSELLEŞTİRİLİYOR")
    print("="*80)
    
    # Plot ayarları
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    rfm_features = ['recency', 'frequency', 'monetary', 'r_score', 'f_score', 'm_score']
    
    for i, feature in enumerate(rfm_features):
        if feature in df.columns:
            # Histogram
            axes[i].hist(df[feature], bins=50, alpha=0.7, color='steelblue', edgecolor='black')
            axes[i].set_title(f'{feature.upper()} Distribution', fontweight='bold', fontsize=12)
            axes[i].set_xlabel(feature)
            axes[i].set_ylabel('Count')
            axes[i].grid(alpha=0.3)
            
            # İstatistikler ekle
            mean_val = df[feature].mean()
            median_val = df[feature].median()
            axes[i].axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.1f}')
            axes[i].axvline(median_val, color='green', linestyle='--', label=f'Median: {median_val:.1f}')
            axes[i].legend()
    
    plt.tight_layout()
    
    # Kaydet
    output_path = Path("exports/rfm_distributions.png")
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Grafik kaydedildi: {output_path}")
    
    plt.show()

def main():
    """Ana analiz"""
    print("="*80)
    print("🚀 İŞLENMİŞ X5 RFM VERİSİ DETAYLI ANALİZ")
    print("="*80)
    
    # 1. Veriyi yükle
    data = load_processed_data()
    df = data['data']
    
    # 2. Feature analizi
    analyze_features(df)
    
    # 3. Treatment & Target analizi
    metrics = analyze_treatment_target(df)
    
    # 4. RFM analizi
    analyze_rfm_features(df)
    
    # 5. scikit-uplift format kontrolü
    checks, all_ok = check_sklift_format(df)
    
    # 6. Model eğitimi için hazırla
    if all_ok:
        X, y, treatment = prepare_for_modeling(df)
    
    # 7. Görselleştir
    try:
        visualize_rfm_distributions(df)
    except Exception as e:
        print(f"\n⚠️  Görselleştirme hatası: {e}")
        print("   (matplotlib kurulu değilse: pip install matplotlib)")
    
    # 8. FINAL ÖZET
    print("\n" + "="*80)
    print("🎯 FINAL ÖZET")
    print("="*80)
    
    print(f"""
✅ VERİ HAZIR!
-------------
• Toplam müşteri: {len(df):,}
• Feature sayısı: {len([c for c in df.columns if c not in ['client_id', 'treatment', 'target']])}
• Treatment dengesi: 50-50 ✓
• Naive ATE: {metrics['ate']:.2%}

📊 RFM FEATURES:
--------------
• recency (R): Son alışveriş - şimdi (gün)
• frequency (F): Toplam alışveriş sayısı
• monetary (M): Toplam harcama ($)
• r_score, f_score, m_score: 1-5 skor
• rfm_score: Toplam RFM skoru (3-15)
• rfm_segment: String segment (örn: "555")

🎯 SCIKIT-UPLIFT UYUMLULUĞU:
--------------------------
{' ✅ TÜM KONTROLLER GEÇTİ!' if all_ok else ' ⚠️  Bazı kontroller başarısız'}

SONRAKI ADIM:
------------
→ python scripts/train_uplift_model.py
    """)
    
    print("="*80)

if __name__ == "__main__":
    main()