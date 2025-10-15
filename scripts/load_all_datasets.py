"""
Tüm scikit-uplift dataset'lerini yükle ve karşılaştır
"""

from sklift.datasets import fetch_x5, fetch_lenta, fetch_megafon, fetch_hillstrom, fetch_criteo
import pandas as pd

print("=" * 80)
print("🚀 UPLIFT MODELING DATASETS YÜKLEME")
print("=" * 80)
 
datasets_info = []

# 1. X5 RetailHero
print("\n1️⃣ X5 RetailHero yükleniyor...")
try:
    x5 = fetch_x5()
    print(f"   ✅ Başarılı!")
    print(f"   • Clients: {len(x5.data['clients']):,} satır")
    print(f"   • Purchases: {len(x5.data['purchases']):,} satır")
    print(f"   • Train: {len(x5.data['train']):,} satır")
    
    datasets_info.append({
        'Dataset': 'X5 RetailHero',
        'Size': f"{len(x5.data['train']):,}",
        'Domain': 'Retail/Grocery',
        'Status': '✅'
    })
except Exception as e:
    print(f"   ❌ Hata: {e}")
    datasets_info.append({'Dataset': 'X5 RetailHero', 'Status': '❌'})

# 2. Lenta
print("\n2️⃣ Lenta yükleniyor...")
try:
    lenta = fetch_lenta()
    X, y, t = lenta.data, lenta.target, lenta.treatment
    print(f"   ✅ Başarılı!")
    print(f"   • Satır: {len(X):,}")
    print(f"   • Features: {X.shape[1] if hasattr(X, 'shape') else 'N/A'}")
    print(f"   • Treatment ratio: {t.mean():.2%}")
    print(f"   • Response rate: {y.mean():.2%}")
    
    datasets_info.append({
        'Dataset': 'Lenta',
        'Size': f"{len(X):,}",
        'Domain': 'Grocery',
        'Status': '✅'
    })
except Exception as e:
    print(f"   ❌ Hata: {e}")
    datasets_info.append({'Dataset': 'Lenta', 'Status': '❌'})

# 3. MegaFon
print("\n3️⃣ MegaFon yükleniyor...")
try:
    megafon = fetch_megafon()
    X, y, t = megafon.data, megafon.target, megafon.treatment
    print(f"   ✅ Başarılı!")
    print(f"   • Satır: {len(X):,}")
    print(f"   • Features: {X.shape[1]}")
    print(f"   • Response rate: {y.mean():.2%}")
    
    datasets_info.append({
        'Dataset': 'MegaFon',
        'Size': f"{len(X):,}",
        'Domain': 'Telecom',
        'Status': '✅'
    })
except Exception as e:
    print(f"   ❌ Hata: {e}")
    datasets_info.append({'Dataset': 'MegaFon', 'Status': '❌'})

# 4. Hillstrom
print("\n4️⃣ Hillstrom yükleniyor...")
try:
    X, y, t = fetch_hillstrom(return_X_y_t=True)
    print(f"   ✅ Başarılı!")
    print(f"   • Satır: {len(X):,}")
    print(f"   • Features: {X.shape[1]}")
    print(f"   • Treatment groups: {len(t.unique())}")
    
    datasets_info.append({
        'Dataset': 'Hillstrom',
        'Size': f"{len(X):,}",
        'Domain': 'Email Marketing',
        'Status': '✅'
    })
except Exception as e:
    print(f"   ❌ Hata: {e}")
    datasets_info.append({'Dataset': 'Hillstrom', 'Status': '❌'})

# 5. Criteo (10% sample)
print("\n5️⃣ Criteo (10% sample) yükleniyor...")
try:
    X, y, t = fetch_criteo(percent10=True, return_X_y_t=True)
    print(f"   ✅ Başarılı!")
    print(f"   • Satır: {len(X):,}")
    print(f"   • Features: {X.shape[1]}")
    
    datasets_info.append({
        'Dataset': 'Criteo (10%)',
        'Size': f"{len(X):,}",
        'Domain': 'Display Ads',
        'Status': '✅'
    })
except Exception as e:
    print(f"   ❌ Hata: {e}")
    datasets_info.append({'Dataset': 'Criteo (10%)', 'Status': '❌'})

# Özet tablo
print("\n" + "=" * 80)
print("📊 DATASET ÖZET TABLOSU")
print("=" * 80)

df_summary = pd.DataFrame(datasets_info)
print(df_summary.to_string(index=False))

# Öneriler
print("\n" + "=" * 80)
print("💡 ÖNERİLER")
print("=" * 80)

print("\n🥇 EN İYİ 3 DATASET (Projen İçin):")
print("   1. X5 RetailHero - 2M+ satır, retail, transaction history")
print("   2. Lenta - 687K satır, grocery, demografik features")
print("   3. Hillstrom - 64K satır, email marketing, RFM features")

print("\n🎯 KULLANIM STRATEJİSİ:")
print("   • MVP: Hillstrom (hızlı, anlamlı features)")
print("   • Production: X5 RetailHero (büyük, gerçek veri)")
print("   • Validation: Lenta (orta ölçek, dengeli)")

print("\n🚀 SONRAKI ADIM:")
print("   python scripts/process_x5_retailhero.py  # X5'i işle")

print("\n" + "=" * 80)