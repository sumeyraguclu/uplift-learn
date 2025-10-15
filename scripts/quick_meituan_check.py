"""
Meituan dataset hızlı kontrol ve analiz
"""

import pandas as pd
from pathlib import Path

# Paths
data_dir = Path("data/MT-LIFT")
train_path = data_dir / "data" / "train.csv"
test_path = data_dir / "data" / "test.csv"

# Alternatif path kontrol
if not train_path.exists():
    # Belki farklı yapıda
    possible_paths = list(data_dir.glob("**/train.csv"))
    if possible_paths:
        train_path = possible_paths[0]
        print(f"✅ Train bulundu: {train_path}")
    else:
        print(f"❌ train.csv bulunamadı!")
        print(f"\n📂 MT-LIFT klasör yapısı:")
        import os
        for root, dirs, files in os.walk(data_dir):
            level = root.replace(str(data_dir), '').count(os.sep)
            indent = ' ' * 2 * level
            print(f'{indent}{os.path.basename(root)}/')
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                print(f'{subindent}{file}')
        exit(1)

print("=" * 60)
print("🔍 MEITUAN DATASET HIZLI ANALİZ")
print("=" * 60)

# İlk 1000 satırı oku (hızlı test)
print("\n📖 İlk 1000 satır okunuyor...")
df = pd.read_csv(train_path, nrows=1000)

print(f"✅ Başarılı!")
print(f"\n📊 Veri Yapısı:")
print(f"   • Satır: {len(df):,}")
print(f"   • Sütun: {len(df.columns)}")
print(f"   • Bellek: {df.memory_usage(deep=True).sum() / 1024:.1f} KB")

print(f"\n📋 Sütunlar ({len(df.columns)} adet):")
print(df.columns.tolist())

print(f"\n🔍 İlk 3 satır:")
print(df.head(3))

print(f"\n📈 Sütun Tipleri:")
print(df.dtypes.value_counts())

print(f"\n✅ Veri başarıyla okundu!")
print(f"\nŞimdi tam pipeline'ı çalıştırmaya hazırsın:")
print(f"   python scripts/process_meituan.py")