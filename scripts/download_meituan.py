"""
Meituan MT-LIFT dataset indirme ve hazırlama

En güncel ve en uygun uplift modeling dataset'i (2024)
- 5.5M satır
- Food delivery + Coupon kampanyası
- 99 features
- 5 treatment tipi
- Entire chain: Click + Conversion
"""

import pandas as pd
import numpy as np
from pathlib import Path
import urllib.request
import zipfile
from tqdm import tqdm

def download_file_with_progress(url: str, output_path: Path):
    """Progress bar ile dosya indir"""
    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path.name) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def main():
    print("=" * 60)
    print("🚀 MEITUAN MT-LIFT DATASET İNDİRME")
    print("=" * 60)
    print("\n📊 Dataset Bilgileri:")
    print("   • Kaynak: Meituan (Çin'in lider food delivery platformu)")
    print("   • Boyut: 5.5M satır, 99 features")
    print("   • Domain: Food delivery + Coupon kampanyası")
    print("   • Treatment: 5 farklı coupon tipi")
    print("   • Outcome: Click + Conversion (entire chain)")
    print("   • RCT: Randomized trial ✅")
    print("   • Yayın: Şubat 2024 (çok güncel!)")
    
    # Paths
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # GitHub repo
    github_repo = "https://github.com/MTDJDSP/MT-LIFT"
    
    # Dataset dosyaları (GitHub'dan direkt indirme linkleri)
    # Not: Gerçek linkleri README'den alacağız
    print("\n⏱️  Tahmini süre: 10-15 dakika")
    print("   • İndirme: 5-10 dk")
    print("   • İşleme: 3-5 dk")
    
    print(f"\n📥 İndirme Yöntemi:")
    print(f"\n🔗 GitHub Repo: {github_repo}")
    print("\n⚠️  NOT: Bu dataset GitHub LFS kullanıyor.")
    print("   Manuel indirme önerilir:")
    print("\n   1. GitHub'a git: https://github.com/MTDJDSP/MT-LIFT")
    print("   2. 'Code' → 'Download ZIP' tıkla")
    print("   3. ZIP'i data/ klasörüne çıkart")
    print("\n   VEYA Git ile klon:")
    print("   git clone https://github.com/MTDJDSP/MT-LIFT")
    print("   cd MT-LIFT")
    print("   git lfs pull  # LFS dosyalarını indir")
    
    # Alternatif: Kullanıcıya yol göster
    print("\n" + "=" * 60)
    print("💡 MANUEL İNDİRME TALİMATLARI")
    print("=" * 60)
    
    print("\n1️⃣ Tarayıcıdan:")
    print("   https://github.com/MTDJDSP/MT-LIFT")
    print("   → 'Releases' sekmesine git")
    print("   → En son release'i indir")
    
    print("\n2️⃣ Git CLI ile (önerilen):")
    print("   cd data")
    print("   git clone https://github.com/MTDJDSP/MT-LIFT")
    
    print("\n3️⃣ Dataset yapısı:")
    print("   MT-LIFT/")
    print("   ├── data/")
    print("   │   ├── train.csv    # Training set")
    print("   │   └── test.csv     # Test set")
    print("   └── README.md")
    
    # Veri olup olmadığını kontrol et
    meituan_dir = data_dir / "MT-LIFT"
    
    if meituan_dir.exists():
        print(f"\n✅ {meituan_dir} klasörü bulundu!")
        
        # İçeriği kontrol et
        train_file = meituan_dir / "data" / "train.csv"
        test_file = meituan_dir / "data" / "test.csv"
        
        if train_file.exists() and test_file.exists():
            print(f"✅ Dataset dosyaları mevcut!")
            
            # Veriyi yükle ve analiz et
            print("\n📊 Veri analizi yapılıyor...")
            analyze_meituan_dataset(train_file, test_file)
        else:
            print(f"⚠️  data/ klasöründe CSV dosyaları bulunamadı")
            print(f"   Beklenen:")
            print(f"   • {train_file}")
            print(f"   • {test_file}")
    else:
        print(f"\n❌ {meituan_dir} bulunamadı")
        print("\nLütfen yukarıdaki talimatları izleyerek indirin.")
    
    print("\n" + "=" * 60)
    print("📝 SONRAKI ADIMLAR")
    print("=" * 60)
    print("\n1. Dataset'i indir (yukarıdaki yöntemlerle)")
    print("2. python scripts/process_meituan.py  # Veriyi işle")
    print("3. python tests/test_data.py  # Test et")


def analyze_meituan_dataset(train_path: Path, test_path: Path):
    """Meituan dataset'ini analiz et"""
    print("\n" + "=" * 60)
    print("📊 MEITUAN DATASET ANALİZİ")
    print("=" * 60)
    
    # Training set
    print("\n📖 Training set okunuyor...")
    df_train = pd.read_csv(train_path, nrows=10000)  # İlk 10K satır
    
    print(f"   ✅ Başarılı! (İlk 10K satır)")
    print(f"   • Sütun sayısı: {len(df_train.columns)}")
    print(f"   • Satır sayısı (sample): {len(df_train):,}")
    
    print(f"\n📋 Sütunlar:")
    print(df_train.columns.tolist()[:20])  # İlk 20 sütun
    if len(df_train.columns) > 20:
        print(f"   ... ve {len(df_train.columns) - 20} sütun daha")
    
    # Treatment dağılımı
    if 'treatment' in df_train.columns:
        print(f"\n🎯 Treatment Dağılımı:")
        print(df_train['treatment'].value_counts())
        print(f"   Treatment tipi sayısı: {df_train['treatment'].nunique()}")
    
    # Outcome dağılımı
    if 'click' in df_train.columns:
        print(f"\n📈 Click Rate: {df_train['click'].mean():.2%}")
    
    if 'conversion' in df_train.columns:
        print(f"💰 Conversion Rate: {df_train['conversion'].mean():.2%}")
    
    # İlk satırlar
    print(f"\n🔍 İlk 3 Satır:")
    print(df_train.head(3))
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()