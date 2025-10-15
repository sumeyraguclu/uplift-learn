"""
Criteo veri setini güvenli şekilde indir

100K dengeli sample için optimize edildi
"""

import pandas as pd
import requests
from pathlib import Path
from tqdm import tqdm
import gzip
import shutil

def download_with_progress(url: str, output_path: Path):
    """Progress bar ile güvenli indirme"""
    print(f"\n📥 İndiriliyor: {url}")
    print("⏱️  Tahmini süre: 5-10 dakika")
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(output_path, 'wb') as f, tqdm(
        desc=output_path.name,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))
    
    print(f"✅ İndirme tamamlandı!")


def extract_gz(gz_path: Path, csv_path: Path):
    """GZ dosyasını aç"""
    print(f"\n📦 Açılıyor: {gz_path.name}")
    print("⏱️  Tahmini süre: 2-3 dakika")
    
    with gzip.open(gz_path, 'rb') as f_in:
        with open(csv_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    
    print(f"✅ Açıldı: {csv_path.name}")


def create_balanced_sample(csv_path: Path, output_path: Path, sample_size: int = 100_000):
    """Dengeli sample oluştur"""
    print(f"\n🔬 {sample_size:,} satırlık dengeli sample oluşturuluyor...")
    n_per_group = sample_size // 2
    
    # Treatment: İlk 2M satırdan
    print(f"\n1️⃣ Treatment sampling (ilk 2M satır)...")
    print("   ⏱️  ~1 dakika...")
    
    df_head = pd.read_csv(csv_path, nrows=2_000_000)
    df_treatment = df_head[df_head['treatment'] == 1]
    
    if len(df_treatment) < n_per_group:
        raise ValueError(f"Yetersiz treatment! {len(df_treatment)} < {n_per_group}")
    
    df_treatment_sample = df_treatment.sample(n=n_per_group, random_state=42)
    print(f"   ✅ {n_per_group:,} treatment sample alındı")
    
    # Control: Son 3M satırdan
    print(f"\n2️⃣ Control sampling (son 3M satır)...")
    print("   ⏱️  ~2 dakika...")
    
    # Toplam satır sayısı (yaklaşık)
    total_lines = 13_900_000
    skip_rows = total_lines - 3_000_000
    
    df_tail = pd.read_csv(
        csv_path,
        skiprows=range(1, skip_rows),
        names=['f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11',
               'treatment', 'conversion', 'visit', 'exposure'],
        low_memory=False
    )
    
    df_control = df_tail[df_tail['treatment'] == 0]
    
    if len(df_control) < n_per_group:
        print(f"   ⚠️  Control az ({len(df_control):,}), tümünü kullanıyoruz")
        df_control_sample = df_control
    else:
        df_control_sample = df_control.sample(n=n_per_group, random_state=42)
        print(f"   ✅ {n_per_group:,} control sample alındı")
    
    # Birleştir
    print(f"\n3️⃣ Birleştirme ve kaydetme...")
    df_balanced = pd.concat([df_treatment_sample, df_control_sample], ignore_index=True)
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Kaydet
    df_balanced.to_parquet(output_path, index=False, compression='snappy')
    
    # Rapor
    print(f"\n✅ Kaydedildi: {output_path}")
    print(f"   Dosya boyutu: {output_path.stat().st_size / 1024**2:.1f} MB")
    print(f"   Toplam:    {len(df_balanced):,} satır")
    print(f"   Treatment: {(df_balanced['treatment']==1).sum():,} ({(df_balanced['treatment']==1).mean():.1%})")
    print(f"   Control:   {(df_balanced['treatment']==0).sum():,} ({(df_balanced['treatment']==0).mean():.1%})")
    
    # Baseline metrics
    print(f"\n📊 Baseline Metrikleri:")
    cr_t = df_balanced[df_balanced['treatment']==1]['visit'].mean()
    cr_c = df_balanced[df_balanced['treatment']==0]['visit'].mean()
    ate = cr_t - cr_c
    
    print(f"   CR (Treatment): {cr_t:.4f} ({cr_t:.2%})")
    print(f"   CR (Control):   {cr_c:.4f} ({cr_c:.2%})")
    print(f"   ATE:            {ate:+.4f}")
    
    if cr_c > 0:
        print(f"   Relative:       {ate/cr_c:+.2%}")


def main():
    print("=" * 60)
    print("🚀 CRITEO DATASET İNDİRME VE HAZIRLIK")
    print("=" * 60)
    print("\n⏱️  Toplam süre: ~15-20 dakika")
    print("   • İndirme: 5-10 dk")
    print("   • Açma: 2-3 dk")
    print("   • Sampling: 3-5 dk")
    
    # Paths
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    url = "http://go.criteo.net/criteo-research-uplift-v2.1.csv.gz"
    gz_file = data_dir / "criteo-uplift-v2.1.csv.gz"
    csv_file = data_dir / "criteo-uplift-v2.1.csv"
    parquet_file = data_dir / "criteo_sample.parquet"
    
    # 1. İndir
    if not gz_file.exists() and not csv_file.exists():
        try:
            download_with_progress(url, gz_file)
        except Exception as e:
            print(f"\n❌ İndirme hatası: {e}")
            print("\n💡 Alternatifler:")
            print("   1. İnternet bağlantınızı kontrol edin")
            print("   2. VPN kullanmayı deneyin")
            print("   3. Manuel indirin: " + url)
            return
    else:
        print("\n✅ Dosya zaten mevcut, indirme atlanıyor")
    
    # 2. Aç
    if gz_file.exists() and not csv_file.exists():
        try:
            extract_gz(gz_file, csv_file)
            # GZ'yi sil (yer kazan)
            gz_file.unlink()
            print("🗑️  GZ dosyası silindi")
        except Exception as e:
            print(f"\n❌ Açma hatası: {e}")
            print("GZ dosyası bozuk olabilir. Silin ve tekrar indirin:")
            print(f"   del {gz_file}")
            return
    elif csv_file.exists():
        print("✅ CSV zaten mevcut, açma atlanıyor")
    
    # 3. Sample oluştur
    if not parquet_file.exists():
        try:
            create_balanced_sample(csv_file, parquet_file, sample_size=100_000)
        except Exception as e:
            print(f"\n❌ Sampling hatası: {e}")
            return
    else:
        print(f"\n✅ {parquet_file.name} zaten mevcut!")
    
    # 4. Özet
    print("\n" + "=" * 60)
    print("🎉 TAMAMLANDI!")
    print("=" * 60)
    
    print("\n📁 Oluşturulan Dosyalar:")
    if csv_file.exists():
        print(f"   • {csv_file} ({csv_file.stat().st_size / 1024**3:.1f} GB)")
    if parquet_file.exists():
        print(f"   • {parquet_file} ({parquet_file.stat().st_size / 1024**2:.1f} MB)")
    
    print("\n💾 Disk Kullanımı:")
    if csv_file.exists():
        print(f"   CSV: {csv_file.stat().st_size / 1024**3:.1f} GB")
        print("   💡 CSV'yi silebilirsiniz (parquet yeterli):")
        print(f"      del {csv_file}")
    
    print("\n🎯 Sonraki Adımlar:")
    print("   1. ✅ Veri hazır!")
    print("   2. python tests/test_data.py -v  # Test et")
    print("   3. jupyter notebook  # Data exploration")
    print("   4. T-Learner implementasyonuna geç!")


if __name__ == "__main__":
    main()