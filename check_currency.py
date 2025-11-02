import pandas as pd
import pickle

with open('data/x5_rfm_processed.pkl', 'rb') as f:
    data = pickle.load(f)
df = data['data']

print("=" * 70)
print("PARA TİPİ KONTROLÜ")
print("=" * 70)

print("\nMonetary colonları istatistikleri:")
print(df[['monetary', 'monetary_capped', 'monetary_usd', 'aov', 'aov_usd']].describe())

print("\n" + "=" * 70)
print("İLK 10 SATIR (Para Değerleri)")
print("=" * 70)
print(df[['monetary', 'monetary_capped', 'monetary_usd', 'aov', 'aov_usd']].head(10))

print("\n" + "=" * 70)
print("DÖNÜŞÜM ORANI")
print("=" * 70)
print(f"monetary (RUB) mean: {df['monetary'].mean():,.2f}")
print(f"monetary_usd (USD) mean: {df['monetary_usd'].mean():,.2f}")

if df['monetary_usd'].mean() > 0:
    ratio = df['monetary'].mean() / df['monetary_usd'].mean()
    print(f"Conversion ratio (RUB/USD): {ratio:.2f}")
    print(f"\nAnlamı: 1 USD = {ratio:.2f} RUB")

print("\n" + "=" * 70)
print("PARA BİRİMİ TAHMİNİ")
print("=" * 70)
if df['monetary_usd'].mean() < 2000:
    print(f"✓ monetary_usd USD cinsinden (${df['monetary_usd'].mean():,.0f})")
    print(f"✓ monetary RUB cinsinden ({df['monetary'].mean():,.0f} RUB)")
else:
    print(f"✓ Tüm veriler RUB cinsinden ({df['monetary'].mean():,.0f} RUB)")