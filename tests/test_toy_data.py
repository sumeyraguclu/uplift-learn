# test_toy_data.py oluştur
from src.data import (
    create_toy_dataset,
    train_test_split_uplift,
    check_treatment_balance,
    calculate_baseline_metrics
)

print("🎯 TOY DATASET TESTİ")
print("=" * 60)

# 1. Toy dataset oluştur
df = create_toy_dataset(n_samples=5000, treatment_effect_size=0.15)
print(f"✅ {len(df):,} satırlık toy dataset oluşturuldu")

# 2. Train/test split
X_train, X_test, y_train, y_test, t_train, t_test = \
    train_test_split_uplift(df, test_size=0.25, random_state=42)

print(f"\n📊 Train size: {len(X_train):,}")
print(f"📊 Test size: {len(X_test):,}")

# 3. Treatment balance
check_treatment_balance(t_train, "Training Set")
check_treatment_balance(t_test, "Test Set")

# 4. Baseline metrics
print("\n📈 BASELINE METRICS")
metrics = calculate_baseline_metrics(y_train, t_train)

print("\n✅ Toy dataset testi başarılı!")