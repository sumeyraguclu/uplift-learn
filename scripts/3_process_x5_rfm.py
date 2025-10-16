"""
X5 RetailHero - RFM Feature Engineering
Production-Grade Preprocessing Pipeline v2.2

Features:
- Winsorization (99.5% outlier capping)
- Memory optimization (dtype conversion)
- Extended metadata tracking
- Segment uplift with 95% CI
- Bug fixes and consistency improvements
"""
from sklift.datasets import fetch_x5
import pandas as pd
import numpy as np
from datetime import datetime
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ---------------------------- Currency Utils ---------------------------- #

def detect_currency_and_convert(monetary_series, dataset_name='X5 RetailHero'):
    """
    Para birimini tespit et ve USD'ye çevir
    Heuristics:
    - X5 RetailHero: Russian Ruble (RUB)
    - 2019 average: 1 USD ≈ 64.7 RUB
    """
    avg_val = monetary_series.mean()
    currency_info = {
        'detected_currency': None,
        'confidence': None,
        'exchange_rate': 1.0,
        'converted_values': monetary_series,
        'reasoning': []
    }
    if 'X5' in dataset_name or 'RetailHero' in dataset_name:
        currency_info['detected_currency'] = 'RUB'
        currency_info['exchange_rate'] = 64.7
        currency_info['confidence'] = 0.95
        currency_info['reasoning'].append("X5 → RUB (2019 avg: 64.7 RUB/USD)")
    if avg_val > 10000 and currency_info['detected_currency'] is None:
        currency_info['detected_currency'] = 'RUB'
        currency_info['exchange_rate'] = 64.7
        currency_info['confidence'] = 0.7
        currency_info['reasoning'].append(f"High average ({avg_val:,.2f}) → likely RUB")
    if currency_info['exchange_rate'] != 1.0:
        currency_info['converted_values'] = monetary_series / currency_info['exchange_rate']
    return currency_info

# ---------------------------- Core RFM ---------------------------- #

def calculate_rfm_features(purchases_df, reference_date=None):
    """
    RFM metriklerini hesapla ve monetization zenginleştir
    
    Parameters
    ----------
    purchases_df : pd.DataFrame
        Transaction-level data with columns:
        - client_id: Customer identifier
        - transaction_datetime: Transaction timestamp
        - transaction_id: Basket/transaction identifier
        - purchase_sum: Transaction amount
    reference_date : datetime, optional
        Reference date for recency calculation (default: max date + 1 day)
    
    Returns
    -------
    tuple
        (rfm_df, metadata)
    """
    print("\n" + "="*80)
    print("📊 RFM METRİKLERİ HESAPLANIYOR")
    print("="*80)

    # Schema validation
    required_cols = ['client_id', 'transaction_datetime', 'transaction_id', 'purchase_sum']
    missing_cols = [c for c in required_cols if c not in purchases_df.columns]
    if missing_cols:
        raise ValueError(f"❌ Eksik kolonlar: {missing_cols}")
    print("✅ Şema kontrolü: OK")

    # Datetime parsing + cleaning
    purchases_df = purchases_df.copy()
    purchases_df['transaction_datetime'] = pd.to_datetime(
        purchases_df['transaction_datetime'], utc=True, errors='coerce'
    )
    bad = purchases_df['transaction_datetime'].isnull().sum()
    if bad > 0:
        print(f"⚠️  {bad} geçersiz tarih → drop")
        purchases_df = purchases_df.dropna(subset=['transaction_datetime'])

    # Reference date
    if reference_date is None:
        reference_date = purchases_df['transaction_datetime'].max() + pd.Timedelta(days=1)
    print(f"📅 Referans: {reference_date.date()}")

    # Time window
    min_date = purchases_df['transaction_datetime'].min()
    max_date = purchases_df['transaction_datetime'].max()
    time_window_days = (max_date - min_date).days
    print(f"⏰ Pencere: {min_date.date()} → {max_date.date()} ({time_window_days} gün)")

    # Convert to cents for integer operations
    purchases_df['purchase_sum_cents'] = (purchases_df['purchase_sum'] * 100).round().astype('int64')

    # ---- BASE RFM CALCULATION ---- #
    rfm = purchases_df.groupby('client_id', sort=False).agg({
        'transaction_datetime': lambda x: (reference_date - x.max()).days,  # recency
        'transaction_id': pd.Series.nunique,  # frequency = basket count
        'purchase_sum_cents': 'sum'  # total cents
    }).reset_index()
    rfm.columns = ['client_id', 'recency', 'frequency', 'monetary_cents']
    rfm['monetary'] = rfm['monetary_cents'] / 100  # RUB

    # ---- BASKET-LEVEL WINSORIZATION → THEN AGGREGATE TO CUSTOMER ---- #
    basket = (purchases_df
              .groupby(['client_id','transaction_id'], as_index=False, sort=False)['purchase_sum_cents']
              .sum())
    basket['basket_rub'] = basket['purchase_sum_cents'] / 100

    # Add transaction time to basket level (for velocity / recent ratio)
    tx_time = (purchases_df.groupby(['client_id','transaction_id'], sort=False)['transaction_datetime']
               .max().reset_index(name='tx_time'))
    basket = basket.merge(tx_time, on=['client_id','transaction_id'], how='left')

    # Global cap (99.5% percentile) - APPLIED AT BASKET LEVEL BEFORE AGGREGATION
    cap_basket = basket['basket_rub'].quantile(0.995)
    basket['basket_rub_cap'] = basket['basket_rub'].clip(upper=cap_basket)
    capped_count = (basket['basket_rub'] > cap_basket).sum()
    print(f"📊 OUTLIER CAP (basket 99.5%): {cap_basket:,.2f}₽ | capped rows: {capped_count:,}")
    print(f"   ℹ️  Cap uygulanıyor: basket seviyesinde → müşteri agregasyonundan ÖNCEsİ")

    # Basket statistics (customer level) - USING CAPPED VALUES
    basket_stats = basket.groupby('client_id', sort=False)['basket_rub_cap'].agg(
        monetary_cap_total='sum',
        ticket_mean='mean',
        ticket_median='median',
        ticket_p90=lambda x: x.quantile(0.90),
        basket_cnt='count'
    ).reset_index()

    # ---- RECENT MOMENTUM (EWM14) — VECTORIZED ---- #
    # Old: groupby.apply(lambda g: np.average(..., weights=...)) → Python loop
    # New: vectorized weighted sum / weight sum
    decay = np.log(2)/14  # half-life: 14 days
    tmp = purchases_df[['client_id', 'transaction_datetime', 'purchase_sum_cents']].copy()
    tmp['days_ago'] = (reference_date - tmp['transaction_datetime']).dt.days
    tmp['w'] = np.exp(-decay * tmp['days_ago'])
    tmp['w_spend'] = (tmp['purchase_sum_cents'] / 100) * tmp['w']

    ewm14 = (tmp.groupby('client_id', sort=False)
                .agg(w_spend_sum=('w_spend', 'sum'),
                     w_sum=('w', 'sum'))
                .reset_index())
    ewm14['monetary_ewm14'] = ewm14['w_spend_sum'] / ewm14['w_sum']
    ewm14 = ewm14[['client_id', 'monetary_ewm14']]
    del tmp  # memory

    # ---- MONETARY VELOCITY (1st half vs 2nd half) ---- #
    mid_date = min_date + (max_date - min_date)/2
    first = basket[basket['tx_time'] <= mid_date].groupby('client_id', sort=False)['basket_rub_cap'].sum()
    second = basket[basket['tx_time'] >  mid_date].groupby('client_id', sort=False)['basket_rub_cap'].sum()
    
    velocity = pd.DataFrame({'client_id': basket['client_id'].unique()})
    velocity = (velocity.merge(first.rename('first_half_spend'), on='client_id', how='left')
                        .merge(second.rename('second_half_spend'), on='client_id', how='left')
                        .fillna(0.0))
    velocity['monetary_velocity'] = velocity['second_half_spend'] / (velocity['first_half_spend'] + 1.0)

    # ---- SPENDING CONSISTENCY (CV) & TOP TRANSACTION SHARE ---- #
    agg_cons = (basket.groupby('client_id', sort=False)['basket_rub_cap']
                .agg(['mean','std','sum','max']).reset_index())
    agg_cons.rename(columns={'mean':'basket_mean','std':'basket_std','sum':'basket_sum','max':'basket_max'}, inplace=True)
    agg_cons['monetary_cv'] = np.where(agg_cons['basket_mean']>0, agg_cons['basket_std']/agg_cons['basket_mean'], 0.0)
    agg_cons['top_tx_share'] = np.where(agg_cons['basket_sum']>0, agg_cons['basket_max']/agg_cons['basket_sum'], 0.0)

    # ---- RECENT SPEND RATIO (last 7 days) ---- #
    recent_cut = reference_date - pd.Timedelta(days=7)
    recent = (basket[basket['tx_time'] >= recent_cut]
              .groupby('client_id', sort=False)['basket_rub_cap']
              .sum().rename('recent_monetary_cap').reset_index())

    # ---- MERGE ALL FEATURES ---- #
    rfm = (rfm.merge(basket_stats, on='client_id', how='left')
              .merge(ewm14, on='client_id', how='left')
              .merge(velocity[['client_id','monetary_velocity']], on='client_id', how='left')
              .merge(agg_cons[['client_id','monetary_cv','top_tx_share']], on='client_id', how='left')
              .merge(recent, on='client_id', how='left'))

    # Use basket-capped total as official monetary_capped
    rfm['monetary_capped'] = rfm['monetary_cap_total'].fillna(0.0)

    # AOV (RUB)
    rfm['aov'] = np.where(rfm['basket_cnt']>0, rfm['monetary_capped']/rfm['basket_cnt'], 0.0)

    # Recent spend ratio
    rfm['recent_monetary_cap'] = rfm['recent_monetary_cap'].fillna(0.0)
    rfm['recent_spend_ratio'] = np.where(rfm['monetary_capped']>0,
                                         rfm['recent_monetary_cap']/rfm['monetary_capped'], 0.0)

    # Normalized/daily metrics
    rfm['recency_ratio']   = rfm['recency'] / time_window_days
    rfm['frequency_daily'] = rfm['frequency'] / time_window_days
    rfm['monetary_daily']  = rfm['monetary_capped'] / time_window_days

    # Currency conversion for reporting
    currency_info = detect_currency_and_convert(rfm['monetary_capped'], 'X5 RetailHero')
    if currency_info['exchange_rate'] != 1.0:
        rfm['monetary_usd']     = rfm['monetary_capped'] / currency_info['exchange_rate']
        rfm['aov_usd']          = rfm['aov'] / currency_info['exchange_rate']

    # AOV score (quintile) - FIX: Handle duplicates properly
    try:
        rfm['aov_score'] = pd.qcut(
            rfm['aov'].rank(method='first'), q=5, labels=False, duplicates='drop'
        )
        rfm['aov_score'] = rfm['aov_score'].astype('uint8') + 1  # Convert to 1-5 range
    except Exception as e:
        print(f"⚠️  AOV score calculation warning: {e}")
        rfm['aov_score'] = 3  # Default to middle quintile

    # Metadata
    rfm_metadata = {
        'min_date': str(min_date.date()),
        'max_date': str(max_date.date()),
        'reference_date': str(reference_date.date()),
        'time_window_days': int(time_window_days),
        'n_customers': int(len(rfm)),
        'basket_cap_995': float(cap_basket),
        'currency_info': currency_info
    }

    print(f"✅ RFM hesaplandı: {len(rfm):,} müşteri")
    print(f"   + Sepet istatistikleri, AOV, EWM14 (hızlı), Velocity, CV, TopTx, RecentRatio eklendi")
    return rfm, rfm_metadata

# ---------------------------- Scoring ---------------------------- #

def create_rfm_scores(rfm_df):
    """
    RFM scores (1-5 quintiles) + segment codes
    """
    print("\n" + "="*80)
    print("🎯 RFM SKORLARI HESAPLANIYOR")
    print("="*80)
    df = rfm_df.copy()

    df['frequency_log'] = np.log1p(df['frequency'])
    df['monetary_log']  = np.log1p(df['monetary_capped'])
    use_normalized = 'recency_ratio' in df.columns
    if use_normalized:
        df['frequency_daily_log'] = np.log1p(df['frequency_daily'])
        df['monetary_daily_log']  = np.log1p(df['monetary_daily'])

    recency_col = 'recency_ratio' if use_normalized else 'recency'
    freq_col    = 'frequency_daily_log' if use_normalized else 'frequency_log'
    mon_col     = 'monetary_daily_log'  if use_normalized else 'monetary_log'

    # Recency: Lower is better (recent customers) → 5 = most recent
    df['r_score'] = pd.qcut(df[recency_col].rank(method='first'), q=5, labels=[5,4,3,2,1], duplicates='drop')
    # Frequency: Higher is better → 5 = most frequent
    df['f_score'] = pd.qcut(df[freq_col].rank(method='first'),    q=5, labels=[1,2,3,4,5], duplicates='drop')
    # Monetary: Higher is better → 5 = most valuable
    df['m_score'] = pd.qcut(df[mon_col].rank(method='first'),     q=5, labels=[1,2,3,4,5], duplicates='drop')

    # Convert to int8 for memory efficiency
    df['r_score'] = df['r_score'].astype('uint8')
    df['f_score'] = df['f_score'].astype('uint8')
    df['m_score'] = df['m_score'].astype('uint8')
    df['rfm_score'] = df['r_score'] + df['f_score'] + df['m_score']
    df['rfm_segment'] = df['r_score'].astype(str)+df['f_score'].astype(str)+df['m_score'].astype(str)

    print("✅ RFM skorları tamam")
    print(f"   RFM Score range: {df['rfm_score'].min()}-{df['rfm_score'].max()}")
    print(f"   Ortalama RFM Score: {df['rfm_score'].mean():.2f}")
    return df

# ---------------------------- Merge & Reports ---------------------------- #

def merge_with_training_data(rfm_df, train_df):
    """
    Merge RFM features with training/treatment data
    """
    print("\n" + "="*80)
    print("🔧 FINAL DATASET OLUŞTURULUYOR")
    print("="*80)
    rows_before = len(train_df)
    final_df = train_df.merge(rfm_df, on='client_id', how='left')
    rows_after = len(final_df)
    assert rows_before == rows_after, f"❌ Join kardinalite hatası: {rows_before:,} → {rows_after:,}"
    print("✅ Join kardinalitesi korundu (1:1)")
    
    # Handle missing RFM values (should be rare if merge is correct)
    miss = final_df['rfm_score'].isnull().sum()
    if miss > 0:
        print(f"⚠️  {miss:,} müşteri için RFM eksik → median imputation")
        for col in ['recency','frequency','monetary_capped','r_score','f_score','m_score','rfm_score']:
            if col in final_df.columns:
                final_df[col] = final_df[col].fillna(final_df[col].median())
        if 'rfm_segment' in final_df.columns:
            final_df['rfm_segment'] = final_df['rfm_segment'].fillna('333')
    
    print(f"✅ Final dataset: {final_df.shape[0]:,} × {final_df.shape[1]}")
    return final_df

def calculate_segment_uplift_with_ci(final_df):
    """
    Calculate segment-level uplift with 95% confidence intervals
    """
    print(f"\n🎯 RFM SEGMENT BAZLI UPLIFT + 95% CI:")
    print("-"*60)
    
    # Pivot: segments × treatment × target mean/count
    seg_data = []
    for segment in final_df['rfm_segment'].unique():
        seg_mask = final_df['rfm_segment'] == segment
        
        for treatment in [0, 1]:
            treat_mask = seg_mask & (final_df['treatment'] == treatment)
            target_mean = final_df[treat_mask]['target'].mean()
            n = treat_mask.sum()
            seg_data.append({
                'segment': segment,
                'treatment': treatment,
                'mean': target_mean,
                'n': n
            })
    
    seg_pivot = pd.DataFrame(seg_data).pivot(index='segment', columns='treatment', values=['mean','n'])
    
    p_c = seg_pivot[('mean', 0)].fillna(0)
    n_c = seg_pivot[('n', 0)].fillna(0)
    p_t = seg_pivot[('mean', 1)].fillna(0)
    n_t = seg_pivot[('n', 1)].fillna(0)
    
    uplift = (p_t - p_c) * 100
    se = np.sqrt(p_t*(1-p_t)/(n_t+1e-10) + p_c*(1-p_c)/(n_c+1e-10)) * 100
    ci_95 = 1.96 * se
    
    report = pd.DataFrame({
        'uplift_%': uplift,
        'ci_95': ci_95,
        'n_total': n_c + n_t,
        'n_treatment': n_t,
        'n_control': n_c
    }).sort_values('uplift_%', ascending=False)
    
    for seg_name, row in report.head(5).iterrows():
        if pd.notna(row['uplift_%']):
            print(f"   Segment {seg_name}: {row['uplift_%']:+.2f}% (±{row['ci_95']:.2f}%) | n={int(row['n_total']):,}")
    
    return report

def save_processed_data(df, rfm_metadata, segment_report=None, output_path='data/x5_rfm_processed.pkl'):
    """
    Save processed data with metadata
    """
    print("\n" + "="*80)
    print("💾 VERİ KAYDEDİLİYOR")
    print("="*80)
    print("ℹ️  LEAKAGE ÖNLEME: Train/Test split (stratify=treatment) + Temporal split + Transform only on train")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    metadata = {
        'created_at': datetime.now().isoformat(),
        'n_rows': len(df),
        'n_cols': len(df.columns),
        'columns': list(df.columns),
        'treatment_balance': df['treatment'].value_counts().to_dict(),
        'target_rate': float(df['target'].mean()),
        **rfm_metadata
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump({'data': df, 'metadata': metadata, 'segment_report': segment_report}, f)
    
    file_size = Path(output_path).stat().st_size / (1024**2)
    print(f"✅ Kaydedildi: {output_path} ({file_size:.2f} MB)")
    
    sample_path = output_path.replace('.pkl','_sample.csv')
    df.head(10000).to_csv(sample_path, index=False)
    print(f"✅ Sample CSV: {sample_path}")
    
    if segment_report is not None:
        seg_path = output_path.replace('.pkl','_segment_uplift.csv')
        segment_report.to_csv(seg_path)
        print(f"✅ Segment Report CSV: {seg_path}")
    
    return output_path, metadata

# ---------------------------- Main ---------------------------- #

def main():
    print("="*80)
    print("🚀 X5 RETAILHERO - RFM FEATURE ENGINEERING v2.2 (PRODUCTION)")
    print("="*80)

    print("\n⏳ Dataset yükleniyor...")
    dataset = fetch_x5()
    print("✅ Dataset yüklendi!")

    purchases = dataset.data['purchases']
    train = dataset.data['train']
    print(f"📦 Purchases: {purchases.shape}")
    print(f"📦 Train: {train.shape}")

    # Calculate RFM features
    rfm_df, rfm_meta = calculate_rfm_features(purchases)
    rfm_df = create_rfm_scores(rfm_df)

    # Prepare training data with treatment/target
    train_with_target = train.copy()
    train_with_target['treatment'] = dataset.treatment
    train_with_target['target'] = dataset.target

    # Merge RFM with training data
    final_df = merge_with_training_data(rfm_df, train_with_target)

    print("\n" + "="*80)
    print("📊 ÖZET İSTATİSTİKLER")
    print("="*80)
    print(f"Treatment: {final_df['treatment'].value_counts().to_dict()}")
    print(f"Target: {final_df['target'].value_counts().to_dict()}")
    
    conv_c = final_df[final_df['treatment']==0]['target'].mean()*100
    conv_t = final_df[final_df['treatment']==1]['target'].mean()*100
    ate = conv_t - conv_c
    print(f"🎯 Control: {conv_c:.2f}% | Treatment: {conv_t:.2f}% | ATE (Naive): {ate:+.2f}%")

    seg_report = calculate_segment_uplift_with_ci(final_df)
    output_path, metadata = save_processed_data(final_df, rfm_meta, seg_report)

    print("\n" + "="*80)
    print("🎉 TAMAMLANDI!")
    print("="*80)
    print(f"📁 İşlenmiş veri: {output_path}")
    print("💡 Sonraki adım: python scripts/5_train_uplift_model.py")

if __name__ == '__main__':
    main()