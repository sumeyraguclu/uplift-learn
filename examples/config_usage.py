"""
Configuration Module Usage Examples
"""

from src.config import Config, get_config, load_config


# ======================== EXAMPLE 1: Basic Usage ========================

def example_basic():
    """Basic configuration loading"""
    print("=" * 70)
    print("EXAMPLE 1: Basic Configuration Loading")
    print("=" * 70)
    
    # Load config
    config = Config.load()
    
    print("\nCampaign Settings:")
    print(f"  Margin: ${config.campaign.margin}")
    print(f"  Contact Cost: ${config.campaign.contact_cost}")
    print(f"  Budget: ${config.campaign.budget:,}")
    print(f"  Test Split: {config.campaign.test_split_ratio*100:.0f}%")
    
    print("\nModel Settings:")
    print(f"  Random State: {config.model.random_state}")
    print(f"  Test Size: {config.model.test_size}")
    print(f"  XGBoost Max Depth: {config.model.xgboost['max_depth']}")
    
    print("\nPaths:")
    print(f"  Data Dir: {config.paths.data_dir}")
    print(f"  RFM Data: {config.paths.rfm_data}")
    print(f"  Model File: {config.paths.tlearner_model}")


# ======================== EXAMPLE 2: Global Config ========================

def example_global():
    """Using global configuration singleton"""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Global Configuration Singleton")
    print("=" * 70)
    
    # Get global instance (loaded once, reused)
    config = get_config()
    
    print(f"\nConfig: {config}")
    print(f"\nMetrics Settings:")
    print(f"  Qini Bins: {config.metrics.qini_bins}")
    print(f"  Uplift K Values: {config.metrics.uplift_k_values}")
    print(f"  Confidence Level: {config.metrics.confidence_level}")
    
    # Access again (same instance)
    config2 = get_config()
    print(f"\nSame instance: {config is config2}")


# ======================== EXAMPLE 3: Environment Overrides ========================

def example_environments():
    """Environment-specific configuration"""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Environment-Specific Config")
    print("=" * 70)
    
    # Development environment
    print("\n[Development]")
    dev_config = Config.load(environment='development')
    print(f"  Budget: ${dev_config.campaign.budget:,}")
    print(f"  Log Level: {dev_config._raw.get('logging', {}).get('level')}")
    
    # Production environment
    print("\n[Production]")
    prod_config = Config.load(environment='production')
    print(f"  Budget: ${prod_config.campaign.budget:,}")
    print(f"  Log Level: {prod_config._raw.get('logging', {}).get('level')}")


# ======================== EXAMPLE 4: Using in Scripts ========================

def example_in_script():
    """How to use config in actual scripts"""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Using Config in Scripts")
    print("=" * 70)
    
    from src.optimize import greedy_optimizer
    import numpy as np
    
    # Load config
    config = get_config()
    
    # Use config values
    print("\nSimulating campaign optimization...")
    uplift = np.random.beta(2, 5, 1000) * 0.1
    
    result = greedy_optimizer(
        uplift=uplift,
        margin=config.campaign.margin,
        contact_cost=config.campaign.contact_cost,
        budget=config.campaign.budget
    )
    
    print(f"\nUsing config values:")
    print(f"  Margin: ${config.campaign.margin}")
    print(f"  Cost: ${config.campaign.contact_cost}")
    print(f"  Budget: ${config.campaign.budget:,}")
    print(f"\nResults:")
    print(f"  Selected: {result['n_selected']} customers")
    print(f"  ROI: {result['roi_pct']:.1f}%")


# ======================== EXAMPLE 5: Path Management ========================

def example_paths():
    """Path management and directory creation"""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Path Management")
    print("=" * 70)
    
    config = get_config()
    
    print("\nConfigured Paths:")
    print(f"  Data: {config.paths.data_dir}")
    print(f"  Results: {config.paths.results_dir}")
    print(f"  Models: {config.paths.models_dir}")
    print(f"  Exports: {config.paths.exports_dir}")
    print(f"  Plots: {config.paths.plots_dir}")
    print(f"  Logs: {config.paths.logs_dir}")
    
    # Ensure directories exist
    config.ensure_dirs()
    print("\n✓ All directories ensured to exist")
    
    # Use in file operations
    import pandas as pd
    from pathlib import Path
    
    # Example: Save results
    results = pd.DataFrame({'metric': ['qini'], 'value': [0.15]})
    output_path = Path(config.paths.results_dir) / 'example_metrics.csv'
    results.to_csv(output_path, index=False)
    print(f"\n✓ Saved to: {output_path}")


# ======================== EXAMPLE 6: Feature Configuration ========================

def example_features():
    """Feature engineering configuration"""
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Feature Engineering Config")
    print("=" * 70)
    
    config = get_config()
    
    print("\nFeature Settings:")
    print(f"  Scaling: {config.features.scaling}")
    print(f"  Handle Missing: {config.features.handle_missing}")
    print(f"\nExcluded Columns:")
    for col in config.features.exclude_columns:
        print(f"    - {col}")
    
    # Use in feature preparation
    import pandas as pd
    import numpy as np
    
    # Simulated dataframe
    df = pd.DataFrame({
        'client_id': range(100),
        'target': np.random.randint(0, 2, 100),
        'treatment': np.random.randint(0, 2, 100),
        'feature_1': np.random.randn(100),
        'feature_2': np.random.randn(100)
    })
    
    # Select features using config
    feature_cols = [
        col for col in df.columns
        if col not in config.features.exclude_columns
    ]
    
    print(f"\nSelected Features: {feature_cols}")
    print(f"Total: {len(feature_cols)} features")


# ======================== EXAMPLE 7: Plotting Configuration ========================

def example_plotting():
    """Plotting configuration"""
    print("\n" + "=" * 70)
    print("EXAMPLE 7: Plotting Configuration")
    print("=" * 70)
    
    config = get_config()
    
    print("\nPlotting Settings:")
    print(f"  DPI: {config.plotting.dpi}")
    print(f"  Figure Size: {config.plotting.figsize}")
    print(f"  Style: {config.plotting.style}")
    print(f"  Color Palette: {config.plotting.color_palette}")
    
    # Use in plotting
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Apply config
    plt.figure(figsize=config.plotting.figsize)
    
    # Generate sample plot
    x = np.linspace(0, 10, 100)
    colors = config.plotting.color_palette
    
    for i, color in enumerate(colors[:3]):
        y = np.sin(x + i)
        plt.plot(x, y, color=color, label=f'Series {i+1}', linewidth=2)
    
    plt.title('Example Plot with Config Colors')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Save with config DPI
    from pathlib import Path
    output = Path(config.paths.plots_dir) / 'config_example.png'
    plt.savefig(output, dpi=config.plotting.dpi, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Saved plot: {output}")


# ======================== EXAMPLE 8: Dot-Notation Access ========================

def example_dot_notation():
    """Access config with dot notation"""
    print("\n" + "=" * 70)
    print("EXAMPLE 8: Dot-Notation Access")
    print("=" * 70)
    
    config = get_config()
    
    print("\nDirect Access:")
    print(f"  config.campaign.margin = {config.campaign.margin}")
    
    print("\nDot-Notation String Access:")
    print(f"  config.get('campaign.margin') = {config.get('campaign.margin')}")
    print(f"  config.get('model.xgboost.max_depth') = {config.get('model.xgboost.max_depth')}")
    print(f"  config.get('paths.rfm_data') = {config.get('paths.rfm_data')}")
    
    # With default value
    print(f"\nWith Default:")
    print(f"  config.get('nonexistent.key', default=999) = {config.get('nonexistent.key', default=999)}")


# ======================== RUN ALL EXAMPLES ========================

if __name__ == '__main__':
    example_basic()
    example_global()
    example_environments()
    example_in_script()
    example_paths()
    example_features()
    example_plotting()
    example_dot_notation()
    
    print("\n" + "=" * 70)
    print("ALL CONFIG EXAMPLES COMPLETED!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  • Use get_config() for global singleton")
    print("  • Config.load(environment='prod') for env-specific")
    print("  • config.ensure_dirs() creates all directories")
    print("  • Access nested values with dot notation")
    print("  • All magic numbers centralized in config.yaml")