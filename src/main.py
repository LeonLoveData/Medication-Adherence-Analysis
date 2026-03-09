# src/main.py
from data_process import (
    load_synthetic_data,
    filter_diabetes_prescriptions,
    compute_adherence_metrics,
)
from analysis import (
    build_feature_table,
    encode_features,
    train_models,
)
from visualization import (
    plot_adherence_distributions,
    plot_model_curves,
)

def main():
    patients, prescriptions, stores, interventions, financial = load_synthetic_data()

    prescriptions_diab = filter_diabetes_prescriptions(prescriptions)
    adherence = compute_adherence_metrics(prescriptions_diab)

    feature_df = build_feature_table(
        patients, stores, prescriptions_diab, adherence, interventions, financial
    )

    X, y = encode_features(feature_df)
    results = train_models(X, y)

    plot_adherence_distributions(adherence)
    plot_model_curves(results)

    print("\nPipeline complete.")

if __name__ == "__main__":
    main()
