use flowgrid_ml::{
    binary_classification_metrics, fit_linear_regression, regression_metrics, ClassificationMetrics,
    LinearModel, RegressionMetrics,
};
use serde::Serialize;
use std::env;
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Serialize)]
struct GoldenClassicalMlReport {
    kind: &'static str,
    model: LinearModel,
    regression: RegressionMetrics,
    classification: ClassificationMetrics,
}

fn output_path_from_args() -> PathBuf {
    let mut args = env::args().skip(1);
    while let Some(a) = args.next() {
        if a == "--out" {
            if let Some(path) = args.next() {
                return PathBuf::from(path);
            }
        }
    }
    PathBuf::from("target/mlops/classical_ml_report.json")
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let x = [0.0, 1.0, 2.0, 3.0, 4.0];
    let y = [1.0, 3.0, 5.0, 7.0, 9.0];
    let model = fit_linear_regression(&x, &y)?;
    let pred = model.predict_batch(&x);
    let regression = regression_metrics(&y, &pred)?;

    let y_true = [1, 0, 1, 0, 1, 0];
    let y_pred = [1, 0, 1, 0, 0, 0];
    let classification = binary_classification_metrics(&y_true, &y_pred)?;

    let report = GoldenClassicalMlReport {
        kind: "classical_ml_golden_path_v1",
        model,
        regression,
        classification,
    };
    let out = output_path_from_args();
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&out, serde_json::to_string_pretty(&report)?)?;
    println!("Wrote classical ML report: {}", out.display());
    Ok(())
}
