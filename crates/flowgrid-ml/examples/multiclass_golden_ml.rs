use flowgrid_ml::{multiclass_classification_metrics, MulticlassMetrics};
use serde::Serialize;
use std::env;
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Serialize)]
struct MulticlassGoldenReport {
    kind: &'static str,
    num_classes: usize,
    metrics: MulticlassMetrics,
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
    PathBuf::from("target/mlops/multiclass_classical_ml.json")
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let y_true = [0u8, 1, 2, 0, 1, 2, 0, 1];
    let y_pred = [0u8, 1, 2, 0, 0, 2, 1, 1];
    let num_classes = 3usize;
    let metrics = multiclass_classification_metrics(&y_true, &y_pred, num_classes)?;

    let report = MulticlassGoldenReport {
        kind: "multiclass_ml_golden_path_v1",
        num_classes,
        metrics,
    };
    let out = output_path_from_args();
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&out, serde_json::to_string_pretty(&report)?)?;
    println!("Wrote multiclass ML report: {}", out.display());
    Ok(())
}
