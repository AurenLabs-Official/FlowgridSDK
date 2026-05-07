//! Classical ML helpers for Flowgrid pipelines.
//!
//! This crate intentionally stays lightweight and dependency-minimal so it can be embedded
//! in CLI/eval flows as a baseline path beside the LLM stack.

#![allow(missing_docs)]

use serde::{Deserialize, Serialize};

#[derive(Debug, thiserror::Error)]
pub enum MlError {
    #[error("empty input vectors")]
    Empty,
    #[error("mismatched lengths: x={x_len}, y={y_len}")]
    MismatchedLen { x_len: usize, y_len: usize },
    #[error("input variance is zero")]
    ZeroVariance,
}

pub type MlResult<T> = Result<T, MlError>;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct RegressionMetrics {
    pub mae: f64,
    pub mse: f64,
    pub rmse: f64,
    pub r2: f64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct LinearModel {
    pub slope: f64,
    pub intercept: f64,
}

impl LinearModel {
    pub fn predict(&self, x: f64) -> f64 {
        self.slope * x + self.intercept
    }

    pub fn predict_batch(&self, x: &[f64]) -> Vec<f64> {
        x.iter().map(|&v| self.predict(v)).collect()
    }
}

pub fn fit_linear_regression(x: &[f64], y: &[f64]) -> MlResult<LinearModel> {
    if x.is_empty() || y.is_empty() {
        return Err(MlError::Empty);
    }
    if x.len() != y.len() {
        return Err(MlError::MismatchedLen {
            x_len: x.len(),
            y_len: y.len(),
        });
    }
    let n = x.len() as f64;
    let x_mean = x.iter().sum::<f64>() / n;
    let y_mean = y.iter().sum::<f64>() / n;
    let mut num = 0.0_f64;
    let mut den = 0.0_f64;
    for (&xi, &yi) in x.iter().zip(y.iter()) {
        let dx = xi - x_mean;
        num += dx * (yi - y_mean);
        den += dx * dx;
    }
    if den <= f64::EPSILON {
        return Err(MlError::ZeroVariance);
    }
    let slope = num / den;
    let intercept = y_mean - slope * x_mean;
    Ok(LinearModel { slope, intercept })
}

pub fn regression_metrics(y_true: &[f64], y_pred: &[f64]) -> MlResult<RegressionMetrics> {
    if y_true.is_empty() || y_pred.is_empty() {
        return Err(MlError::Empty);
    }
    if y_true.len() != y_pred.len() {
        return Err(MlError::MismatchedLen {
            x_len: y_true.len(),
            y_len: y_pred.len(),
        });
    }
    let n = y_true.len() as f64;
    let y_mean = y_true.iter().sum::<f64>() / n;
    let mut abs_sum = 0.0_f64;
    let mut sq_sum = 0.0_f64;
    let mut ss_tot = 0.0_f64;
    for (&t, &p) in y_true.iter().zip(y_pred.iter()) {
        let err = t - p;
        abs_sum += err.abs();
        sq_sum += err * err;
        let d = t - y_mean;
        ss_tot += d * d;
    }
    let mse = sq_sum / n;
    let r2 = if ss_tot <= f64::EPSILON {
        0.0
    } else {
        1.0 - (sq_sum / ss_tot)
    };
    Ok(RegressionMetrics {
        mae: abs_sum / n,
        mse,
        rmse: mse.sqrt(),
        r2,
    })
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct ClassificationMetrics {
    pub accuracy: f64,
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
}

/// Binary classification metrics (`0/1` labels).
pub fn binary_classification_metrics(y_true: &[u8], y_pred: &[u8]) -> MlResult<ClassificationMetrics> {
    if y_true.is_empty() || y_pred.is_empty() {
        return Err(MlError::Empty);
    }
    if y_true.len() != y_pred.len() {
        return Err(MlError::MismatchedLen {
            x_len: y_true.len(),
            y_len: y_pred.len(),
        });
    }
    let mut tp = 0.0_f64;
    let mut tn = 0.0_f64;
    let mut fp = 0.0_f64;
    let mut fn_ = 0.0_f64;
    for (&t, &p) in y_true.iter().zip(y_pred.iter()) {
        match (t, p) {
            (1, 1) => tp += 1.0,
            (0, 0) => tn += 1.0,
            (0, 1) => fp += 1.0,
            (1, 0) => fn_ += 1.0,
            _ => {}
        }
    }
    let n = y_true.len() as f64;
    let accuracy = (tp + tn) / n.max(1.0);
    let precision = if tp + fp <= f64::EPSILON { 0.0 } else { tp / (tp + fp) };
    let recall = if tp + fn_ <= f64::EPSILON { 0.0 } else { tp / (tp + fn_) };
    let f1 = if precision + recall <= f64::EPSILON {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    };
    Ok(ClassificationMetrics {
        accuracy,
        precision,
        recall,
        f1,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_fit_and_metrics_work() {
        let x = [0.0, 1.0, 2.0, 3.0];
        let y = [1.0, 3.0, 5.0, 7.0];
        let model = fit_linear_regression(&x, &y).unwrap();
        assert!((model.slope - 2.0).abs() < 1e-9);
        assert!((model.intercept - 1.0).abs() < 1e-9);
        let pred = model.predict_batch(&x);
        let m = regression_metrics(&y, &pred).unwrap();
        assert!(m.rmse < 1e-9);
        assert!(m.r2 > 0.999999);
    }

    #[test]
    fn binary_metrics_work() {
        let y_true = [1, 0, 1, 0, 1];
        let y_pred = [1, 0, 0, 0, 1];
        let m = binary_classification_metrics(&y_true, &y_pred).unwrap();
        assert!((m.accuracy - 0.8).abs() < 1e-9);
        assert!(m.f1 > 0.79 && m.f1 < 0.81);
    }
}
