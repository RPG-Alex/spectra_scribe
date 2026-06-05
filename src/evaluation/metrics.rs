use crate::evaluation::confusion::ConfusionMatrix;
use serde::Serialize;

#[derive(Debug, Serialize)]
pub struct ElementMetrics {
    pub element: String,
    pub true_positive: u32,
    pub true_negative: u32,
    pub false_positive: u32,
    pub false_negative: u32,
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
    pub mcc: f64,
}

#[derive(Debug, Serialize)]
pub struct AggregateMetrics {
    pub macro_precision: f64,
    pub macro_recall: f64,
    pub macro_f1: f64,
    pub macro_mcc: f64,
    pub micro_precision: f64,
    pub micro_recall: f64,
    pub micro_f1: f64,
    pub micro_mcc: f64,
}

pub fn element_metrics_from_matrices(matrices: &[ConfusionMatrix]) -> Vec<ElementMetrics> {
    matrices
        .iter()
        .map(|matrix| {
            let tp = f64::from(matrix.true_positive);
            let tn = f64::from(matrix.true_negative);
            let fp = f64::from(matrix.false_positive);
            let fne = f64::from(matrix.false_negative);

            let precision = safe_div(tp, tp + fp);
            let recall = safe_div(tp, tp + fne);
            let f1 = safe_div(2.0 * precision * recall, precision + recall);
            let mcc = mcc(tp, tn, fp, fne);

            ElementMetrics {
                element: matrix.element.clone(),
                true_positive: matrix.true_positive,
                true_negative: matrix.true_negative,
                false_positive: matrix.false_positive,
                false_negative: matrix.false_negative,
                precision,
                recall,
                f1,
                mcc,
            }
        })
        .collect()
}

pub fn aggregate_metrics(metrics: &[ElementMetrics]) -> AggregateMetrics {
    let n = metrics.len() as f64;

    let macro_precision = metrics.iter().map(|m| m.precision).sum::<f64>() / n;
    let macro_recall = metrics.iter().map(|m| m.recall).sum::<f64>() / n;
    let macro_f1 = metrics.iter().map(|m| m.f1).sum::<f64>() / n;
    let macro_mcc = metrics.iter().map(|m| m.mcc).sum::<f64>() / n;

    let tp = metrics.iter().map(|m| f64::from(m.true_positive)).sum::<f64>();
    let tn = metrics.iter().map(|m| f64::from(m.true_negative)).sum::<f64>();
    let fp = metrics.iter().map(|m| f64::from(m.false_positive)).sum::<f64>();
    let fne = metrics.iter().map(|m| f64::from(m.false_negative)).sum::<f64>();

    let micro_precision = safe_div(tp, tp + fp);
    let micro_recall = safe_div(tp, tp + fne);
    let micro_f1 = safe_div(
        2.0 * micro_precision * micro_recall,
        micro_precision + micro_recall,
    );
    let micro_mcc = mcc(tp, tn, fp, fne);

    AggregateMetrics {
        macro_precision,
        macro_recall,
        macro_f1,
        macro_mcc,
        micro_precision,
        micro_recall,
        micro_f1,
        micro_mcc,
    }
}

fn safe_div(numerator: f64, denominator: f64) -> f64 {
    if denominator.abs() < f64::EPSILON {
        0.0
    } else {
        numerator / denominator
    }
}

fn mcc(tp: f64, tn: f64, fp: f64, fne: f64) -> f64 {
    let numerator = fp.mul_add(-fne, tp *tn);
    let denominator = ((tp + fp) * (tp + fne) * (tn + fp) * (tn + fne)).sqrt();
    safe_div(numerator, denominator)
}
