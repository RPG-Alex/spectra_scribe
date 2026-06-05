use serde::Serialize;

use crate::{data::ELEMENTS, holdout::Holdout};

#[derive(Debug, Serialize)]
pub struct ClassDistribution {
    pub class_index: usize,
    pub element: String,
    pub train_positive: usize,
    pub validation_positive: usize,
    pub total_positive: usize,
    pub train_fraction_of_positives: f64,
    pub validation_fraction_of_positives: f64,
    pub warning: String,
}

pub fn class_distribution_report<H: Holdout>(holdout: &H) -> Vec<ClassDistribution> {
    holdout
        .class_indices()
        .iter()
        .map(|&class_index| {
            let train_positive = holdout
                .train_dataset()
                .samples()
                .iter()
                .filter(|sample| sample.element_present[class_index])
                .count();

            let validation_positive = holdout
                .validation_dataset()
                .samples()
                .iter()
                .filter(|sample| sample.element_present[class_index])
                .count();

            let total_positive = train_positive + validation_positive;
            let train_fraction_of_positives = fraction(train_positive, total_positive);
            let validation_fraction_of_positives = fraction(validation_positive, total_positive);

            let warning = match (train_positive, validation_positive, total_positive) {
                (0, _, _) => "NO_TRAIN_POSITIVES".to_string(),
                (_, 0, total) if total > 1 => "NO_VALIDATION_POSITIVES".to_string(),
                _ => String::new(),
            };
            ClassDistribution {
                class_index,
                element: ELEMENTS[class_index].symbol().to_string(),
                train_positive,
                validation_positive,
                total_positive,
                train_fraction_of_positives,
                validation_fraction_of_positives,
                warning,
            }
        })
        .collect()
}

fn fraction(part: usize, total: usize) -> f64 {
    if total == 0 {
        0.0
    } else {
        part as f64 / total as f64
    }
}
