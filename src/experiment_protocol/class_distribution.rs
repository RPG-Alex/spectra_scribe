use crate::holdout::Holdout;

pub struct ClassDistribution {
    pub class_index: usize,
    pub train_positive: usize,
    pub validation_positive: usize,
    pub total_positive: usize,
}

impl ClassDistribution {
    pub fn new(class_index: usize, train_positive: usize, validation_positive: usize) -> Self {
        Self {
            class_index,
            train_positive,
            validation_positive,
            total_positive: train_positive + validation_positive,
        }
    }

    pub fn class_distribution_report<H: Holdout>(holdout: &H) -> Vec<Self> {
        holdout
            .class_indices()
            .iter()
            .map(|&class_index| {
                let train_positive = holdout
                    .train_dataset()
                    .dataset
                    .iter()
                    .filter(|sample| sample.element_present[class_index])
                    .count();

                let validation_positive = holdout
                    .validation_dataset()
                    .dataset
                    .iter()
                    .filter(|sample| sample.element_present[class_index])
                    .count();

                Self::new(class_index, train_positive, validation_positive)
            })
            .collect()
    }
}
