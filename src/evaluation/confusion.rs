use serde::{Deserialize, Serialize};

use crate::{
    data::{ELEMENTS, SpectrumSample},
    error::SpectraError,
};

use burn::{prelude::*, tensor::Transaction};

#[derive(Serialize, Deserialize, Debug)]
pub struct ConfusionMatrix {
    pub element: String,
    pub true_positive: u32,
    pub true_negative: u32,
    pub false_positive: u32,
    pub false_negative: u32,
}

impl ConfusionMatrix {
    pub fn new(element:String) -> Self {
        Self {
            element,
            true_positive: 0,
            true_negative: 0,
            false_positive: 0,
            false_negative: 0
        }
    }
}

pub fn create_confusion_matrices<B: Backend>(
    predictions: Tensor<B, 2>,
    items: &[SpectrumSample],
    class_indices: &[usize],
    threshold: f64,
) -> Result<Vec<ConfusionMatrix>,SpectraError> {
    let mut confusion_matrices: Vec<ConfusionMatrix> = class_indices.iter().map(|&class_index| ConfusionMatrix::new(ELEMENTS[class_index].symbol().to_string())).collect();

    let predicted_tensor = predictions.greater_elem(threshold).int();
    let pred_iter = predicted_tensor.iter_dim(0);

    for (prediction_row, sample) in pred_iter.into_iter().zip(items.iter()) {
        let [output_data] = Transaction::default().register(prediction_row).execute().try_into().expect("Correct amount of the tensor data");
        let predicted_values = output_data.as_slice::<i32>()?;
        for (prediction_column, &class_index) in class_indices.iter().enumerate() {
            let predicted_atom = predicted_values[prediction_column];
            let true_atom = sample.element_present[class_index];
            match (predicted_atom, true_atom) {
                (1, true) => confusion_matrices[prediction_column].true_positive += 1,
                (0, false) => confusion_matrices[prediction_column].true_negative += 1,
                (1, false) => confusion_matrices[prediction_column].false_positive += 1,
                (0, true) => confusion_matrices[prediction_column].false_negative += 1,
                (i32::MIN..=-1_i32 | 2_i32..=i32::MAX, _) => {
                    unreachable!("Values outside of o and 1 should be unreachable")
                }
            }
        }
    }
    Ok(confusion_matrices)
}

