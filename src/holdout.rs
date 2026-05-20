use elements_rs::Element;

use crate::{data::SpectrumSample, holdout};

/// Defines the methods for a single holdout
pub trait Holdout {
    /// Returns a slice of [`Element`] that constitute the class
    fn classes(&self) -> &[Element];
    /// The iteration of the holdout
    fn holdout_number(&self) -> usize;
    /// the value of the random seed for the holdout
    fn random_seed(&self) -> usize;
    /// Returns a tuple of slices of the training and validation [`SpectrumSample`]
    fn split(&self) -> (&[SpectrumSample], &[SpectrumSample]);
    /// the total spectra that are in the holdout's training set
    fn training_len(&self) -> usize {
        self.split().0.len()
    }
    /// the total spectra in the holdout's validation set
    fn validation_len(&self) -> usize {
        self.split().1.len()
    }
}


pub struct RandomHoldout {
    train: Vec<SpectrumSample>,
    validation: Vec<SpectrumSample>,
    seed: u64,
    number: usize,
}

impl Holdout for RandomHoldout {
    fn classes(&self) -> &[Element] {
        todo!()
    }

    fn holdout_number(&self) -> usize {
        todo!()
    }

    fn random_seed(&self) -> usize {
        todo!()
    }

    fn split(&self) -> (&[SpectrumSample], &[SpectrumSample]) {
        todo!()
    }
}