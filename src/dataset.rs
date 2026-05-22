use mascot_rs::mascot_generic_format::MGFVec;
use mass_spectrometry::traits::Spectrum;
use molecular_formulas::{ChemicalFormula, MolecularFormula};

use burn::data::dataset::Dataset;

use crate::{
    data::{ELEMENT_COUNT, ELEMENTS, SpectrumSample},
    error::SpectraError,
};

#[derive(Clone, Debug)]
pub struct SpectraData {
    pub(crate) dataset: Vec<SpectrumSample>,
    pub(crate) class_weights: Vec<f32>,
    pub(crate) bin_size: usize,
}

impl SpectraData {
    pub fn new(bin_size: usize) -> Result<Self, SpectraError> {
        let data = load_spectra(bin_size)?;
        let weights = get_class_weights(&data);
        Ok(Self {
            dataset: data,
            class_weights: weights,
            bin_size,
        })
    }

    pub fn class_weights_for(&self, class_indices: &[usize]) -> Vec<f32> {
        class_indices
            .iter()
            .map(|&index| self.class_weights[index])
            .collect()
    }
    pub fn bin_size(&self) -> usize {
        self.bin_size
    }
}

fn load_spectra(bin_size: usize) -> Result<Vec<SpectrumSample>, SpectraError> {
    let load = pollster::block_on(
        MGFVec::<f64>::annotated_ms2()
            .target_directory("data")
            .load(),
    )?;
    let mut output: Vec<SpectrumSample> = Vec::with_capacity(load.spectra().len());
    for spec in load.spectra() {
        let Some(formula) = spec.metadata().formula() else {
            continue;
        };
        let spectra = spec
            .linear_binned_intensities(0.0, 1000.0, bin_size)?
            .to_vec();
        output.push(SpectrumSample {
            spectra,
            element_present: *spec_occurrence(formula)
                .as_array::<ELEMENT_COUNT>()
                .ok_or(SpectraError::InvalidArray)?,
        });
    }
    Ok(output)
}

fn spec_occurrence(formula: &ChemicalFormula<u32, i32>) -> [bool; ELEMENT_COUNT] {
    let mut elements_occurrence = [false; ELEMENT_COUNT];
    for (i, &e) in ELEMENTS.iter().enumerate() {
        if formula.contains_element(e) {
            elements_occurrence[i] = true;
        }
    }
    elements_occurrence
}

fn get_class_weights(data: &[SpectrumSample]) -> Vec<f32> {
    let mut output: Vec<f32> = vec![0.0; ELEMENT_COUNT];
    let n_samples = data.len() as f32;
    let n_classes = ELEMENT_COUNT as f32;
    for d in data {
        for (i, &element_present) in d.element_present.iter().enumerate() {
            if element_present {
                output[i] += 1.0;
            }
        }
    }
    for weight in &mut output {
        if *weight == 0.0 {
            *weight = 1e-3;
        }
        *weight = n_samples / (*weight * n_classes);
    }
    output
}

impl Dataset<SpectrumSample> for SpectraData {
    fn get(&self, index: usize) -> Option<SpectrumSample> {
        self.dataset.get(index).cloned()
    }
    fn len(&self) -> usize {
        self.dataset.len()
    }
}
