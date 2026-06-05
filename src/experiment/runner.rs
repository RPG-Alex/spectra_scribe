use std::{fs, fs::File, path::PathBuf};

use burn::backend::{Autodiff, Wgpu};
use csv::Writer;
use serde::Serialize;

use crate::{
    dataset::SpectraData,
    error::SpectraError,
    evaluation::{aggregate_metrics, create_confusion_matrices, element_metrics_from_matrices},
    experiment::{
        config::{ClassWeighting, ExperimentConfig},
        protocol::ExperimentProtocol,
    },
    holdout::{Holdout, class_distribution_report},
    model::ModelConfig,
    training::TrainingConfig,
};

pub fn run_experiment<P>(config: ExperimentConfig<P>) -> Result<(), SpectraError>
where
    P: ExperimentProtocol,
{
    type MyBackend = Wgpu<f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;
    println!("Loading spectra...");
    let dataset = SpectraData::new(config.features.bin_size)?;
    println!("Spectra loaded!");

    let holdouts = config.protocol.generate_holdouts(&dataset);
    let device = burn::backend::wgpu::WgpuDevice::default();

    let experiment_dir = format!("./experiments/experiment{}", config.run.experiment_num);
    let results_dir =
        PathBuf::from("./results").join(format!("experiment{}", config.run.experiment_num));

    fs::create_dir_all(&experiment_dir)?;
    fs::create_dir_all(&results_dir)?;

    for holdout in holdouts {
        debug_assert_eq!(holdout.num_classes(), holdout.class_indices().len());

        println!(
            "Running holdout {} with seed {}.",
            holdout.holdout_number(),
            holdout.random_seed(),
        );

        println!(
            "Training samples: {}, validation samples: {}.",
            holdout.training_len(),
            holdout.validation_len(),
        );

        let distribution = class_distribution_report(&holdout);
        write_csv(
            results_dir.join(format!(
                "holdout_{}_class_distribution.csv",
                holdout.holdout_number()
            )),
            &distribution,
        )?;

        let artifact_dir = format!("{experiment_dir}/holdout_{}", holdout.holdout_number());

        let class_weights = match config.loss.class_weighting {
            ClassWeighting::None => None,
            ClassWeighting::InverseFrequency { clamp } => Some(
                holdout
                    .train_dataset()
                    .class_weights_for(holdout.class_indices(), clamp),
            ),
        };

        let model_config = ModelConfig::new(
            holdout.num_classes(),
            config.model.hidden_size,
            config.features.bin_size,
            config.model.dropout,
        )
        .with_class_weights(class_weights);

        let training_config = TrainingConfig::new_with_values(
            model_config,
            config.training.epochs,
            config.training.batch_size,
            config.training.workers,
            holdout.random_seed() as u64,
            config.training.learning_rate,
            holdout.class_indices().to_vec(),
        );

        crate::training::train_holdout::<MyAutodiffBackend, _>(
            &artifact_dir,
            &holdout,
            training_config,
            device.clone(),
        );

        let validation_items = holdout.validation_dataset().samples().to_vec();
        let predictions =
            crate::inference::infer::<MyBackend>(&artifact_dir, &device, validation_items.clone());

        for threshold in &config.evaluation.thresholds {
            let confusion_matrices = create_confusion_matrices(
                predictions.clone(),
                &validation_items,
                holdout.class_indices(),
                *threshold,
            )?;

            let threshold_name = format_threshold(*threshold);
            write_csv(
                results_dir.join(format!(
                    "holdout_{}_threshold_{}_element_metrics.csv",
                    holdout.holdout_number(),
                    threshold_name
                )),
                &confusion_matrices,
            )?;

            let element_metrics = element_metrics_from_matrices(&confusion_matrices);
            write_csv(
                results_dir.join(format!(
                    "holdout_{}_threshold_{}_element_metrics.csv",
                    holdout.holdout_number(),
                    threshold_name
                )),
                &element_metrics,
            )?;

            let aggregate = aggregate_metrics(&element_metrics);
            write_csv(
                results_dir.join(format!(
                    "holdout_{}_threshold_{}aggregate_metrics.csv",
                    holdout.holdout_number(),
                    threshold_name
                )),
                &[aggregate],
            )?;
        }
    }
    Ok(())
}

fn write_csv<T: Serialize>(path: PathBuf, rows: &[T]) -> Result<(), SpectraError> {
    let file = File::create(path)?;
    let mut writer = Writer::from_writer(file);

    for row in rows {
        writer.serialize(row)?;
    }
    writer.flush()?;
    Ok(())
}

fn format_threshold(threshold: f64) -> String {
    format!("{threshold:.2}").replace('.', "_")
}
