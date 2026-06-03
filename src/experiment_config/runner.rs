use std::{
    fs::{self, File},
    path::PathBuf,
};

use burn::backend::{Autodiff, Wgpu};
use csv::Writer;

use crate::{
    dataset::SpectraData, error::SpectraError, experiment_config::{ExperimentProtocol, experiment_run::ExperimentRunConfig},
    holdout::Holdout, inference::create_confusion_matrices, model::ModelConfig,
    training::TrainingConfig,
};

pub fn run_experiment<P>(config: ExperimentRunConfig<P>) -> Result<(), SpectraError>
where
    P: ExperimentProtocol,
{
    
    type MyBackend = Wgpu<f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    println!("Loading spectra.");
    let dataset = SpectraData::new(config.features.bin_size)?;
    println!("Finished loading spectra.");

    let holdouts = config.protocol.generate_holdouts(&dataset);

    let device = burn::backend::wgpu::WgpuDevice::default();

    let experiment_dir = format!(
        "./experiments/experiment{}",
        config.run.experiment_num
    );

    let results_dir = "./results";

    fs::create_dir_all(&experiment_dir)?;
    fs::create_dir_all(results_dir)?;

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

        let artifact_dir = format!("{experiment_dir}/holdout_{}", holdout.holdout_number(),);
        let class_weights: Option<Vec<f32>> = config.loss.weight_range.map(|weights| {
            holdout
                .train_dataset()
                .class_weights_for(holdout.class_indices(), weights)
        });

        let model_config = ModelConfig::new(
            holdout.num_classes(),
            config.model.hidden_size,
            config.features.bin_size,
            config.model.dropout,
        )
        .with_class_weights(class_weights);

        let training_config = TrainingConfig::new_with_values(
            model_config,
            config.optimizer.epochs,
            config.optimizer.batch_size,
            config.optimizer.workers,
            holdout.random_seed() as u64,
            config.optimizer.learning_rate,
            holdout.class_indices().to_vec(),
        );

        crate::training::train_holdout::<MyAutodiffBackend, _>(
            &artifact_dir,
            &holdout,
            training_config,
            device.clone(),
        );

        let validation_items = holdout.validation_dataset().dataset.clone();

        let predictions =
            crate::inference::infer::<MyBackend>(&artifact_dir, &device, validation_items.clone());

        let confusion_matrices = create_confusion_matrices(
            predictions,
            &validation_items,
            holdout.class_indices(),
            config.evaluation.threshold,
        )?;

        let results_path = PathBuf::from(results_dir).join(format!(
            "experiment{}_holdout_{}_results.csv",
            config.run.experiment_num,
            holdout.holdout_number(),
        ));

        let file = File::create(results_path)?;
        let mut writer = Writer::from_writer(file);

        for matrix in confusion_matrices {
            writer.serialize(matrix)?;
        }

        writer.flush()?;
    }

    Ok(())
}
