from src.utils.mol.dataset_splitter import predetermined_split, random_split
from src.utils.basic.io import Print
from .ensemble import training_ensemble_models
import os

def predetermined_cross_validation(base_save_dir,
                                   model_class,
                                   config,
                                   dataset,
                                   split_index_file,
                                   dataloader_class,
                                   batch_size=64,
                                   ensemble_num=10,
                                   scaler_targets=False,
                                   **other_train_arguments):

    total_datasets = predetermined_split(dataset, split_index_file)

    num_folds = len(total_datasets)
    for fold in range(num_folds):
        Print(f"Fold {fold}.")
        fold_save_dir = os.path.join(base_save_dir, f"fold_{fold}")

        train_args = {"dataset": total_datasets[fold][0],
                      "batch_size": batch_size,
                      "shuffle": True}

        train_args.update(other_train_arguments)
        train_dataloader = dataloader_class(**train_args)

        valid_args = {"dataset": total_datasets[fold][1],
                      "batch_size": batch_size,
                      "shuffle": False}
        valid_dataloader = dataloader_class(**valid_args)

        test_args = {"dataset": total_datasets[fold][2],
                     "batch_size": batch_size,
                     "shuffle": False}
        test_dataloader = dataloader_class(**test_args)

        if len(test_dataloader) == 0:
            test_dataloader = None

        training_ensemble_models(fold_save_dir,
                                 model_class,
                                 config,
                                 train_dataloader,
                                 valid_dataloader=valid_dataloader,
                                 test_dataloader=test_dataloader,
                                 ensemble_num=ensemble_num,
                                 scaler_targets=scaler_targets)

def random_cross_validation_notest(base_save_dir,
                                   model_class,
                                   config,
                                   dataset,
                                   num_folds,
                                   dataloader_class,
                                   batch_size=64,
                                   ensemble_num=10,
                                   independent_test_set=None,
                                   **other_train_arguments):

    total_datasets = random_split(dataset, sizes=(0.9, 0.1, 0.0), num_folds=num_folds)

    for fold in range(num_folds):
        Print(f"Fold {fold}.")
        fold_save_dir = os.path.join(base_save_dir, f"fold_{fold}")
        train_args = {"dataset":total_datasets[fold][0], "batch_size": batch_size, "shuffle": True}
        train_args.update(other_train_arguments)
        train_dataloader = dataloader_class(**train_args)

        unshuffle_train_args = {"dataset":total_datasets[fold][0], "batch_size": batch_size, "shuffle": False}
        unshuffle_train_dataloader = dataloader_class(**unshuffle_train_args)

        valid_args = {"dataset": total_datasets[fold][1], "batch_size": batch_size, "shuffle": False}
        valid_dataloader = dataloader_class(**valid_args)

        test_args = {"dataset": total_datasets[fold][2], "batch_size": batch_size, "shuffle": False}
        test_dataloader = dataloader_class(**test_args)

        if len(test_dataloader) == 0:
            test_dataloader = None

        if independent_test_set is not None:
            ind_args = {"dataset": independent_test_set, "batch_size": batch_size, "shuffle": False}
            ind_dataloader = dataloader_class(**ind_args)

        training_ensemble_models(fold_save_dir,
                                 model_class,
                                 config,
                                 train_dataloader,
                                 unshuffle_train_dataloader=unshuffle_train_dataloader,
                                 valid_dataloader=valid_dataloader,
                                 test_dataloader=test_dataloader,
                                 ensemble_num=ensemble_num,
                                 independent_test_dataloader=ind_dataloader)


