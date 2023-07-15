from src.utils.basic.io import makedirs, Print, load_checkpoint
from copy import deepcopy
import os

def training_ensemble_models(base_save_dir,
                             model_class,
                             config,
                             train_dataloader,
                             unshuffle_train_dataloader=None,
                             valid_dataloader=None,
                             test_dataloader=None,
                             ensemble_num=10,
                             scaler_targets=False):

    makedirs(base_save_dir)
    for i in range(ensemble_num):
        Print(f"Training {i}th model.")
        save_dir = os.path.join(base_save_dir, f"model_{i}")
        single_config = deepcopy(config)
        single_model = model_class(single_config, save_dir=save_dir)

        if scaler_targets is True:
            train_mean, train_std = train_dataloader.scaler_property
            single_model.change_train_mean(train_mean)
            single_model.change_train_std(train_std)

        # fit_on_dataloader() == model.train()
        single_model.fit_on_dataloader(train_dataloader, valid_dataloader)
        final_model = load_checkpoint(os.path.join(save_dir, "trained_model.config"), save_dir=save_dir)

        if valid_dataloader is not None:
            final_model.report_on_dataloader(valid_dataloader, "valid_prediction.csv")

        if unshuffle_train_dataloader is not None:
            final_model.report_on_dataloader(unshuffle_train_dataloader, "train_prediction.csv")

        if test_dataloader is not None:
            final_model.report_on_dataloader(test_dataloader, "test_prediction.csv")




