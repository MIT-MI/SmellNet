from models import *
from load_data import *
from train import *
from evaluate import *
import logging
import os
import time
from dataset import FusionDataset
from torch.utils.data import DataLoader, TensorDataset
import torch

log_dir = "/home/dewei/workspace/SmellNet/logs"

log_file_path = os.path.join(log_dir, f"translation_{time.time()}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler(),
    ],
)


def main(dropout=False, noisy=False):
    # set up logging
    logger = logging.getLogger()

    training_path = "/home/dewei/workspace/SmellNet/training"
    testing_path = "/home/dewei/workspace/SmellNet/testing"
    real_time_testing_path = "/home/dewei/workspace/SmellNet/real_time_testing_spice"
    gcms_path = "/home/dewei/workspace/SmellNet/processed_full_gcms_dataframe.csv"

    # for category in ["Nuts", "Spices", "Herbs", "Fruits", "Vegetables"]:
    #     logger.info(category)
    training_data, testing_data, real_time_testing_data, min_len = load_sensor_data(
        training_path, testing_path, real_time_testing_path=real_time_testing_path
    )

    gcms_scaled, y_encoded, le, scaler = load_gcms_data(gcms_path)

    training_data, training_label, _ = process_data_regular(training_data, le=le)
    testing_data, testing_label, _ = process_data_regular(testing_data, le=le)
    real_testing_data, real_testing_label, _ = process_data_regular(
        real_time_testing_data, le=le
    )

    training_pair_data, _ = create_pair_data(
        training_data, training_label, gcms_scaled, le, fusion=True
    )

    sensor_model = Encoder(input_dim=12, output_dim=32)

    model = TranslationModel(
        smell_encoder=sensor_model,
        gcms_dim=17,
        num_classes=len(le.classes_),
    )

    batch_size = 32
    num_epochs = 64

    training_fusion_dataset = FusionDataset(training_pair_data, le=le)

    data_loader = DataLoader(training_fusion_dataset, batch_size=batch_size)

    fusion_train(
        data_loader,
        model,
        logger,
        epochs=num_epochs,
        feature_dropout_fn=dropout,
        noisy=noisy,
        translation=True,
    )
    return model

    # torch.save(model.state_dict(), 'saved_models/translation/dropout_model_weights.pth')

    model.load_state_dict(
        torch.load("saved_models/translation/regular_model_weights.pth")
    )

    dataset = TensorDataset(torch.tensor(testing_data), torch.tensor(testing_label))
    data_loader = DataLoader(dataset, batch_size=batch_size)

    regular_evaluate(model, data_loader, le, logger, translation=True)
    regular_evaluate_top5(model, data_loader, le, logger, translation=True)


def main_evaluate(model):
    # set up logging
    logger = logging.getLogger()

    training_path = "/home/dewei/workspace/SmellNet/training"
    testing_path = "/home/dewei/workspace/SmellNet/testing"
    real_time_testing_path = "/home/dewei/workspace/SmellNet/real_time_testing_nut"
    gcms_path = "/home/dewei/workspace/SmellNet/processed_full_gcms_dataframe.csv"

    training_data, testing_data, real_time_testing_data, min_len = load_sensor_data(
        training_path, testing_path, real_time_testing_path=real_time_testing_path
    )

    gcms_scaled, y_encoded, le, scaler = load_gcms_data(gcms_path)

    testing_data, testing_label, _ = process_data_regular(testing_data, le)

    real_testing_data, real_testing_label, _ = process_data_regular(
        real_time_testing_data, le
    )

    batch_size = 32

    dataset = TensorDataset(torch.tensor(testing_data), torch.tensor(testing_label))
    data_loader = DataLoader(dataset, batch_size=batch_size)

    regular_evaluate(model, data_loader, le, logger, translation=True)
    regular_evaluate_top5(model, data_loader, le, logger, translation=True)

    dataset = TensorDataset(
        torch.tensor(real_testing_data), torch.tensor(real_testing_label)
    )
    data_loader = DataLoader(dataset, batch_size=batch_size)

    regular_evaluate(model, data_loader, le, logger, translation=True)
    regular_evaluate_top5(model, data_loader, le, logger, translation=True)

    real_time_testing_path = "/home/dewei/workspace/SmellNet/real_time_testing_spice"

    training_data, testing_data, real_time_testing_data, min_len = load_sensor_data(
        training_path, testing_path, real_time_testing_path=real_time_testing_path
    )

    real_testing_data, real_testing_label, _ = process_data_regular(
        real_time_testing_data, le
    )

    dataset = TensorDataset(
        torch.tensor(real_testing_data), torch.tensor(real_testing_label)
    )
    data_loader = DataLoader(dataset, batch_size=batch_size)

    regular_evaluate(model, data_loader, le, logger, translation=True)
    regular_evaluate_top5(model, data_loader, le, logger, translation=True)

    for category in ["Nuts", "Spices", "Herbs", "Fruits", "Vegetables"]:
        logger.info(category)
        training_data, testing_data, real_time_testing_data, min_len = load_sensor_data(
            training_path,
            testing_path,
            real_time_testing_path=real_time_testing_path,
            categories=[category],
        )

        gcms_scaled, y_encoded, le, scaler = load_gcms_data(gcms_path)

        testing_data, testing_label, _ = process_data_regular(testing_data, le)

        batch_size = 32
        epochs = 64

        dataset = TensorDataset(torch.tensor(testing_data), torch.tensor(testing_label))
        data_loader = DataLoader(dataset, batch_size=batch_size)

        regular_evaluate(model, data_loader, le, logger, translation=True)
        regular_evaluate_top5(model, data_loader, le, logger, translation=True)


def run_experiment(name, runs, **kwargs):
    logger = logging.getLogger()
    logger.info(
        f"------------------------------------{name}-------------------------------------------"
    )
    for run_id in range(runs):
        logger.info(f"[{name} Run {run_id+1}] Starting")
        start_time = time.time()
        model = main(**kwargs)
        end_time = time.time() - start_time
        logger.info(f"[{name} Run {run_id+1}] Training time: {end_time:.2f}s")
        main_evaluate(model)


if __name__ == "__main__":
    runs = 10
    run_experiment("Regular", runs)
    run_experiment("Dropout", runs, dropout=True)
    run_experiment("Noisy", runs, noisy=True)
