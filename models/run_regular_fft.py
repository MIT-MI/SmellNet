from models import *
from load_data import *
from train import *
from evaluate import *
import logging
import os
import time
import torch

log_dir = "/home/dewei/workspace/SmellNet/logs"

log_file_path = os.path.join(log_dir, f"fft_{time.time()}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler(),
    ],
)


def main(dropout=False, noisy=False):
    logger = logging.getLogger()

    # === Paths ===
    training_path = "/home/dewei/workspace/SmellNet/training"
    testing_path = "/home/dewei/workspace/SmellNet/testing"
    real_time_testing_path = "/home/dewei/workspace/SmellNet/real_time_testing_nut"
    gcms_path = "/home/dewei/workspace/SmellNet/processed_full_gcms_dataframe.csv"

    # === Load raw data ===
    training_data, testing_data, real_time_testing_data, min_len = load_sensor_data(
        training_path, testing_path, real_time_testing_path=real_time_testing_path
    )
    gcms_scaled, y_encoded, le, scaler = load_gcms_data(gcms_path)

    # === FFT-enabled Transformer Preprocessing ===
    train_data, train_label, _ = prepare_data_transformer(training_data, le, apply_fft=True)
    test_data, test_label, _ = prepare_data_transformer(testing_data, le, apply_fft=True)
    real_test_data, real_test_label, _ = prepare_data_transformer(real_time_testing_data, le, apply_fft=True)

    # === Train model ===
    batch_size = 32
    num_epochs = 128

    dataset = TensorDataset(torch.tensor(train_data), torch.tensor(train_label))
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = TimeSeriesTransformer(input_dim=12, model_dim=64, num_classes=len(le.classes_))
    train(data_loader, model, logger, epochs=num_epochs, feature_dropout_fn=dropout, noisy=noisy)

    return model

    # torch.save(model.state_dict(), f'saved_models/transformer/noisy_model_weights.pth')

    dataset = TensorDataset(torch.tensor(test_data), torch.tensor(test_label))
    data_loader = DataLoader(dataset, batch_size=batch_size)
    model.load_state_dict(
        torch.load(f"saved_models/transformer/regular_model_weights.pth")
    )
    regular_evaluate(model, data_loader, le, logger)
    regular_evaluate_top5(model, data_loader, le, logger)

    # transformer_evaluate(model, real_time_testing_data, le, logger)


def main_evaluate(model):
    # set up logging
    logger = logging.getLogger()

    training_path = "/home/dewei/workspace/SmellNet/training"
    testing_path = "/home/dewei/workspace/SmellNet/testing"
    real_time_testing_path = "/home/dewei/workspace/SmellNet/real_time_testing_nut"
    gcms_path = "/home/dewei/workspace/SmellNet/processed_full_gcms_dataframe.csv"

    # model = Encoder(input_dim=12, output_dim=50)
    # model.load_state_dict(torch.load('saved_models/regular/noisy_model_weights.pth'))

    # for category in ["Nuts", "Spices", "Herbs", "Fruits", "Vegetables"]:
    #     logger.info(category)

    training_data, testing_data, real_time_testing_data, min_len = load_sensor_data(
        training_path, testing_path, real_time_testing_path=real_time_testing_path
    )

    gcms_scaled, y_encoded, le, scaler = load_gcms_data(gcms_path)

    testing_data, testing_label, _ = prepare_data_transformer(testing_data, le=le, apply_fft=True)

    real_testing_data, real_testing_label, _ = prepare_data_transformer(
        real_time_testing_data, le=le, apply_fft=True
    )

    batch_size = 32

    dataset = TensorDataset(torch.tensor(testing_data), torch.tensor(testing_label))
    data_loader = DataLoader(dataset, batch_size=batch_size)

    regular_evaluate(model, data_loader, le, logger)
    regular_evaluate_top5(model, data_loader, le, logger)

    dataset = TensorDataset(
        torch.tensor(real_testing_data), torch.tensor(real_testing_label)
    )
    data_loader = DataLoader(dataset, batch_size=batch_size)

    regular_evaluate(model, data_loader, le, logger)
    regular_evaluate_top5(model, data_loader, le, logger)

    real_time_testing_path = "/home/dewei/workspace/SmellNet/real_time_testing_spice"

    training_data, testing_data, real_time_testing_data, min_len = load_sensor_data(
        training_path, testing_path, real_time_testing_path=real_time_testing_path
    )

    real_testing_data, real_testing_label, _ = prepare_data_transformer(
        real_time_testing_data, le=le, apply_fft=True
    )

    dataset = TensorDataset(
        torch.tensor(real_testing_data), torch.tensor(real_testing_label)
    )
    data_loader = DataLoader(dataset, batch_size=batch_size)

    regular_evaluate(model, data_loader, le, logger)
    regular_evaluate_top5(model, data_loader, le, logger)

    for category in ["Nuts", "Spices", "Herbs", "Fruits", "Vegetables"]:
        logger.info(category)
        training_data, testing_data, real_time_testing_data, min_len = load_sensor_data(
            training_path,
            testing_path,
            real_time_testing_path=real_time_testing_path,
            categories=[category],
        )

        gcms_scaled, y_encoded, le, scaler = load_gcms_data(gcms_path)

        testing_data, testing_label, _ = prepare_data_transformer(testing_data, le=le, apply_fft=True)

        batch_size = 32
        epochs = 64

        dataset = TensorDataset(torch.tensor(testing_data), torch.tensor(testing_label))
        data_loader = DataLoader(dataset, batch_size=batch_size)

        regular_evaluate(model, data_loader, le, logger)
        regular_evaluate_top5(model, data_loader, le, logger)


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
    runs = 1
    run_experiment("Regular", runs)
    run_experiment("Dropout", runs, dropout=True)
    run_experiment("Noisy", runs, noisy=True)
