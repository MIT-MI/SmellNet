from models import *
from load_data import *
from train import *
from evaluate import *
import logging
import os
import time
import torch

log_dir = "/home/dewei/workspace/SmellNet/logs"

log_file_path = os.path.join(log_dir, f"transformer_gradient_{time.time()}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler(),
    ],
)


def main(period_len=25):
    # set up logging
    logger = logging.getLogger()

    training_path = "/home/dewei/workspace/SmellNet/training"
    testing_path = "/home/dewei/workspace/SmellNet/testing"
    real_time_testing_path = "/home/dewei/workspace/SmellNet/real_time_testing_spice"
    gcms_path = "/home/dewei/workspace/SmellNet/processed_full_gcms_dataframe.csv"

    training_data, testing_data, real_time_testing_data, min_len = load_sensor_data(
        training_path, testing_path, real_time_testing_path=real_time_testing_path
    )

    gcms_scaled, y_encoded, le, scaler = load_gcms_data(gcms_path)

    train_data, train_label, _ = prepare_data_transformer_gradient(
        training_data, le=le, period_len=period_len, apply_fft=True
    )
    testing_data, testing_label, _ = prepare_data_transformer_gradient(
        testing_data, le=le, period_len=period_len, apply_fft=True
    )
    real_testing_data, real_testing_label, _ = prepare_data_transformer_gradient(
        real_time_testing_data, le=le, period_len=period_len, apply_fft=True
    )

    batch_size = 32
    num_epochs = 128

    dataset = TensorDataset(torch.tensor(train_data), torch.tensor(train_label))
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = TimeSeriesTransformer(
        input_dim=12, model_dim=64, num_classes=len(le.classes_)
    )

    train(data_loader, model, logger, epochs=num_epochs)
    return model

    # # torch.save(model.state_dict(), f'saved_models/transformer/gradient_period_{period_len}_model_weights.pth')
    # dataset = TensorDataset(torch.tensor(testing_data), torch.tensor(testing_label))
    # data_loader = DataLoader(dataset, batch_size=batch_size)
    # model.load_state_dict(torch.load(f'saved_models/transformer/gradient_period_{period_len}_model_weights.pth'))
    # regular_evaluate(model, data_loader, le, logger)
    # regular_evaluate_top5(model, data_loader, le, logger)


def main_evaluate(model, period_len=25):
    # set up logging
    logger = logging.getLogger()

    training_path = "/home/dewei/workspace/SmellNet/training"
    testing_path = "/home/dewei/workspace/SmellNet/testing"
    real_time_testing_path = "/home/dewei/workspace/SmellNet/real_time_testing_nut"
    gcms_path = "/home/dewei/workspace/SmellNet/processed_full_gcms_dataframe.csv"

    batch_size = 32

    training_data, testing_data, real_time_testing_data, min_len = load_sensor_data(
        training_path, testing_path, real_time_testing_path=real_time_testing_path
    )

    gcms_scaled, y_encoded, le, scaler = load_gcms_data(gcms_path)

    testing_data, testing_label, _ = prepare_data_transformer_gradient(
        testing_data, le=le, period_len=period_len, apply_fft=True
    )
    real_testing_data, real_testing_label, _ = prepare_data_transformer_gradient(
        real_time_testing_data, le=le, period_len=period_len, apply_fft=True
    )

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

    real_testing_data, real_testing_label, _ = prepare_data_transformer_gradient(
        real_time_testing_data, le=le, period_len=period_len, apply_fft=True
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

        testing_data, testing_label, _ = prepare_data_transformer_gradient(
            testing_data, le=le, period_len=period_len, apply_fft=True
        )

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
        main_evaluate(model, **kwargs)


if __name__ == "__main__":
    logger = logging.getLogger()
    runs = 1

    run_experiment("Gradient Period 25", runs)
    run_experiment("Gradient Period 50", runs, period_len=50)
