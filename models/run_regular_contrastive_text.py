from models import *
from load_data import *
from train import *
from evaluate import *
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import logging
import os
import time
from dataset import *
import torch

log_dir = "/home/dewei/workspace/SmellNet/logs"

log_file_path = os.path.join(log_dir, f"text_regular_{time.time()}.log")

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
    real_time_testing_path = "/home/dewei/workspace/SmellNet/real_time_testing_nut"
    text_path = "/home/dewei/workspace/SmellNet/clip_text_embeddings.npy"

    training_data, testing_data, real_time_testing_data, min_len = load_sensor_data(
        training_path, testing_path, real_time_testing_path=real_time_testing_path
    )

    text_scaled, y_encoded, le, scaler = load_text_data(text_path)

    training_X, training_label, _ = process_data_regular(training_data, le)

    testing_X, testing_label, _ = process_data_regular(testing_data, le)

    real_testing_X, real_testing_label, _ = process_data_regular(
        real_time_testing_data, le
    )

    training_pair_data, _ = create_pair_data(
        training_X, training_label, text_scaled, le
    )

    train_dataset = PairedDataset(training_pair_data)
    sensor_model = Encoder(input_dim=12, output_dim=32)
    text_model = Encoder(input_dim=512, output_dim=32)

    batch_size = 32
    num_epochs = 64

    sampler = UniqueGCMSampler(train_dataset.data, batch_size)
    loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)

    contrastive_train(
        text_model,
        sensor_model,
        loader,
        logger,
        num_epochs=num_epochs,
        feature_dropout_fn=dropout,
        noisy=noisy,
    )

    if not dropout and not noisy:
        torch.save(
            sensor_model.state_dict(), "saved_models/text/sensor_model_weights.pth"
        )
        torch.save(text_model.state_dict(), "saved_models/text/text_model_weights.pth")
    elif dropout:
        torch.save(
            sensor_model.state_dict(),
            "saved_models/text/dropout_sensor_model_weights.pth",
        )
        torch.save(
            text_model.state_dict(), "saved_models/text/dropout_text_model_weights.pth"
        )
    elif noisy:
        torch.save(
            sensor_model.state_dict(),
            "saved_models/text/noisy_sensor_model_weights.pth",
        )
        torch.save(
            text_model.state_dict(), "saved_models/text/noisy_text_model_weights.pth"
        )

    # sensor_model.load_state_dict(torch.load('saved_models/contrastive/regular_sensor_model_weights.pth'))
    # text_model.load_state_dict(torch.load('saved_models/contrastive/regular_text_model_weights.pth'))
    return text_model, sensor_model


def main_evaluate(text_model, sensor_model):
    # set up logging
    logger = logging.getLogger()

    training_path = "/home/dewei/workspace/SmellNet/training"
    testing_path = "/home/dewei/workspace/SmellNet/testing"
    real_time_testing_path = "/home/dewei/workspace/SmellNet/real_time_testing_nut"
    text_path = "/home/dewei/workspace/SmellNet/clip_text_embeddings.npy"

    training_data, testing_data, real_time_testing_data, min_len = load_sensor_data(
        training_path, testing_path, real_time_testing_path=real_time_testing_path
    )

    text_scaled, y_encoded, le, scaler = load_text_data(text_path)

    testing_data, testing_label, _ = process_data_regular(testing_data, le)

    real_testing_data, real_testing_label, _ = process_data_regular(
        real_time_testing_data, le
    )

    contrastive_evaluate(
        testing_data, text_scaled, testing_label, text_model, sensor_model, logger
    )

    contrastive_evaluate(
        real_testing_data,
        text_scaled,
        real_testing_label,
        text_model,
        sensor_model,
        logger,
    )

    real_time_testing_path = "/home/dewei/workspace/SmellNet/real_time_testing_spice"

    training_data, testing_data, real_time_testing_data, min_len = load_sensor_data(
        training_path, testing_path, real_time_testing_path=real_time_testing_path
    )

    real_testing_data, real_testing_label, _ = process_data_regular(
        real_time_testing_data, le
    )

    contrastive_evaluate(
        real_testing_data,
        text_scaled,
        real_testing_label,
        text_model,
        sensor_model,
        logger,
    )

    for category in ["Nuts", "Spices", "Herbs", "Fruits", "Vegetables"]:
        logger.info(category)
        training_data, testing_data, real_time_testing_data, min_len = load_sensor_data(
            training_path,
            testing_path,
            real_time_testing_path=real_time_testing_path,
            categories=[category],
        )

        text_scaled, y_encoded, le, scaler = load_text_data(text_path)

        testing_data, testing_label, _ = process_data_regular(testing_data, le)

        contrastive_evaluate(
            testing_data, text_scaled, testing_label, text_model, sensor_model, logger
        )


def run_experiment(name, runs, **kwargs):
    logger = logging.getLogger()
    logger.info(
        f"------------------------------------{name}-------------------------------------------"
    )
    for run_id in range(runs):
        logger.info(f"[{name} Run {run_id+1}] Starting")
        start_time = time.time()
        text_model, sensor_model = main(**kwargs)
        end_time = time.time() - start_time
        logger.info(f"[{name} Run {run_id+1}] Training time: {end_time:.2f}s")
        main_evaluate(text_model, sensor_model)


if __name__ == "__main__":
    runs = 1
    run_experiment("Regular", runs)
    run_experiment("Dropout", runs, dropout=True)
    run_experiment("Noisy", runs, noisy=True)
