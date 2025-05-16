from models import *
from load_data import *
from train import *
from evaluate import *
import logging
import os
import time
import torch

log_dir = "/home/dewei/workspace/smell-net/logs"

log_file_path = os.path.join(log_dir, f"{time.time()}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler(),
    ],
)


def main():
    # set up logging
    logger = logging.getLogger()

    training_path = "/home/dewei/workspace/smell-net/training"
    testing_path = "/home/dewei/workspace/smell-net/testing"
    real_time_testing_path = "/home/dewei/workspace/smell-net/real_time_testing_nut"
    gcms_path = "/home/dewei/workspace/smell-net/processed_full_gcms_dataframe.csv"

    for category in ["Nuts", "Spices", "Herbs", "Fruits", "Vegetables"]:
        logger.info(category)

        training_data, testing_data, real_time_testing_data, min_len = load_sensor_data(
            training_path, testing_path, real_time_testing_path=real_time_testing_path, categories=[category])

        gcms_scaled, y_encoded, le, scaler = load_gcms_data(gcms_path)
        
        train_data, train_label, _ = prepare_data_transformer(training_data, le=le)
        
        test_data, test_label, _ = prepare_data_transformer(testing_data, le=le)

        real_test_data, real_test_label, _ = prepare_data_transformer(real_time_testing_data, le=le)

        batch_size = 32
        num_epochs = 128

        # dataset = TensorDataset(torch.tensor(train_data), torch.tensor(train_label))
        # data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        model = TimeSeriesTransformer(
            input_dim=12, model_dim=64, num_classes=len(le.classes_)
        )

        # train(data_loader, model, logger, epochs=num_epochs, noisy=True)

        # torch.save(model.state_dict(), f'saved_models/transformer/noisy_model_weights.pth')

        dataset = TensorDataset(torch.tensor(test_data), torch.tensor(test_label))
        data_loader = DataLoader(dataset, batch_size=batch_size)
        model.load_state_dict(torch.load(f'saved_models/transformer/regular_model_weights.pth'))
        regular_evaluate(model, data_loader, le, logger)
        regular_evaluate_top5(model, data_loader, le, logger)

        # transformer_evaluate(model, real_time_testing_data, le, logger)


if __name__ == "__main__":
    main()
