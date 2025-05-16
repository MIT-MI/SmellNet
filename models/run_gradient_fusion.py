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
    real_time_testing_path = "/home/dewei/workspace/smell-net/real_time_testing_spice"
    gcms_path = "/home/dewei/workspace/smell-net/processed_full_gcms_dataframe.csv"

    period_len=50

    for category in ["Nuts", "Spices", "Herbs", "Fruits", "Vegetables"]:
        logger.info(category)
        training_data, testing_data, real_time_testing_data, min_len = load_sensor_data(training_path, testing_path, real_time_testing_path=real_time_testing_path, categories=[category])

        gcms_scaled, y_encoded, le, scaler = load_gcms_data(gcms_path)

        training_data, training_label, _ = prepare_data_gradient(training_data, period_len=period_len, le=le)

        testing_data, testing_label, _ = prepare_data_gradient(testing_data, period_len=period_len, le=le)

        real_testing_data, real_testing_label, _ = prepare_data_gradient(real_time_testing_data, period_len=period_len, le=le)

        training_pair_data, _ = create_pair_data(training_data, training_label, gcms_scaled, le, fusion=True)

        sensor_model = Encoder(input_dim=12, output_dim=32)
        gcms_model = Encoder(input_dim=17, output_dim=32)

        model = FusionModelWithGCMSDropout(
            smell_encoder=sensor_model,
            gcms_encoder=gcms_model,
            combined_dim=100,
            output_dim=len(le.classes_),
            gcms_dropout_p=0.3
        )

        batch_size = 32
        num_epochs = 64

        # train_fusion_dataset = FusionDataset(training_pair_data, le=le)

        # data_loader = DataLoader(train_fusion_dataset, batch_size=batch_size)

        # fusion_train(data_loader, model, logger, epochs=num_epochs)

        # torch.save(model.state_dict(), f'saved_models/fusion/gradient_period_{period_len}_model_weights.pth')

        model.load_state_dict(torch.load(f'saved_models/fusion/gradient_period_{period_len}_model_weights.pth'))

        dataset = TensorDataset(torch.tensor(testing_data), torch.tensor(testing_label))
        data_loader = DataLoader(dataset, batch_size=batch_size)
        
        fusion_evaluate(model, data_loader, le)


if __name__ == "__main__":
    main()
