from models import *
from load_data import *
from train import *
from evaluate import *
from dataset import *
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
    real_time_testing_path = "/home/dewei/workspace/smell-net/real_time_testing_spice"
    gcms_path = "/home/dewei/workspace/smell-net/processed_full_gcms_dataframe.csv"

    period_len = 25

    # for category in ["Nuts", "Spices", "Herbs", "Fruits", "Vegetables"]:
    #     logger.info(category)

    training_data, testing_data, real_time_testing_data, min_len = load_sensor_data(
        training_path, testing_path, real_time_testing_path=real_time_testing_path)

    gcms_scaled, y_encoded, le, scaler = load_gcms_data(gcms_path)
        
    train_data, train_label, _ = prepare_data_transformer_gradient(training_data, le=le, period_len=period_len)
    test_data, test_label, _ = prepare_data_transformer_gradient(testing_data, le=le, period_len=period_len)
    real_test_data, real_test_label, _ = prepare_data_transformer_gradient(real_time_testing_data, le=le, period_len=period_len)

    training_pair_data, _ = create_pair_data(train_data, train_label, gcms_scaled, le)

    train_dataset = PairedDataset(training_pair_data)

    batch_size = 32
    num_epochs = 256

    # sampler = UniqueGCMSampler(train_dataset.data, batch_size)
    # loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    
    sensor_model = LSTMNet(
        input_dim=12, hidden_dim=128, embedding_dim=len(le.classes_), num_classes=len(le.classes_)
    )

    gcms_model = Encoder(input_dim=17, output_dim=len(le.classes_))

    # contrastive_train(gcms_model, sensor_model, loader, logger, num_epochs=num_epochs, lstm=True)

    # torch.save(sensor_model.state_dict(), f'saved_models/lstm_contrastive/gradient_sensor_model_weights.pth')
    # torch.save(gcms_model.state_dict(), f'saved_models/lstm_contrastive/gradient_gcms_model_weights.pth')

    sensor_model.load_state_dict(torch.load(f'saved_models/lstm_contrastive/gradient_sensor_model_weights.pth'))
    gcms_model.load_state_dict(torch.load(f'saved_models/lstm_contrastive/gradient_gcms_model_weights.pth'))

    contrastive_evaluate(real_test_data, gcms_scaled, real_test_label, gcms_model, sensor_model, logger, lstm=True)


if __name__ == "__main__":
    main()
