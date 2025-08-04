from models import *
from load_data import *
from train import *
from evaluate import *
import logging
import os
import random
import time
import torch
from collections import Counter
from torch.utils.data import TensorDataset, DataLoader, random_split


log_dir = "/home/dewei/workspace/SmellNet/logs"

log_file_path = os.path.join(log_dir, f"four_channel_lstm_gradient_{time.time()}.log")

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

    training_path = "/home/dewei/workspace/SmellNet/four_channel"
    # testing_path = "/home/dewei/workspace/SmellNet/testing"

    # for category in ["Nuts", "Spices", "Herbs", "Fruits", "Vegetables"]:
    #     logger.info(category)

    training_data, min_len = load_four_channel_sensor_data(
        training_path
    )

    training_data, training_label, testing_data, testing_label = process_directory_to_windows(
        training_data
    )

    batch_size = 32

    train_dataset = TensorDataset(torch.tensor(training_data), torch.tensor(training_label))
    test_dataset = TensorDataset(torch.tensor(testing_data), torch.tensor(testing_label))

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Model
    model = SmellReproductionLSTMNet(
        input_dim=4,
        hidden_dim=256,
        embedding_dim=12,
        num_classes=12,
    )

    # Train
    reproduction_train(train_loader, model, logger, epochs=256, lstm=True)

    # Optional: Evaluate on test set
    evaluate_model(test_loader, model, logger)

    # return model

    torch.save(model.state_dict(), f'/home/dewei/workspace/SmellNet/saved_models/four_channel_reconstruction/model_weights.pth')
    # dataset = TensorDataset(torch.tensor(testing_data), torch.tensor(testing_label))
    # data_loader = DataLoader(dataset, batch_size=batch_size)
    # model.load_state_dict(torch.load(f'saved_models/lstm/gradient_period_{period_len}_model_weights.pth'))
    # regular_evaluate(model, data_loader, le, logger, lstm=True)
    # regular_evaluate_top5(model, data_loader, le, logger, lstm=True)


def evaluate_model(test_loader, model, logger):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    total_loss = 0
    total_samples = 0
    total_quant_error = 0.0  # For quantitative error
    criterion = soft_cross_entropy
    bad_sample = 0

    with torch.no_grad():
        for batch_x, batch_label in test_loader:
            if batch_label.sum().item() == 0:
                bad_sample += 1
                continue
            batch_x = batch_x.to(device, dtype=torch.float32)
            batch_label = batch_label.to(device, dtype=torch.float32)
            
            logits, _ = model(batch_x)
            probs = torch.softmax(logits, dim=1)

            # Apply thresholding to nearest 0.1
            probs_rounded = torch.round(probs * 10) / 10.0

            # Quantitative error: mean absolute difference between rounded and true probs
            quant_error = torch.sum(torch.abs(probs_rounded - batch_label))
            total_quant_error += quant_error.item()

            # Loss calculation
            loss = criterion(logits, batch_label)
            total_loss += loss.item()
            total_samples += batch_x.size(0)

    avg_loss = total_loss / (len(test_loader) - bad_sample)
    avg_quant_error = total_quant_error / total_samples

    logger.info(f"Test: Loss = {avg_loss:.4f}, Quant Error = {avg_quant_error:.4f}")


def run_experiment(name, runs, **kwargs):
    logger = logging.getLogger()
    logger.info(
        f"------------------------------------{name}-------------------------------------------"
    )
    for run_id in range(runs):
        logger.info(f"[{name} Run {run_id+1}] Starting")
        start_time = time.time()
        model = main()
        end_time = time.time() - start_time
        logger.info(f"[{name} Run {run_id+1}] Training time: {end_time:.2f}s")


def load_model(model_path):
    model = SmellReproductionLSTMNet(
        input_dim=4,
        hidden_dim=256,
        embedding_dim=12,
        num_classes=12,
    )
    model.load_state_dict(torch.load(model_path))
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Succesfully loaded model on to device {device}")
    return model


if __name__ == "__main__":
    # logger = logging.getLogger()
    # runs = 10

    # run_experiment("Gradient Period 25", runs)
    # run_experiment("Gradient Period 50", runs, period_len=50)
    run_experiment("Gradient", 1)

    # to run the model
    # you do probs = torch.softmax(logits, dim=1)

    # # Apply thresholding to nearest 0.1
    # probs_rounded = torch.round(probs * 10) / 10.0