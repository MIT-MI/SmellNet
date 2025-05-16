import torch
from scipy.stats import mode
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    accuracy_score,
)


def transformer_evaluate(model, testing_data, le, logger):
    WINDOW_SIZE = 100
    STRIDE = 50

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.double()

    def predict_by_sliding_window(df, model, ingredient):
        segments = []
        for start in range(0, len(df) - WINDOW_SIZE + 1, STRIDE):
            window = df.iloc[start : start + WINDOW_SIZE].values
            segments.append(window)

        X = torch.tensor(segments, dtype=torch.double).to(device)

        with torch.no_grad():
            logits = model(X)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            top5_preds = torch.topk(probs, k=5, dim=1).indices  # [num_windows, 5]

        return preds.cpu().numpy(), top5_preds.cpu().numpy(), [ingredient] * len(segments)

    # === Run on test_data ===
    all_preds = []
    all_true_labels = []
    all_top5 = []

    model.eval()
    for ingredient, dfs in testing_data.items():
        for df in dfs:
            df = df.iloc[:512]  # clip to 512 time steps
            preds, top5_preds, labels = predict_by_sliding_window(df, model, ingredient)
            all_preds.extend(preds)
            all_top5.append(top5_preds)
            all_true_labels.extend(labels)

    # === Evaluation ===
    true_labels_encoded = le.transform(all_true_labels)
    all_preds_np = np.array(all_preds)
    all_top5_np = np.vstack(all_top5)  # stack all top-5 arrays

    accuracy = accuracy_score(true_labels_encoded, all_preds_np) * 100

    # Compute Top-5 accuracy
    top5_correct = np.any(all_top5_np == true_labels_encoded[:, None], axis=1)
    top5_accuracy = np.mean(top5_correct) * 100

    decoded_preds = le.inverse_transform(all_preds_np)

    logger.info(f"✅ Window-Level Test Accuracy (Top-1): {accuracy:.2f}%")
    logger.info(f"✅ Window-Level Top-5 Accuracy: {top5_accuracy:.2f}%")

    # for true, pred in zip(all_true_labels, decoded_preds):
    #     logger.info(f"True: {true:15s} | Predicted: {pred}")


def regular_evaluate(model, data_loader, le, logger=None, lstm=False, translation=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.to(device)
    model.double()
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device, dtype=torch.double)
            labels = labels.to(device)

            if lstm:
                logits, embedding = model(inputs)
            elif translation:
                gcms_pred, logits = model(inputs)
            else:
                logits = model(inputs)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds) * 100

    if logger:
        logger.info(f"✅ Regular Evaluation Accuracy: {acc:.2f}%")
        # decoded_preds = le.inverse_transform(all_preds)
        # decoded_true = le.inverse_transform(all_labels)
        # for true, pred in zip(decoded_true, decoded_preds):
        #     logger.info(f"True: {true:15s} | Predicted: {pred}")
    else:
        print(f"✅ Accuracy: {acc:.2f}%")

    return acc


def regular_evaluate_top5(model, data_loader, le, logger=None, lstm=False, translation=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.to(device)
    model.double()
    model.eval()

    all_top5_correct = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device, dtype=torch.double)
            labels = labels.to(device)

            if lstm:
                logits, embedding = model(inputs)
            elif translation:
                _, logits = model(inputs)
            else:
                logits = model(inputs)
            probs = torch.softmax(logits, dim=1)
            
            # Get top-5 predictions
            top5_preds = torch.topk(probs, k=5, dim=1).indices  # shape [batch_size, 5]

            # Check if true label is in top-5 predictions
            for i in range(labels.size(0)):
                if labels[i].item() in top5_preds[i]:
                    all_top5_correct.append(1)
                else:
                    all_top5_correct.append(0)

            all_labels.extend(labels.cpu().numpy())

    # Calculate top-5 accuracy
    top5_acc = sum(all_top5_correct) / len(all_top5_correct) * 100

    if logger:
        logger.info(f"✅ Regular Evaluation Top-5 Accuracy: {top5_acc:.2f}%")
    else:
        print(f"✅ Top-5 Accuracy: {top5_acc:.2f}%")

    return top5_acc


def fusion_evaluate(model, data_loader, le, logger=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.to(device)
    model.double()
    model.eval()

    all_preds = []
    all_labels = []
    all_top5 = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device, dtype=torch.double)
            labels = labels.to(device)

            logits = model(inputs, torch.zeros((inputs.shape[0], 17), device=device, dtype=torch.double))
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            top5_preds = torch.topk(probs, k=5, dim=1).indices  # [batch_size, 5]

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_top5.append(top5_preds.cpu())

    all_labels_np = np.array(all_labels)
    all_preds_np = np.array(all_preds)
    all_top5_np = torch.cat(all_top5, dim=0).numpy()

    acc = accuracy_score(all_labels_np, all_preds_np) * 100

    # Compute Top-5 Accuracy
    top5_correct = np.any(all_top5_np == all_labels_np[:, None], axis=1)
    top5_acc = np.mean(top5_correct) * 100

    if logger:
        logger.info(f"✅ Regular Evaluation Accuracy (Top-1): {acc:.2f}%")
        logger.info(f"✅ Top-5 Accuracy: {top5_acc:.2f}%")
        decoded_preds = le.inverse_transform(all_preds_np)
        decoded_true = le.inverse_transform(all_labels_np)
        # for true, pred in zip(decoded_true, decoded_preds):
        #     logger.info(f"True: {true:15s} | Predicted: {pred}")
    else:
        print(f"✅ Accuracy (Top-1): {acc:.2f}%")
        print(f"✅ Top-5 Accuracy: {top5_acc:.2f}%")

    return acc, top5_acc


def contrastive_evaluate(
    test_smell_data,
    gcms_data,
    test_smell_label,
    gcms_encoder,
    sensor_encoder,
    logger,
    lstm=False
):
    """
    Evaluate how well the model matches GCMS embeddings to sensor embeddings.
    Returns:
      - Top-1 accuracy
      - Top-5 accuracy
      - Top-5 predicted indices per sample
      - Top-5 cosine similarity scores (pre-softmax)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gcms_encoder.to(device)
    sensor_encoder.to(device)

    gcms_encoder.eval()
    sensor_encoder.eval()

    # Move to device
    gcms_data = torch.tensor(gcms_data, dtype=torch.float).to(device)
    test_smell_data = torch.tensor(test_smell_data, dtype=torch.float).to(device)
    test_smell_label = torch.tensor(test_smell_label).to(device)

    with torch.no_grad():
        z_gcms = F.normalize(gcms_encoder(gcms_data), dim=1)
        if lstm:
            z_smell = F.normalize(sensor_encoder(test_smell_data)[0], dim=1)
        else:
            z_smell = F.normalize(sensor_encoder(test_smell_data), dim=1)

        # Cosine similarity matrix: [num_test_samples, num_gcms_samples]
        sim = torch.matmul(z_smell, z_gcms.T)

        # Top-1 accuracy
        top1_pred = sim.argmax(dim=1)
        top1_correct = (top1_pred == test_smell_label).float().mean().item() * 100

        # Top-5 predictions
        top5_sim, top5_pred = torch.topk(sim, k=5, dim=1)  # values and indices
        top5_correct = (
            top5_pred == test_smell_label.unsqueeze(1)
        ).any(dim=1).float().mean().item() * 100

    logger.info("------------------Test Statistics---------------------")
    logger.info(f"Top-1 Accuracy: {top1_correct:.2f}%")
    logger.info(f"Top-5 Accuracy: {top5_correct:.2f}%")

    print(top5_pred.cpu().numpy().shape)

    return top1_correct, top5_correct, top5_pred.cpu().numpy(), top5_sim.cpu().numpy()


def contrastive_evaluate_transformer(
    testing_data,
    testing_label,
    gcms_data,
    gcms_encoder,
    sensor_encoder,
    logger,
    window_size=100,
    stride=50,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gcms_encoder = gcms_encoder.float().to(device)
    sensor_encoder = sensor_encoder.float().to(device)
    gcms_encoder.eval()
    sensor_encoder.eval()

    # Prepare GCMS embeddings
    gcms_data_tensor = torch.tensor(gcms_data, dtype=torch.float).to(device)
    with torch.no_grad():
        gcms_embeddings = gcms_encoder(gcms_data_tensor)
        z_gcms = F.normalize(gcms_embeddings, dim=1)

    all_smell_embeddings = []
    all_smell_labels = []

    # Process smell data with sliding window
    for array, label_idx in zip(testing_data, testing_label):
        df_len = array.shape[0]
        segments = []
        for start in range(0, df_len - window_size + 1, stride):
            window = array[start : start + window_size]
            segments.append(window)

        if not segments:
            continue

        X = torch.tensor(segments, dtype=torch.float).to(device)

        with torch.no_grad():
            smell_embed = sensor_encoder(X)  # Already outputs [B, 32]
            smell_embed = F.normalize(smell_embed, dim=1)

        all_smell_embeddings.append(smell_embed)
        all_smell_labels.extend([label_idx] * smell_embed.shape[0])

    if not all_smell_embeddings:
        logger.warning("No valid smell embeddings were generated — check input data.")
        return 0, 0, 0, 0, None

    # Combine all smell embeddings
    all_smell_embeddings = torch.cat(all_smell_embeddings, dim=0)
    all_smell_labels = torch.tensor(all_smell_labels, dtype=torch.long).to(device)

    # Compute similarity matrix
    sim = torch.matmul(all_smell_embeddings, z_gcms.T)
    

    # Top-1 GCMS prediction
    predicted = sim.argmax(dim=1)

    # Evaluate
    correct = predicted == all_smell_labels
    accuracy = correct.float().mean().item()
    precision = precision_score(all_smell_labels.cpu(), predicted.cpu(), average="macro")
    recall = recall_score(all_smell_labels.cpu(), predicted.cpu(), average="macro")
    f1 = f1_score(all_smell_labels.cpu(), predicted.cpu(), average="macro")
    conf_matrix = confusion_matrix(all_smell_labels.cpu(), predicted.cpu())

    logger.info("------------------Transformer Contrastive Evaluation---------------------")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision (macro): {precision:.4f}")
    logger.info(f"Recall (macro): {recall:.4f}")
    logger.info(f"F1-Score (macro): {f1:.4f}")
    logger.info("Confusion Matrix:")
    logger.info(f"\n{conf_matrix}")

    return accuracy, precision, recall, f1, conf_matrix