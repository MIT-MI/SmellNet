import torch
import torch.nn as nn
import torch.nn.functional as F


def cross_modal_contrastive_loss(z1, z2, temperature=0.07):
    """
    Contrastive loss between two batches of embeddings, z1 and z2.
    We treat (z1[i], z2[i]) as the positive pair, and all others as negatives.
    """
    # 1) L2-normalize each embedding
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    batch_size = z1.size(0)

    # 2) Similarity matrix: [batch_size, batch_size]
    # each entry sim[i, j] = dot(z1[i], z2[j]) / temperature
    sim = torch.matmul(z1, z2.t()) / temperature

    # 3) For row i, the correct "label" is i (the diagonal)
    labels = torch.arange(batch_size, device=z1.device)

    # 4) Cross entropy loss
    # We'll interpret each row i of 'sim' as a distribution over j,
    # and the "correct" j is i.
    loss_12 = F.cross_entropy(sim, labels)
    loss_21 = F.cross_entropy(sim.t(), labels)
    loss = 0.5 * (loss_12 + loss_21)

    return loss


class CrossModalTranslationLoss(nn.Module):
    def __init__(self, lambda_=0.5):
        """
        lambda_: weight for the GC-MS translation loss (MSE),
                 (1 - lambda_) is used for classification loss
        """
        super().__init__()
        self.lambda_ = lambda_
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, gcms_pred, gcms_target, class_logits, class_labels):
        """
        gcms_pred: predicted GC-MS vector (B, Dg)
        gcms_target: ground truth GC-MS vector (B, Dg)
        class_logits: predicted class logits (B, C)
        class_labels: ground truth class index (B,)
        """
        loss_gcms = self.mse_loss(gcms_pred, gcms_target)
        loss_class = self.ce_loss(class_logits, class_labels)
        loss = self.lambda_ * loss_gcms + (1 - self.lambda_) * loss_class
        return loss, loss_gcms, loss_class