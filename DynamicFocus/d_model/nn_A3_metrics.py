import torch
from tensorboard.compat.proto.histogram_pb2 import HistogramProto

from d_model.nn_A0_utils import RAM
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix


def calc_confusion_matrix(preds, targets, num_classes):
    """
    计算每个类别的 TP、FP、TN 和 FN
    :param preds: 模型的预测输出
    :param targets: 真实标签
    :param num_classes: 类别总数
    :return: 每个类别的 TP、FP、TN 和 FN
    """
    confusion_matrix = torch.zeros((num_classes, 4))  # 每个类别的 [TP, FP, FN, TN]

    for cls in range(num_classes):
        # 当前类别的布尔掩码
        pred_class = preds == cls
        true_class = targets == cls

        # 计算 TP, FP, FN, TN
        TP = (pred_class & true_class).sum().item()
        FP = (pred_class & ~true_class).sum().item()
        FN = (~pred_class & true_class).sum().item()
        TN = (~pred_class & ~true_class).sum().item()

        confusion_matrix[cls] = torch.tensor([TP, FP, FN, TN])

    return confusion_matrix


def calc_metrics(confusion_matrix):
    """
    根据混淆矩阵计算 IoU、Precision、Recall、F1 和 Accuracy
    :param confusion_matrix: 每个类别的 [TP, FP, FN, TN]
    :return: 每个类别的 IoU、Precision、Recall、F1 和 Accuracy
    """
    TP = confusion_matrix[:, 0]
    FP = confusion_matrix[:, 1]
    FN = confusion_matrix[:, 2]
    TN = confusion_matrix[:, 3]
    eps = 1e-7
    # 计算 IoU
    iou = TP / (TP + FP + FN + eps)

    # 计算 Precision
    precision = TP / (TP + FP + eps)

    # 计算 Recall
    recall = TP / (TP + FN + eps)

    # 计算 Accuracy
    accuracy = (TP + TN) / (TP + TN + FP + FN + eps)

    # 计算 F1-score
    f1 = 2 * (precision * recall) / (precision + recall + eps)

    return iou, f1, accuracy, precision, recall


def get_report(y_pred_N: torch.Tensor, y_trgt_N: torch.Tensor, num_classes: int):
    confusion_matrix = calc_confusion_matrix(y_pred_N, y_trgt_N, num_classes)

    iou, precision, recall, f1, accuracy = calc_metrics(confusion_matrix)

    miou = iou.mean().item()
    mean_f1 = f1.mean().item()
    mean_precision = precision.mean().item()
    mean_recall = recall.mean().item()
    mean_accuracy = accuracy.mean().item()

    print(f"Mean IoU (mIoU): {miou:.4f}")
    print(f"Mean F1-score: {mean_f1:.4f}")
    print(f"Mean Precision: {mean_precision:.4f}")
    print(f"Mean Recall: {mean_recall:.4f}")
    print(f"Mean Accuracy: {mean_accuracy:.4f}")


def evaluate_segmentation(pred_Bx1xHxW: torch.Tensor, target_Bx1xHxW: torch.Tensor):
    # 将 BxHxW 展平为 B x (H * W)
    B, _, H, W = pred_Bx1xHxW.shape
    threshold = 0.5
    mgpu = RAM()

    mgpu.pred_BxHW = pred_Bx1xHxW.flatten(start_dim=1) >= threshold  # B x (H * W)
    mgpu.target_BxHW = target_Bx1xHxW.flatten(start_dim=1) >= threshold  # B x (H * W)

    # 计算 True Positive, True Negative, False Positive, False Negative
    mgpu.TP_B = torch.sum((mgpu.pred_BxHW) & (mgpu.target_BxHW), dim=1).float()
    mgpu.TN_B = torch.sum((~mgpu.pred_BxHW) & (~mgpu.target_BxHW), dim=1).float()
    mgpu.FP_B = torch.sum((mgpu.pred_BxHW) & (~mgpu.target_BxHW), dim=1).float()
    mgpu.FN_B = torch.sum((~mgpu.pred_BxHW) & (mgpu.target_BxHW), dim=1).float()
    del mgpu.pred_BxHW, mgpu.target_BxHW

    # # 将结果堆叠成 Bx4 的张量
    # confusion_matrix_Bx4 = torch.stack([TP, TN, FP, FN], dim=1)

    eps = 1e-7
    mgpu.iou_B = mgpu.TP_B / (mgpu.TP_B + mgpu.FP_B + mgpu.FN_B + eps)
    mgpu.precision_B = mgpu.TP_B / (mgpu.TP_B + mgpu.FP_B + eps)
    mgpu.recall_B = mgpu.TP_B / (mgpu.TP_B + mgpu.FN_B + eps)
    mgpu.accuracy_B = (mgpu.TP_B + mgpu.TN_B) / (mgpu.TP_B + mgpu.TN_B + mgpu.FP_B + mgpu.FN_B + eps)
    mgpu.f1_B = 2 * (mgpu.precision_B * mgpu.recall_B) / (mgpu.precision_B + mgpu.recall_B + eps)

    del mgpu.TP_B, mgpu.TN_B, mgpu.FP_B, mgpu.FN_B

    iou_B = mgpu.iou_B.tolist()
    precision_B = mgpu.precision_B.tolist()
    recall_B = mgpu.recall_B.tolist()
    accuracy_B = mgpu.accuracy_B.tolist()
    f1_B = mgpu.f1_B.tolist()

    del mgpu.iou_B, mgpu.precision_B, mgpu.recall_B, mgpu.accuracy_B, mgpu.f1_B
    mgpu.gc()

    return iou_B, f1_B, accuracy_B, precision_B, recall_B


def evaluate_classification(predict_BxK: torch.Tensor, target_Bx1: torch.Tensor, class_num: int):
    # Convert predictions to class indices
    predict_B = torch.argmax(predict_BxK, dim=1).cpu().numpy()  # Convert to numpy array for sklearn
    target_B = target_Bx1.squeeze(dim=1).cpu().numpy()  # Convert to numpy array for sklearn

    # Initialize dictionaries to hold precision, recall, F1, accuracy per class
    precision_per_class = {}
    recall_per_class = {}
    f1_per_class = {}
    accuracy_per_class = {}

    # For each class, calculate precision, recall, F1-score, and accuracy
    for k in range(class_num):
        # For binary classification of class k, treat class k as the positive class
        # and all other classes as negative.
        binary_target = (target_B == k).astype(int)  # Convert to binary labels for class k
        binary_predict = (predict_B == k).astype(int)  # Convert to binary predictions for class k

        # Precision, recall, F1 for class k
        precision_per_class[k] = precision_score(binary_target, binary_predict, zero_division=0)
        recall_per_class[k] = recall_score(binary_target, binary_predict, zero_division=0)
        f1_per_class[k] = f1_score(binary_target, binary_predict, zero_division=0)

        # Accuracy for class k (proportion of correct predictions for both class k and non-k)
        accuracy_per_class[k] = accuracy_score(binary_target, binary_predict)

    return f1_per_class, accuracy_per_class, precision_per_class, recall_per_class

