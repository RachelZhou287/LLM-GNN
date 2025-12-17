import torch
import numpy as np
import random
import os
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, precision_recall_curve, auc

def set_seed(seed=42):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, precision_recall_curve, auc, recall_score, precision_score
from scipy.stats import ks_2samp
import numpy as np

def evaluate_metrics(y_true, y_pred_prob, threshold=0.5):
    """
    计算风控场景下的核心指标 (AUC, KS, AUPRC, F1)
    
    Args:
        y_true: 真实标签 (numpy array), 0=正常, 1=欺诈
        y_pred_prob: 模型预测为'1'(欺诈)的概率 (numpy array, 0.0~1.0)
        threshold: 判定为欺诈的概率阈值 (默认0.5)
    """
    # 确保输入是 numpy 数组
    if not isinstance(y_true, np.ndarray): y_true = np.array(y_true)
    if not isinstance(y_pred_prob, np.ndarray): y_pred_prob = np.array(y_pred_prob)

    # 1. 硬分类预测 (0 或 1)
    y_pred = (y_pred_prob > threshold).astype(int)
    
    # 2. 计算 AUC-ROC (排序能力)
    try:
        roc_auc = roc_auc_score(y_true, y_pred_prob)
    except ValueError:
        roc_auc = 0.0 # 防止测试集只有一个类别时报错

    # 3. 计算 KS 值 (核心风控指标)
    # 逻辑：分别拿到"坏人"和"好人"的预测概率分布，计算其最大距离
    try:
        probs_bad = y_pred_prob[y_true == 1]
        probs_good = y_pred_prob[y_true == 0]
        ks_statistic, _ = ks_2samp(probs_bad, probs_good)
    except:
        ks_statistic = 0.0

    # 4. 计算 AUPRC (不平衡数据下的精确率-召回率曲线面积)
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred_prob)
    auprc = auc(recall_curve, precision_curve)
    
    # 5. 基础指标 (F1, Precision, Recall)
    f1 = f1_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    
    return {
        "AUC": roc_auc,
        "KS": ks_statistic,   # 新增
        "AUPRC": auprc,
        "F1": f1,
        "Recall": recall,
        "Precision": precision
    }

def save_checkpoint(model, optimizer, epoch, path="checkpoints/best_model.pth"):
    """Saves the model state."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)
    print(f"Checkpoint saved to {path}")