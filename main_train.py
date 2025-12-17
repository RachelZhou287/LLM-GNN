import torch
import torch.nn.functional as F
from src.dataset import YelpDataset
from src.gnn import FraudGNN
from src.utils import set_seed, evaluate_metrics, save_checkpoint
import numpy as np

# --- HYPERPARAMETERS ---
HIDDEN_DIM = 128
LEARNING_RATE = 0.001
WEIGHT_DECAY = 5e-4
EPOCHS = 500
DROPOUT = 0.5

def train():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load Data
    # Since we ran main_preprocess.py, this just loads the .pt file instantly
    dataset = YelpDataset(root='./data/')
    data = dataset[0].to(device)
    
    # 2. Calculate Class Weights for Imbalance
    # Fraud (Class 1) is rare, so we give it higher weight in the loss function
    num_normal = (data.y == 0).sum().item()
    num_fraud = (data.y == 1).sum().item()
    weight_fraud = num_normal / num_fraud
    
    # Pass weights to CrossEntropyLoss
    # Weights must be a tensor [weight_class_0, weight_class_1]
    class_weights = torch.tensor([1.0, weight_fraud]).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    
    print(f"Dataset Statistics:")
    print(f"  - Normal nodes: {num_normal}")
    print(f"  - Fraud nodes: {num_fraud}")
    print(f"  - Fraud Weight: {weight_fraud:.2f} (Fraud error counts {weight_fraud:.2f}x more)")

    # 3. Initialize Model
    model = FraudGNN(in_channels=data.num_features, 
                     hidden_channels=HIDDEN_DIM, 
                     out_channels=2).to(device) # Binary classification
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # 4. Training Loop
    best_auc = 0.0
    
    print("\nStarting Training...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        
        # Forward Pass
        out = model(data.x, data.edge_index)
        
        # Compute Loss (Only on Training Mask)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        
        # Backward Pass
        loss.backward()
        optimizer.step()

        # 5. Evaluation (Every 10 epochs)
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                # Get probabilities for class 1 (Fraud)
                logits = model(data.x, data.edge_index)
                probs = F.softmax(logits, dim=1)[:, 1] # Probability of being Fraud
                
                # Convert to CPU for sklearn metrics
                y_true_val = data.y[data.val_mask].cpu().numpy()
                y_probs_val = probs[data.val_mask].cpu().numpy()
                
                metrics = evaluate_metrics(y_true_val, y_probs_val)
                
                print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | "
                      f"Val AUC: {metrics['AUC']:.4f} | Val F1: {metrics['F1']:.4f}")
                
                # Save Best Model based on AUC
                if metrics['AUC'] > best_auc:
                    best_auc = metrics['AUC']
                    save_checkpoint(model, optimizer, epoch, path="checkpoints/best_fraud_gnn.pth")

    # 6. Final Test
    print("\nLoading best model for Test Set evaluation...")
    checkpoint = torch.load("checkpoints/best_fraud_gnn.pth",weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        probs = F.softmax(logits, dim=1)[:, 1]
        
        y_true_test = data.y[data.test_mask].cpu().numpy()
        y_probs_test = probs[data.test_mask].cpu().numpy()
        
        test_metrics = evaluate_metrics(y_true_test, y_probs_test)
        
    print("="*50)
    print("FINAL TEST RESULTS (Risk Control Report)")
    print("="*50)
    print(f"AUC (Ranking Quality):     {test_metrics['AUC']:.4f}")
    print(f"KS:                        {test_metrics['KS']:.4f}")
    print(f"F1 Score (Balance):        {test_metrics['F1']:.4f}")
    print(f"AUPRC (Precision-Recall):  {test_metrics['AUPRC']:.4f}")
    print(f"Recall (Fraud Catch Rate): {test_metrics['Recall']:.4f}")
    print("="*50)

if __name__ == "__main__":
    train()
