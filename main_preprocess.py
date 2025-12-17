import time
from src.dataset import YelpDataset  # Or whatever you named your dataset class
from src.utils import set_seed

def main():
    set_seed(42)
    
    print("="*50)
    print("STARTING PREPROCESSING PIPELINE")
    print("This step runs the LLM (Qwen) to generate embeddings.")
    print("For 600k reviews, this may take time. Ensure GPU is active.")
    print("="*50)

    start_time = time.time()
    
    # Initialize the dataset. 
    # This automatically calls the .process() method in dataset.py 
    # if the processed file (data.pt) does not exist.
    dataset = YelpDataset(root='./data/')
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    print(f"\nPreprocessing Complete!")
    print(f"Time taken: {elapsed/60:.2f} minutes")
    print(f"Graph Info:")
    print(f"  - Number of Nodes: {dataset[0].num_nodes}")
    print(f"  - Number of Edges: {dataset[0].num_edges}")
    print(f"  - Node Feature Dim: {dataset[0].num_features} (Should match Qwen hidden dim)")
    print(f"  - Number of Classes: {dataset.num_classes}")

if __name__ == "__main__":
    main()