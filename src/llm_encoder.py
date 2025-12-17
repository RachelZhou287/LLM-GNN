import torch
from transformers import AutoModel, AutoTokenizer

class QwenEncoder:
    def __init__(self, model_name="Qwen/Qwen2.5-1.5B", device="cuda"):
        print(f"Loading LLM: {model_name}...")
        self.device = device
        # Load Qwen model and tokenizer
        # Note: We use the base model, not the instruct/chat version, for better embeddings
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(self.device)
        self.model.eval() # Set to evaluation mode

    @torch.no_grad()
    def get_embedding(self, text_list, batch_size=64):
        embeddings = []
        print(f"Encoding {len(text_list)} texts...")
        
        for i in range(0, len(text_list), batch_size):
            batch_texts = text_list[i : i + batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512
            ).to(self.device)
            
            # Forward pass (Get hidden states)
            outputs = self.model(**inputs)
            
            # Strategy: Mean Pooling of the last hidden state
            # Shape: [Batch_Size, Seq_Len, Hidden_Dim] -> [Batch_Size, Hidden_Dim]
            last_hidden_state = outputs.last_hidden_state
            attention_mask = inputs.attention_mask.unsqueeze(-1)
            
            # Mask padding tokens to avoid noise
            masked_embeddings = last_hidden_state * attention_mask
            sum_embeddings = torch.sum(masked_embeddings, dim=1)
            sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
            
            mean_pooled = sum_embeddings / sum_mask
            embeddings.append(mean_pooled.cpu())
            
        return torch.cat(embeddings, dim=0)