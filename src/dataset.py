import torch
from torch_geometric.data import Data, InMemoryDataset
import pandas as pd
import os
import numpy as np

class YelpDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(YelpDataset, self).__init__(root, transform, pre_transform)
        if os.path.exists(self.processed_paths[0]):
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        # ⚠️ 确保这里的文件名和你 data/raw/ 下的一致
        return ['Yelp.csv'] 

    @property
    def processed_file_names(self):
        return ['yelp_graph_data.pt']

    def download(self):
        print(f"请将数据文件放入 {self.raw_dir} 目录")

    def process(self):
        print("开始处理数据...")
        
        # 1. 读取数据
        csv_path = os.path.join(self.raw_dir, self.raw_file_names[0])
        print(f"读取 CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        df['text'] = df['text'].fillna("No content")
        num_nodes = len(df)

        # 2. LLM 特征提取
        print("正在调用 Qwen 提取特征...")
        from src.llm_encoder import QwenEncoder
        device = "cuda" if torch.cuda.is_available() else "cpu"
        encoder = QwenEncoder(device=device)
        # 跑全量建议 batch_size=128, 测试跑建议 batch_size=8
        x = encoder.get_embedding(df['text'].tolist(), batch_size=64) 
        
        # 3. 标签
        # 将 -1 (欺诈) 变成 1，将 1 (正常) 变成 0
        y = torch.tensor((df['label'].values == -1).astype(int), dtype=torch.long)

        # ==================================================
        # 4. 构建 R-P-R (Review-Product-Review) 图结构
        # ==================================================
        print("正在构建 R-P-R 边 (同商品评论关联)...")
        
        # 步骤 A: 给 prod_id 编号
        if df['prod_id'].dtype == 'object':
            df['prod_code'] = pd.Categorical(df['prod_id']).codes
        else:
            df['prod_code'] = df['prod_id'].values

        # 步骤 B: 排序 (按商品分组，组内按时间排序)
        # 这样相同的商品会排在一起，且时间相邻
        sorted_df = df.sort_values(by=['prod_code', 'date'])
        sorted_indices = sorted_df.index.values # 拿到原始的行号
        prod_codes = sorted_df['prod_code'].values

        # 步骤 C: 错位比较，找到相邻且同商品的一对
        # prod_codes[:-1] 是前一个，prod_codes[1:] 是后一个
        mask = (prod_codes[:-1] == prod_codes[1:]) # 如果前后两个是同一个商品，则为 True
        
        # 获取源节点和目标节点
        src_nodes = sorted_indices[:-1][mask]
        dst_nodes = sorted_indices[1:][mask]
        
        # 步骤 D: 构建双向边 (无向图)
        edge_index_rpr = torch.tensor([
            np.concatenate([src_nodes, dst_nodes]),
            np.concatenate([dst_nodes, src_nodes])
        ], dtype=torch.long)
        
        print(f"R-P-R 边构建完成! 边数量: {edge_index_rpr.shape[1]}")
        # 自环 (Self-loops)
        edge_index_self = torch.stack([
            torch.arange(num_nodes, dtype=torch.long),
            torch.arange(num_nodes, dtype=torch.long)
        ], dim=0)

        # 合并所有边
        edge_index = torch.cat([edge_index_rpr, edge_index_self], dim=1)
        # ==================================================

        # 5. 封装与保存
        data = Data(x=x, edge_index=edge_index, y=y)
        
        # 划分数据集
        indices = torch.randperm(num_nodes)
        data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.train_mask[indices[:int(0.6 * num_nodes)]] = True
        data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.val_mask[indices[int(0.6 * num_nodes):int(0.8 * num_nodes)]] = True
        data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.test_mask[indices[int(0.8 * num_nodes):]] = True

        torch.save(self.collate([data]), self.processed_paths[0])
        print("处理完成，已保存。")