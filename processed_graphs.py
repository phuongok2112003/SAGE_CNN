import os
import glob
import torch
import xml.etree.ElementTree as ET
import networkx as nx
import numpy as np
from torch_geometric.data import Data, Dataset
from gensim.models import Word2Vec

class XMLGraphDataset(Dataset):
    def __init__(self, base_folder):
        super(XMLGraphDataset, self).__init__()
        self.base_folder = base_folder
        self.graphs = self.process_all_xml()
    
    def process_all_xml(self):
        graphs = []
        for folder in os.listdir(self.base_folder):
            folder_path = os.path.join(self.base_folder, folder)
            if os.path.isdir(folder_path):
                label = int(folder.split('_')[1].split('.')[0])  # Lấy nhãn từ tên thư mục
                xml_file = os.path.join(folder_path, "export.xml")
                if os.path.exists(xml_file):
                    graph = self.process_single_xml(xml_file, label)
                    if graph:
                        graphs.append(graph)
        return graphs
    
    def process_single_xml(self, xml_file, label):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        G = nx.DiGraph()
        node_features = {}
        edge_features = []
        node_mapping = {}
        
        # Xử lý các node
        for idx, node in enumerate(root.findall("node")):
            node_id = node.get("id")
            node_data = {data.get("key"): data.text if data.text else "UNKNOWN" for data in node.findall("data")}
            G.add_node(node_id, **node_data)
            node_features[node_id] = node_data
            node_mapping[node_id] = idx
        
        # Xử lý các cạnh
        edges = []
        for edge in root.findall("edge"):
            src = edge.get("source")
            tgt = edge.get("target")
            label = edge.find("data").text if edge.find("data") is not None else "UNKNOWN"
            
            if src in node_mapping and tgt in node_mapping:
                edges.append((node_mapping[src], node_mapping[tgt]))
                edge_features.append(label)
        
        if not edges:
            return None  # Bỏ qua nếu không có cạnh
        
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        
        # Sử dụng Word2Vec để nhúng đặc trưng của đỉnh
        sentences = [list(data.values()) for data in node_features.values()]
        w2v_model = Word2Vec(sentences=sentences, vector_size=100, min_count=1, workers=4)
        
        x = []
        for data in node_features.values():
            words = list(data.values())
            vectors = [w2v_model.wv[word] for word in words if word in w2v_model.wv]
            if vectors:
                x.append(np.mean(vectors, axis=0))
            else:
                x.append(np.zeros(100))
        
        x = torch.tensor(x, dtype=torch.float)
        
        # Mã hóa đặc trưng cạnh bằng CNN (ở bước huấn luyện)
        edge_attr = [hash(label) % 100 for label in edge_features]
        edge_attr = torch.tensor(edge_attr, dtype=torch.float).unsqueeze(1)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=torch.tensor([label], dtype=torch.long))
    
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        return self.graphs[idx]

# Đường dẫn thư mục chứa file XML
base_folder = "output"
dataset = XMLGraphDataset(base_folder)

# Lưu tập dữ liệu
torch.save(dataset.graphs, 'processed_graphs.pt')
print("Đã lưu dữ liệu đồ thị thành công!")
