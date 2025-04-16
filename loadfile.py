import torch

graphs = torch.load('graphs.pt',weights_only=False)


print("Node = ",graphs[0].x)
print("edge_attr = ",graphs[0].edge_attr)
print("edge_index = ",graphs[0].edge_index)
print("lable = ",graphs[0].y[0])