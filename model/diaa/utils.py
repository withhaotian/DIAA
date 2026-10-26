import torch
from safetensors.torch import save_file, load_file
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

def initialize_LUT(vocab_size, topk, device):
    token_LUT = torch.zeros((vocab_size, topk), dtype=torch.long, device=device, requires_grad=False)
    token_LUT_memory = token_LUT.element_size() * token_LUT.nelement()
    token_LUT_memory_MB = token_LUT_memory / (1024 ** 2)
    print(f'token_LUT size: ({vocab_size}, {topk}), memory: {token_LUT_memory_MB:.2f} MB')

    prob_LUT = torch.zeros((vocab_size, topk), dtype=torch.float32, device=device, requires_grad=False)
    prob_LUT_memory = prob_LUT.element_size() * prob_LUT.nelement()
    prob_LUT_memory_MB = prob_LUT_memory / (1024 ** 2)
    print(f'prob_LUT size: ({vocab_size}, {topk}), memory: {prob_LUT_memory_MB:.2f} MB')

    return token_LUT, prob_LUT

def load_LUT(lut_cache_path, device):
    lut_dict = load_file(lut_cache_path + '.safetensors')
    token_LUT = lut_dict['token_LUT'].to(torch.long).contiguous().to(device, non_blocking=True)
    prob_LUT = lut_dict['prob_LUT'].to(torch.float32).contiguous().to(device, non_blocking=True)
    print("Loaded LUT from cache!")
    return token_LUT, prob_LUT

def save_LUT(lut_cache_path, token_LUT, prob_LUT):
    save_file({'token_LUT': token_LUT.cpu(), 'prob_LUT': prob_LUT.cpu()}, lut_cache_path + '.safetensors')
    print("Saved LUT to cache.")

def cal_percentages(LUT):
    nonzero_count = torch.count_nonzero(LUT).item()  
    total_elements = LUT.numel()
    nonzero_percentage = (nonzero_count / total_elements) * 100

    print(f"Non-zero elements: {nonzero_count}/{total_elements} ({nonzero_percentage:.2f}%)")

    return nonzero_percentage

def draw_tree(tree_template):
    tree = defaultdict(list)
    root = "-1"

    for path in tree_template:
        current_node = root
        for node in path:
            child = f"{current_node}_{node}"
            if child not in tree[current_node]:
                tree[current_node].append(child)
            current_node = child

    G = nx.DiGraph()

    for parent, children in tree.items():
        for child in children:
            G.add_edge(parent, child)

    plt.figure(figsize=(12, 8))
    pos = nx.nx_agraph.graphviz_layout(G, prog="dot") 
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color="lightblue", font_size=10, arrows=True)
    plt.show()

if __name__ == "__main__":
    vocab_size = 32000
    topk = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    token_LUT, prob_LUT = initialize_LUT(vocab_size, topk, device)