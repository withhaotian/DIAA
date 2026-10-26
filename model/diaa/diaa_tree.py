import heapq
from model.diaa.utils import *

TREE_TEMPLATE = [[0], [1], [2], [3], [4], [5], [6], [7],
                        [0,0], [0,1], [0,2], [0,3], [0,4], [0,5], [0,6], [0,7], [1,0], [1,1], [1,2], [1,3], [2,0], [2,1], [2,2], [3,0], [3,1], [4,0], [5,0], [6,0], [7,0],
                        [0,0,0], [0,0,1], [0,0,2], [0,0,3], [0,0,4], [0,0,5], [0,0,6], [0,0,7], [0,1,0], [0,1,1], [0,1,2], [0,2,0], [0,2,1], [0,3,0], [0,4,0], [0,5,0], [0,6,0], [0,7,0], [1,0,0], [1,0,1], [1,1,0], [2,0,0], [3,0,0], [4,0,0], [5,0,0],
                        [0,0,0,0], [0,0,0,1], [0,0,0,2], [0,0,0,3], [0,0,0,4], [0,0,1,0], [0,0,1,1], [0,0,2,0], [0,0,3,0], [0,0,4,0], [0,1,0,0], [0,2,0,0], [1,0,0,0], [2,0,0,0], [3,0,0,0],
                        [0,0,0,0,0], [0,0,0,0,1], [0,0,0,0,2], [0,0,0,1,0], [0,0,0,2,0], [0,0,1,0,0], [0,1,0,0,0], [1,0,0,0,0],
                        [0,0,0,0,0,0], [0,0,0,0,0,1], [0,0,0,1,0,0],
                        ]

def tree_node_sort_key(path):
    return (len(path), path)

def build_diaa_tree(version, token_LUT=None, prob_LUT=None, input_ids=None, tree_budget=None, min_thres=None, max_depth=None, topk=8, topk_decay=0.45, depth_decay=0.55, print_tree=False):
    tree_template = None
    if version == 'dynamic':
        tree_template = []
        if token_LUT is None or prob_LUT is None or input_ids is None or tree_budget is None or min_thres is None or max_depth is None:
            raise ValueError("token_LUT, prob_LUT, input_ids, tree_budget, min_thres, and max_depth must be provided for dynamic tree")

        root_token = input_ids
        heap = [(-1.0, root_token, [], 0)]
        visited = set()
        
        for i in range(topk):
            heapq.heappush(heap, (-prob_LUT[root_token][i], token_LUT[root_token][i], [i], 1)) 
            tree_template.append([i])
            visited.add(tuple([i]))

        while heap and len(tree_template) < tree_budget:
            path_prob, token_id, path, depth = heapq.heappop(heap)
            path_prob = -path_prob

            if depth > max_depth:
                continue

            if depth > 1:
                if tuple(path) in visited:
                    continue
                visited.add(tuple(path))
                tree_template.append(path)

                if len(tree_template) >= tree_budget:
                    break

            topk_indices = token_LUT[token_id]
            topk_probs = prob_LUT[token_id]

            max_topk_prob = topk_probs[0]
            if depth == 0 and max_topk_prob == 0.:
                return TREE_TEMPLATE
            
            if depth > 0:
                depth_weight = depth_decay ** depth
            else:
                depth_weight = 1

            for i in range(topk):
                child_token = topk_indices[i]
                child_prob = topk_probs[i]
                if child_prob == 0.0:
                    continue

                new_path = path + [i]
                new_depth = depth + 1
                topk_weight = topk_decay ** i
                new_path_prob = path_prob * child_prob * depth_weight * topk_weight

                if new_path_prob < min_thres:
                    continue

                heapq.heappush(heap, (-new_path_prob, child_token, new_path, new_depth))

        tree_template = sorted(tree_template, key=tree_node_sort_key)

        zero_paths = [path for path in tree_template if all(x == 0 for x in path)]
        longest_zero_path = max(zero_paths, key=len)

        new_nodes = [longest_zero_path + [0]]
        new_nodes.append(new_nodes[-1] + [0])
        new_nodes.append(new_nodes[-1] + [0])

        tree_template.extend(new_nodes)

    elif version == '2.2.2':
        tree_template = TREE_TEMPLATE
    
    if tree_template is None:
        raise ValueError("Invalid version")
    
    if print_tree:
        print(tree_template)
        print("Number of Tree Nodes:", len(tree_template))
        print("Tree Depth:", len(tree_template[-1]))

    return tree_template
