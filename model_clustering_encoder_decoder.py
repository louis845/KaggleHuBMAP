import numpy as np
import torch
import sklearn.cluster

import model_permutation_layers

def compute_image_statistics(image, n_clusters=6):
    # This image is a numpy array of shape (height, width, channel) RGB image. Do hierarchical clustering with BisectingKMeans on the image into n clusters, and return the image with the clusters colored.
    height, width, channel = image.shape
    image = np.reshape(image, (height * width, channel)).astype(np.float64)

    # Do clustering. cluster_array is a deep nested list representing the cluster tree, where each list has either one or two elements.
    cluster_tree = {"items": np.ones((height * width), dtype=bool), "center": np.mean(image, axis=0)}

    for i in range(n_clusters - 1):
        # Loop through the leaves of the cluster tree, and find the cluster with the largest SSE.
        # Split the cluster with the largest SSE into two clusters, and add the two clusters into the cluster tree. Use sklearn's KMeans with k=2 to split the cluster.
        # Repeat until there are n clusters in the cluster tree.

        # Find the cluster with the largest SSE.
        max_sse = -1
        max_sse_stack = []

        # Do preorder traversal on cluster_tree, starting from the root node
        stack = []

        while True:
            current_node = cluster_tree
            for i in stack:
                current_node = current_node["subgroups"][i]

            # Check if the current node is a leaf node
            if "subgroups" not in current_node:
                sse = np.mean(np.square(image[current_node["items"], :] - current_node["center"]))

                if sse > max_sse:
                    max_sse = sse
                    del max_sse_stack
                    max_sse_stack = stack.copy()

                if len(stack) > 0 and stack[-1] == 0:
                    stack[-1] = 1
                else:
                    while len(stack) > 0 and stack[-1] == 1:
                        stack.pop()
                    if len(stack) == 0:
                        break
                    stack[-1] = 1
            else:
                stack.append(0)

        cluster_to_split = cluster_tree
        for i in max_sse_stack:
            cluster_to_split = cluster_to_split["subgroups"][i]

        # Split the cluster with the largest SSE into two clusters, and add the two clusters into the cluster tree. Use sklearn's KMeans with k=2 to split the cluster.
        image_cluster_colors = image[cluster_to_split["items"], :]
        cluster_mask = cluster_to_split["items"]

        kmeans = sklearn.cluster.KMeans(n_clusters=2, n_init=10)
        kmeans.fit(image_cluster_colors)

        first_cluster_items = np.zeros_like(cluster_mask, dtype=bool)
        second_cluster_items = np.zeros_like(cluster_mask, dtype=bool)

        first_cluster_items[cluster_mask] = kmeans.labels_ == 0
        second_cluster_items[cluster_mask] = kmeans.labels_ == 1
        cluster_to_split["subgroups"] = [{"items": first_cluster_items, "center": kmeans.cluster_centers_[0]},
                                         {"items": second_cluster_items, "center": kmeans.cluster_centers_[1]}]

    return image, cluster_tree

def construct_approximate_image(image_reshaped, cluster_tree, height, width, channel, n_clusters=6):
    # Assign each pixel to a cluster
    image_clustered = np.zeros_like(image_reshaped)

    # Do preorder traversal on cluster_tree, starting from the root node
    stack = []

    coloured = 0

    while True:
        current_node = cluster_tree
        for i in stack:
            current_node = current_node["subgroups"][i]

        # Check if the current node is a leaf node
        if "subgroups" not in current_node:
            image_clustered[current_node["items"], :] = current_node["center"]
            # 255.0 * (coloured / (n_clusters - 1))
            coloured += 1

            if len(stack) > 0 and stack[-1] == 0:
                stack[-1] = 1
            else:
                while len(stack) > 0 and stack[-1] == 1:
                    stack.pop()
                if len(stack) == 0:
                    break
                stack[-1] = 1
        else:
            stack.append(0)

    image_clustered = np.reshape(image_clustered, (height, width, channel))

    # Clip the image to be between 0 and 255, and convert to uint8
    image_clustered = np.clip(image_clustered, 0, 255).astype(np.uint8)

    return image_clustered

def torch_cluster_tree(cluster_tree, device):
    """Convert the cluster tree to a torch tensor, and move it to the device, while preserving the cluster tree structure.
    This method directly replaces the numpy arrays in the cluster tree with torch tensors, but does not return anything."""
    # cluster_tree["items"] = torch.tensor(cluster_tree["items"], dtype=torch.bool, device=device)
    cluster_tree["center"] = torch.tensor(cluster_tree["center"], dtype=torch.float32, device=device)
    if "subgroups" in cluster_tree:
        for subgroup in cluster_tree["subgroups"]:
            torch_cluster_tree(subgroup, device)

    return cluster_tree


class BinaryDecision(torch.nn.Module):
    def __init__(self, n_channels, n_clusters, n_hidden_layers=3, n_hidden_dim=16, activation=torch.nn.ELU()):
        super(BinaryDecision, self).__init__()

        self.layers = torch.nn.ModuleList()

        self.layers.append(torch.nn.Linear(n_channels * 2 + n_clusters, n_hidden_dim))
        self.layers.append(activation)
        # Append linear and activation layers
        for i in range(n_hidden_layers):
            self.layers.append(torch.nn.Linear(n_hidden_dim, n_hidden_dim))
            self.layers.append(activation)

        self.layers.append(torch.nn.Linear(n_hidden_dim, 1))
        self.layers.append(torch.nn.Sigmoid())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x

class Encoder(torch.nn.Module):
    def __init__(self, n_channels, n_clusters, n_hidden_layers=3, n_hidden_dim=16, activation=torch.nn.ELU()):
        super(Encoder, self).__init__()

        self.layers = torch.nn.ModuleList()

        self.layers.append(torch.nn.Linear(n_channels, n_hidden_dim))
        self.layers.append(activation)
        # Append linear and activation layers
        for i in range(n_hidden_layers):
            self.layers.append(torch.nn.Linear(n_hidden_dim, n_hidden_dim))
            self.layers.append(activation)

        self.layers.append(torch.nn.Linear(n_hidden_dim, n_clusters))

    def forward(self, cluster_tree):
        current_layer_list = [cluster_tree]

        for layer in range(self.max_layers):
            n_list = len(current_layer_list)
