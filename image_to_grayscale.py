import model_data_manager

import argparse
import time

import cv2
import numpy as np
import sklearn.cluster

def convert_grayscale(image):
    # This image is a numpy array of shape (height, width, channel) RGB image. Do hierarchical clustering with BisectingKMeans on the image into n clusters, and return the image with the clusters colored.
    n_clusters = 16

    height, width, channel = image.shape
    image = np.reshape(image, (height * width, channel)).astype(np.float64)

    # Do clustering. cluster_array is a deep nested list representing the cluster tree, where each list has either one or two elements.
    cluster_tree = {"items": np.ones((height * width), dtype=bool), "center":np.mean(image, axis=0)}

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
        cluster_to_split["subgroups"] = [{"items": first_cluster_items, "center": kmeans.cluster_centers_[0]}, {"items": second_cluster_items, "center": kmeans.cluster_centers_[1]}]

    cluster_stacks = np.empty(shape=n_clusters, dtype=object)
    cluster_colors = np.zeros(shape=(n_clusters, 1, 3), dtype=np.float64)
    count = 0

    stack = []
    while True:
        current_node = cluster_tree
        for i in stack:
            current_node = current_node["subgroups"][i]

        # Check if the current node is a leaf node
        if "subgroups" not in current_node:
            cluster_stacks[count] = stack.copy()
            cluster_colors[count, 0, :] = current_node["center"]
            count += 1

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

    cluster_colors = cv2.cvtColor(cluster_colors.astype(dtype=np.uint8), cv2.COLOR_RGB2HLS)[:, 0, 1]
    rank = np.argsort(np.argsort(cluster_colors))

    # Assign each pixel to a cluster
    image_clustered = np.zeros_like(image)

    # Do preorder traversal on cluster_tree, starting from the root node
    for k in range(cluster_stacks.shape[0]):
        current_node = cluster_tree
        for i in cluster_stacks[k]:
            current_node = current_node["subgroups"][i]
        image_clustered[current_node["items"], :] = 255.0 * (rank[k] / (n_clusters - 1))

    image_clustered = np.reshape(image_clustered, (height, width, channel))

    # Clip the image to be between 0 and 255, and convert to uint8
    image_clustered = np.clip(image_clustered, 0, 255).astype(np.uint8)

    return image_clustered

parser = argparse.ArgumentParser(description="Convert the images to grayscale.")

model_data_manager.transform_add_argparse_arguments(parser, requires_model=False)

args = parser.parse_args()

input_data_loader, output_data_writer, _ = model_data_manager.transform_get_argparse_arguments(args, requires_model=False)

count = 0
ctime = time.time()
for image in model_data_manager.data_information.index:
    image_np = input_data_loader.get_image_data(image)

    image_np = np.expand_dims(convert_grayscale(image_np)[..., 0], axis=-1)

    assert image_np.shape == (512, 512, 1)

    output_data_writer.write_image_data(image, image_np)

    count += 1
    if count % 100 == 0:
        print("Processed {} images in {} seconds.".format(count, time.time() - ctime))
        ctime = time.time()

input_data_loader.close()
output_data_writer.close()