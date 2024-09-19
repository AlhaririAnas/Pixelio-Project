import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity as cos_similarity
from scipy.spatial import distance
from resources.embedding import inception_v3


def euclidean_distance(v1, v2):
    return distance.euclidean(v1, v2)


def manhattan_distance(v1, v2):
    return distance.cityblock(v1, v2)


def cosine_similarity(v1, v2):
    return cos_similarity([v1], [v2])[0][0]


def get_similarities(img, args):
    # Only extract embeddings using inception_v3
    features = inception_v3(img, args.device)
    return features


def get_most_similar(img_paths, args, distance_measure, all_similarities, embedding_clusters):
    similarities = []
    embedding_distances = {}

    distance_func = {"euclidean": euclidean_distance, "manhattan": manhattan_distance, "cosine": cosine_similarity}[distance_measure]

    # Extract embeddings for each image
    for img_path in img_paths:
        img = Image.open(img_path)
        sim = get_similarities(img, args)
        similarities.append(sim)

    # Calculate the mean embedding similarity
    embedding_similarity = np.mean(similarities, axis=0)

    # Use clustering information to find the closest cluster
    model, scaler, vector_ids = embedding_clusters["model"], embedding_clusters["scaler"], embedding_clusters["vector_ids"]
    embedding_similarity_scaled = scaler.transform([embedding_similarity])
    cluster_label = model.predict(embedding_similarity_scaled)[0]

    # Find images in the same cluster and calculate the embedding distance
    same_cluster_ids = [vector_ids[i] for i, label in enumerate(model.labels_) if label == cluster_label]
    for image_id in same_cluster_ids:
        embedding_distance = np.mean([distance_func(all_similarities[image_id][1], embedding_similarity)])
        embedding_distances[image_id] = embedding_distance

    # Return the most similar embeddings
    embedding_most_similar = sorted(embedding_distances, key=embedding_distances.get)[:5]

    return embedding_most_similar
