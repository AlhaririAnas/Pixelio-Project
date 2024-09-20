import os
import torch
import pickle
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from resources.generator import ImageGenerator
from resources.metadata_reader import (
    create_database,
    get_metadata,
    save_metadata_in_database,
    get_last_entry,
    get_filename_from_id,
)
from resources.similarity import get_similarities
from app.app import app, start_app
import numpy as np

# ArgumentParser setup
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument("-m", "--metadata", action="store_true")
parser.add_argument("-s", "--similarity", action="store_true")
parser.add_argument("-p", "--path", action="store", default="C:/")
parser.add_argument("-d", "--device", type=str, default=None, help="Device to use: cuda or cpu")
parser.add_argument("--pkl_file", action="store", default="similarities.pkl")
parser.add_argument("--checkpoint", type=int, default=100)
parser.add_argument("--debug", action="store_true")

args = parser.parse_args()

def default_similarity():
    return [None, None]

def run(args):
    """
    Runs the main image processing pipeline based on the provided arguments.
    """
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    create_database()

    # Load or initialize similarities
    try:
        with open(args.pkl_file, "rb") as f:
            similarities = pickle.load(f)
    except FileNotFoundError:
        similarities = defaultdict(default_similarity)  # Verwende die Funktion statt der Lambda-Funktion
    
    last_db_id = get_last_entry()
    last_sim_id = max(similarities.keys(), default=0)  # Ensure that we handle empty dicts correctly
    id = min(last_sim_id, last_db_id)

    starting_path = None if id == 0 else os.path.join(args.path, get_filename_from_id(id))
    img_gen = ImageGenerator(args.path).image_generator(starting_path=starting_path)

    for img in tqdm(img_gen, total=2313, initial=id):
        id += 1

        # Metadata extraction
        if args.metadata and last_db_id < id:
            try:
                metadata = get_metadata(img)
                save_metadata_in_database(metadata)
            except Exception as e:
                print(f"Error processing metadata for image {img.filename}: {e}")
                continue  # Skip the image and proceed to the next

        # Similarity calculation
        if args.similarity and last_sim_id < id:
            try:
                similarities[id] = get_similarities(img, args)
            except Exception as e:
                print(f"Error processing similarity for image {img.filename}: {e}")
                continue  # Skip the image and proceed to the next

        # Save results at regular intervals
        if id % args.checkpoint == 0:
            with open(args.pkl_file, "wb") as f:
                pickle.dump(similarities, f)

    # Final save of similarities
    with open(args.pkl_file, "wb") as f:
        pickle.dump(similarities, f)



def create_and_save_clustering_model(vectors, vector_ids, filename, clusters):
    """
    Creates and saves a clustering model based on the input vectors.
    """
    # Ensure vectors are in 2D form
    vectors = np.array(vectors)
    if vectors.ndim == 1:  # If it's 1D, reshape to 2D
        vectors = vectors.reshape(1, -1)

    n_samples = vectors.shape[0]
    
    # Ensure there are enough samples for clustering
    if n_samples < clusters:
        print(f"Warning: Number of samples ({n_samples}) is less than number of clusters ({clusters}). Reducing clusters to {n_samples}.")
        clusters = n_samples

    scaler = StandardScaler()
    vectors_scaled = scaler.fit_transform(vectors)

    model = KMeans(n_clusters=clusters, random_state=0)
    model.fit(vectors_scaled)

    with open(filename, "wb") as f:
        pickle.dump({"model": model, "scaler": scaler, "vector_ids": vector_ids}, f)
    print(f"Clustering model saved to {filename}")


def load_pkl_files():
    """
    A function to load pickle files containing similarities and embedding clusters.
    If the files are not found, it creates the clusters using the create_and_save_clustering_model function.
    """
    print("Loading similarities from pickle file...")
    try:
        with open(args.pkl_file, "rb") as f:
            similarities = pickle.load(f)
            if similarities is None:
                raise ValueError("Loaded similarities are None.")
            app.config["SIMILARITIES"] = similarities
    except FileNotFoundError:
        raise ValueError("No similarities found! Run the script with the -s flag.")

    # Check for valid similarities before proceeding with clustering
    if not similarities or all(similarity is None for similarity in similarities.values()):
        raise ValueError("No valid similarities found for clustering.")

    if not os.path.exists("embedding_cluster.pkl"):
        print("No embedding cluster found. Creating...")
        
        # Collect valid vectors and vector IDs for clustering
        vectors = [similarity[1] for similarity in similarities.values() if similarity[1] is not None]
        vector_ids = [v for v in similarities.keys() if similarities[v][1] is not None]

        if vectors and vector_ids:
            create_and_save_clustering_model(
                vectors,
                vector_ids,
                filename="embedding_cluster.pkl",
                clusters=5  # Adjust as needed
            )
        else:
            raise ValueError("No valid vectors for clustering.")

    with open("embedding_cluster.pkl", "rb") as f:
        app.config["EMBEDDING_CLUSTER"] = pickle.load(f)

    print("Done!")

if __name__ == "__main__":
    if args.metadata or args.similarity:
        run(args)
    else:
        app.config["ARGS"] = args
        load_pkl_files()
        if args.debug:
            app.run(debug=True)
        else:
            start_app()
