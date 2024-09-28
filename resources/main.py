import os
import torch
import pickle
import numpy as np
from tqdm import tqdm
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
from app.app import app, start_app, get_image_list


parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument("-m", "--metadata", action="store_true")

parser.add_argument("-s", "--similarity", action="store_true")

parser.add_argument("-p", "--path", action="store", default="D:/")

parser.add_argument("-d", "--device", type=str, default=None, help="Device to use: cuda or cpu")

parser.add_argument("--pkl_file", action="store", default="similarities.pkl")

parser.add_argument("--checkpoint", type=int, default=100)

parser.add_argument("--debug", action="store_true")

parser.add_argument("--total", type=int, default=10000)

parser.add_argument("--embedding", action="store_true")

parser.add_argument("--color", action="store_true")

parser.add_argument("--yolo", action="store_true")

parser.add_argument("--image_count", default=5, type=int)

args = parser.parse_args()


def run(args):
    """
    Runs the main image processing pipeline based on the provided arguments.

    Args:
        args: The arguments containing device information, file paths, and processing options.

    Returns:
        None
    """
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    create_database()
    try:
        with open(args.pkl_file, "rb") as f:
            similarities = pickle.load(f)
    except FileNotFoundError:
        similarities = defaultdict()

    last_db_id = get_last_entry()
    last_sim_id = len(similarities.keys())
    id = min(last_sim_id, last_db_id)

    if id == 0:
        starting_path = None
    else:
        starting_path = os.path.join(args.path, get_filename_from_id(id))
    img_gen = ImageGenerator(args.path).image_generator(starting_path=starting_path)

    for img in tqdm(img_gen, total=args.total, initial=id):
        id += 1
        if args.metadata and last_db_id < id:
            metadata = get_metadata(img)
            save_metadata_in_database(metadata)
        if args.similarity and last_sim_id < id:
            try:
                similarities[id] = get_similarities(img, args)
            except OSError:
                continue
            if id % args.checkpoint == 0:
                with open(args.pkl_file, "wb") as f:
                    pickle.dump(similarities, f)
    with open(args.pkl_file, "wb") as f:
        pickle.dump(similarities, f)


def load_pkl_files():
    print("Loading similarities from pickle file...")
    with open(args.pkl_file, 'rb') as f:
        sim = pickle.load(f)
        app.config["similarities"] = sim
        app.config["color_histograms"] = np.array([sim[key][0] for key in sim.keys()]).astype("float32")
        app.config["embeddings"] = np.array([sim[key][1] for key in sim.keys()]).astype("float32")
    print("Done!")


if __name__ == "__main__":
    if args.metadata or args.similarity:
        run(args)
    else:
        app.config["args"] = args
        load_pkl_files()
        get_image_list()
        if args.debug:
            app.run(debug=True)
        else:
            start_app()
