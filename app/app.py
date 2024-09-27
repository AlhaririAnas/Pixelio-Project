from flask import Flask, render_template, jsonify, request
import os
from waitress import serve
import webbrowser
import faiss
import numpy as np
import sqlite3
from resources.yolo import mean_iou


app = Flask(__name__)


def get_image_list():
    images = []
    for root, _, files in os.walk('app/static/images'):
            for file in files:
                if file.lower().endswith(("png", "jpg", "jpeg")):
                    file_path = os.path.join(root, file)
                    images.append(file_path.replace('app/static/images', ''))
    app.config["images"] = images

def get_similar_images(img_path):
    img_path = ('dataset/' + img_path).replace('/', '\\')
    conn = sqlite3.connect("image_metadata.db")
    query = f"SELECT id FROM metadata WHERE filename = (?)"
    id = conn.execute(query, [img_path]).fetchall()[0][0]
    conn.close()
    if app.config["args"].color:
        index = faiss.IndexFlat(384)
        index.add(app.config["color_histograms"])
        _, indices = index.search(np.array(app.config["similarities"][id][0].reshape(1, -1)), 11)
        print(indices)
    elif app.config["args"].embedding:
        index = faiss.IndexFlat(2048)
        index.add(app.config["embeddings"])
        _, indices = index.search(np.array(app.config["similarities"][id][1].reshape(1, -1)), 11)
    elif app.config["args"].yolo:
        yolo_distances = {}
        yolo_classes = app.config["similarities"][id][2].keys()
        for image_id, vectors in app.config["similarities"].items():
            if any(item in list(vectors[2].keys()) for item in yolo_classes):
                yolo_distance = np.mean([mean_iou(similarity, vectors[2]) for similarity in [app.config["similarities"][id][2]]])
                yolo_distances[image_id] = yolo_distance
        indices = [sorted(yolo_distances, key=yolo_distances.get, reverse=True)[:5]]
        indices = [[i-1 for i in indices[0]]]
    return [app.config["images"][i] for i in indices[0] if i != id-1]


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/load_images')
def load_images():
    offset = int(request.args.get('offset', 0))
    limit = int(request.args.get('limit', 50))
    images = app.config["images"][offset:offset+limit]
    return jsonify(images)

@app.route('/similar_images/<path:image_name>')
def similar_images(image_name):
    similar_images_list = get_similar_images(image_name)
    return jsonify(similar_images_list)

def start_app():
    webbrowser.open("http://127.0.0.1:8080")
    serve(app, host="127.0.0.1", port=8080)

if __name__ == '__main__':
    app.run(debug=True)
