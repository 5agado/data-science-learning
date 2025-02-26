import argparse
import sys
import numpy as np
from pathlib import Path
import cv2
import yaml
import hashlib

import chromadb

from face_utils import CONFIG_PATH, logger
from face_utils.FaceDetector import FaceDetector


def run_face_search(query_face_image: Path, target_search_dir: Path, config_path: Path,
                    similarity_threshold=0.8, min_resolution=50, output_dir: Path = None):
    """
    Run face search on target images and return matches that are above the similarity threshold.
    :param query_face_image: path to the query face image
    :param target_search_dir: directory with target images to search from
    :param similarity_threshold: threshold for face similarity
    :param min_resolution: minimum resolution for a face to be considered
    :param output_dir: optional directory to save the matches
    """
    # init face model
    face_model = get_face_model(config_path)

    # get embedding for query face
    query_emb = _get_faces_embeddings(face_model, query_face_image, min_resolution, 1)

    # run face similarity on target images
    # currently computing similarity individually for each pair. We will optimize this process in future iterations
    matches = []
    for extension in ['*.jpg', '*.png']:
        for img_path in target_search_dir.glob(extension):
            # get all faces from current image
            faces = face_model.detect_faces(cv2.imread(str(img_path)), min_res=min_resolution)
            for face in faces:
                # compute similarity and add match if above threshold
                similarity = cosine_distance(query_emb, face.embedding)
                if similarity > similarity_threshold:
                    matches.append((img_path, similarity, face.rect))
                    if output_dir:
                        out_file = output_dir / f'{int(similarity*100)}.jpg'
                        # copy image to a new dir
                        # shutil.copy(img_path, out_file)
                        # copy face to a new dir
                        cv2.imwrite(str(out_file), face.get_face_img())
    return matches


def populate_face_database(target_dir: Path, config_path: Path, db_path, collection: str, min_resolution: int):
    # get database collection
    collection = get_face_database_collection(db_path, collection)

    # init face model
    face_model = get_face_model(config_path)

    # run face similarity on target images
    # currently computing similarity individually for each pair. We will optimize this process in future iterations
    for extension in ['*.jpg', '*.png']:
        for img_path in target_dir.rglob(extension):
            # get image hash
            image = cv2.imread(str(img_path))
            image_md5 = hashlib.md5(image).hexdigest()
            # check if faces from this image are already present
            if len(collection.get(ids=[f"{image_md5}_0"])['ids']) > 0:
                logger.info(f'Faces from {img_path} are already present in the database. Skipping...')
            else:
                # get all faces from current image
                faces = face_model.detect_faces(image, min_res=min_resolution)
                for face_idx, face in enumerate(faces):
                    add_face_to_collection(collection, face.embedding.tolist(), img_path, face_idx, image_md5)

    return collection


def query_face_collection(query_face_image: Path, collection, nb_results: int, config_path: Path):
    # init face model
    face_model = get_face_model(config_path)

    # get embedding for query face
    query_emb = _get_faces_embeddings(face_model, query_face_image, 50, 1)

    # query db
    results = collection.query(
        query_embeddings=[query_emb.tolist()],
        n_results=nb_results,
    )

    return results


def get_face_model(config_path: Path):
    with open(str(config_path), 'r') as f:
        config = yaml.load(f, yaml.SafeLoader)
    face_model = FaceDetector(config, allowed_modules=['detection', 'recognition'])
    return face_model


def _get_faces_embeddings(face_model, image_path: Path, min_resolution: int, max_faces: int = 0) -> list:
    # get embedding for query face
    image = cv2.imread(str(image_path))
    faces = face_model.detect_faces(np.array(image), min_res=min_resolution)
    if len(faces) == 0:
        logger.warning("No face detected in the query image.")
        return []
    if max_faces and len(faces) > max_faces:
        logger.info(f"Multiple faces detected in the query image. Picking the first {max_faces}.")
        faces = faces[:max_faces]
    return [f.embedding for f in faces]


def get_face_database_collection(db_path, collection_name):
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(
        collection_name,
        metadata={"hnsw:space": "cosine"}  # l2 is the default
    )
    logger.info(f'db loaded from {db_path}. {collection.count()} entries found for collection {collection_name}.')
    return collection


def add_face_to_collection(collection, embedding, img_path, face_idx, image_md5):
    collection.add(
        embeddings=[embedding],
        metadatas=[{'img_path': str(img_path)}],
        ids=[f"{image_md5}_{face_idx}"]
    )


def cosine_distance(v1, v2):
    """Returns cosine distance between the two given vectors"""
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def euclidean_distance(v1, v2):
    return np.linalg.norm(v1 - v2)


def main(_=None):
    # TODO: support face search in videos
    parser = argparse.ArgumentParser(description='Local Face Search')
    parser.add_argument('-q', '--query', required=True, help="path to the image containing the query face")
    parser.add_argument('-t', '--target', required=True, help="path to the directory with target images to search from")
    parser.add_argument('-o', '--outdir', default='', help="optional dir where to save the matches")
    parser.add_argument('--sim-threshold', default=0.5, help="similarity threshold for face search")
    parser.add_argument('--min-res', default=20, help="minimum resolution for a face to be considered")
    parser.add_argument('-c', '--config-path', default=CONFIG_PATH)

    args = parser.parse_args()

    print('Running face search...')
    matches = run_face_search(Path(args.query), Path(args.target), Path(args.config_path),
                              float(args.sim_threshold), int(args.min_res), Path(args.outdir))
    print('Face search completed. Matches found:')
    print(matches)


if __name__ == "__main__":
    main(sys.argv[1:])
