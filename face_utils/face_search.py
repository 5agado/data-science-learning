import argparse
import logging
import sys
import numpy as np
from pathlib import Path
import cv2
import yaml

from face_utils import CONFIG_PATH
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
    with open(str(config_path), 'r') as f:
        config = yaml.load(f, yaml.SafeLoader)
    face_model = FaceDetector(config, allowed_modules=['detection', 'recognition'])

    # get embedding for query face
    query_image = cv2.imread(str(query_face_image))
    query_faces = face_model.detect_faces(np.array(query_image), min_res=min_resolution)
    if len(query_faces) == 0:
        logging.warning("No face detected in the query image. Exiting")
        return []
    elif len(query_faces) > 1:
        logging.info("Multiple faces detected in the query image. Picking the first one.")
    query_emb = query_faces[0].embedding

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
                        out_file = output_dir / f'{similarity:.2f}_{img_path.name}'
                        # copy image to a new dir
                        # shutil.copy(img_path, out_file)
                        # copy face to a new dir
                        cv2.imwrite(str(out_file), face.get_face_img())
    return matches


def cosine_distance(v1, v2):
    """Returns cosine distance between the two given vectors"""
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def euclidean_distance(v1, v2):
    return np.linalg.norm(v1 - v2)


def main(_=None):
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
