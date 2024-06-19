import argparse
import json
import logging
import os
from pathlib import Path

import skimage.io

from docparser import stage1_entity_detector
from docparser import stage2_structure_parser
from docparser.utils.data_utils import create_dir_if_not_exists, find_available_documents, \
    create_eval_output_dir, generate_path_to_eval_dir, DocsDataset, get_dirname_for_path
from docparser.utils.eval_utils import generate_obj_detection_results_based_on_directories, \
    generate_relation_classification_results_based_on_directories, update_with_mAP_singlemodel, \
    convert_bbox_list_to_save_format, convert_table_structure_list_to_save_format, evaluate_icdar_xmls

logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.INFO)
import tqdm

from typing import Any


def detect_structures_single_page(
        image_filepath: str,
        entity_detector: Any,
        output_dir: str
):
    """Detects structures in a single page"""
    image = skimage.io.imread(image_filepath)
    entity_predictions = entity_detector.predict(image)

    det_dir = create_eval_output_dir(output_dir)
    image_name = os.path.basename(image_filepath)
    detections_textfile = os.path.join(image_name + '.txt')
    entity_detector.save_predictions_to_file(entity_predictions, det_dir, detections_textfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='DocParser Demoscript'
    )
    parser.add_argument('--page', action='store_true',
                        help='Run and evaluate DocParser default models for page structure parsing on arXivDocs-target')
    parser.add_argument('image_filepath', type=str, help='Path to the image file')
    parser.add_argument('output_dir', type=str, help='Path for the output predictions')
    args = parser.parse_args()

    cwd_path = os.getcwd()
    demo_dir = 'DocParser'
    current_dir = get_dirname_for_path(cwd_path)
    try:
        assert demo_dir == current_dir
    except AssertionError as e:
        logger.error(
            "Please run script from the 'docparser' directory (/PATH_TO_CODE/emnlp_codes/docparser), current dir: {}".format(
                current_dir))
        raise

    if not any([args.page, args.table, args.icdar, args.finetune]):
        print(parser.print_help())

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.page:
        entity_detector = stage1_entity_detector.EntityDetector(detection_max_instances=100)
        detect_structures_single_page(
            args.image_filepath,
            entity_detector,
            args.output_dir
        )