import argparse
import json
import logging
import os
from pathlib import Path
from glob import glob
import cv2
import numpy as np

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
        structure_detector: Any,
        output_dir: str,
        do_postprocessing: bool,
        save_images_with_predictions: bool
):
    """Detects structures in a single page"""
    image = skimage.io.imread(image_filepath)
    entity_predictions = entity_detector.predict(image)

    det_dir = create_eval_output_dir(output_dir)
    image_name = os.path.basename(image_filepath)
    detections_textfile = os.path.join(image_name + '.txt')

    # Detect entities
    entity_detector.save_predictions_to_file(entity_predictions, det_dir, detections_textfile)

    # Find the detection files
    detection_file = f'{det_dir}/{image_name}.txt'
    assert os.path.exists(detection_file), f"Detection file not found: {detection_file}"
    img_relations_dict = structure_detector.create_structure_for_doc(
        detection_file,
        table_mode=False,
        do_postprocessing=do_postprocessing
    )

    detections_postfix = 'origimg'
    postprocessed_postfix = 'postprocessed'
    detection_files_dir = os.path.join(output_dir, 'detections' + '_' + detections_postfix)
    Path(detection_files_dir).mkdir(parents=True, exist_ok=True)

    if do_postprocessing:
        detection_files_dir_postprocessing = detection_files_dir.replace(detections_postfix, '') + postprocessed_postfix
        Path(detection_files_dir_postprocessing).mkdir(parents=True, exist_ok=True)
        predictions_dict = convert_bbox_list_to_save_format(img_relations_dict['all_bboxes'])
        detection_filename = os.path.basename(detection_file)

        stage1_entity_detector.EntityDetector.write_predictions_to_file(predictions_dict,
                                                                        detection_files_dir_postprocessing,
                                                                        detection_filename)
        relations_subdir = os.path.join(detection_files_dir_postprocessing, 'relations')
    else:
        relations_subdir = os.path.join(detection_files_dir, 'relations')

    Path(relations_subdir).mkdir(parents=True, exist_ok=True)
    relations_filename = os.path.basename(detection_file).replace('.png.txt', 'png.txt_relations.json')
    relations_path = os.path.join(relations_subdir, relations_filename)
    logger.debug('saving relations to {}'.format(relations_path))
    with open(relations_path, 'w') as out_file:
        json.dump(img_relations_dict['relations'], out_file, indent=1)

    # Draw bboxes
    if save_images_with_predictions and do_postprocessing:
        img_out_dir = f'{output_dir}/images'
        Path(img_out_dir).mkdir(parents=True, exist_ok=True)
        img = cv2.imread(image_filepath)
        class_colors = set([x['class_name'] for x in predictions_dict['prediction_list']])
        class_colors = {_class: (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)) for _class in class_colors}
        for info in predictions_dict['prediction_list']:
            print(info)
            bbox = info['bbox_orig_coords']
            y, x, h, w = list(map(int, bbox))
            cv2.rectangle(img, (x, y), (w, h), class_colors[info['class_name']], 2)
            text = info['class_name']
            cv2.putText(img, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, class_colors[info['class_name']], 1)
        output_image_path = f'{img_out_dir}/{image_name}.png'
        cv2.imwrite(output_image_path, img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='DocParser Demoscript'
    )
    parser.add_argument('--page', action='store_true',
                        help='Run and evaluate DocParser default models for page structure parsing on arXivDocs-target')
    parser.add_argument('--image_filepath', type=str, help='Path to the image file')
    parser.add_argument('--output_dir', type=str, help='Path for the output predictions')
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

    if not any([args.page]):
        print(parser.print_help())

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.page:
        entity_detector = stage1_entity_detector.EntityDetector()
        entity_detector.init_model(model_log_dir=args.output_dir, default_weights='highlevel_wsft')
        structure_parser = stage2_structure_parser.StructureParser()
        detect_structures_single_page(
            args.image_filepath,
            entity_detector,
            structure_parser,
            args.output_dir,
            True,
            True,
        )
