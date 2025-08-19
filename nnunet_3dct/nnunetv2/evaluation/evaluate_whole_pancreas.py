#!/usr/bin/env python3
"""
Standalone script for whole pancreas evaluation
Save this as evaluate_whole_pancreas.py and run with python
"""

import sys
import os


from nnunetv2.evaluation.evaluate_predictions import (
    compute_tp_fp_fn_tn, save_summary_json, load_json,
    recursive_fix_for_json_export, default_num_processes
)
from nnunetv2.imageio.reader_writer_registry import determine_reader_writer_from_dataset_json
from batchgenerators.utilities.file_and_folder_operations import subfiles, join, isfile
import multiprocessing
import numpy as np
from typing import Union
from nnunetv2.imageio.base_reader_writer import BaseReaderWriter


def compute_metrics_whole_pancreas(reference_file: str, prediction_file: str, image_reader_writer: BaseReaderWriter,
                                   ignore_label: int = None) -> dict:
    """
    Compute metrics for whole pancreas (combining all labels > 0)
    """
    # load images
    seg_ref, seg_ref_dict = image_reader_writer.read_seg(reference_file)
    seg_pred, seg_pred_dict = image_reader_writer.read_seg(prediction_file)

    ignore_mask = seg_ref == ignore_label if ignore_label is not None else None

    results = {}
    results['reference_file'] = reference_file
    results['prediction_file'] = prediction_file
    results['metrics'] = {}
    
    # Create whole pancreas masks (any label > 0)
    mask_ref = (seg_ref > 0).astype(bool)
    mask_pred = (seg_pred > 0).astype(bool)
    
    # Compute metrics for whole pancreas
    tp, fp, fn, tn = compute_tp_fp_fn_tn(mask_ref, mask_pred, ignore_mask)
    
    whole_pancreas_key = 'whole_pancreas'
    results['metrics'][whole_pancreas_key] = {}
    
    if tp + fp + fn == 0:
        results['metrics'][whole_pancreas_key]['Dice'] = np.nan
        results['metrics'][whole_pancreas_key]['IoU'] = np.nan
    else:
        results['metrics'][whole_pancreas_key]['Dice'] = 2 * tp / (2 * tp + fp + fn)
        results['metrics'][whole_pancreas_key]['IoU'] = tp / (tp + fp + fn)
    
    results['metrics'][whole_pancreas_key]['FP'] = fp
    results['metrics'][whole_pancreas_key]['TP'] = tp
    results['metrics'][whole_pancreas_key]['FN'] = fn
    results['metrics'][whole_pancreas_key]['TN'] = tn
    results['metrics'][whole_pancreas_key]['n_pred'] = fp + tp
    results['metrics'][whole_pancreas_key]['n_ref'] = fn + tp
    
    return results


def compute_metrics_on_folder_whole_pancreas(folder_ref: str, folder_pred: str, output_file: str,
                                            image_reader_writer: BaseReaderWriter,
                                            file_ending: str,
                                            ignore_label: int = None,
                                            num_processes: int = default_num_processes,
                                            chill: bool = True) -> dict:
    """
    Compute metrics for whole pancreas (all labels > 0 combined) on a folder of segmentations
    """
    if output_file is not None:
        assert output_file.endswith('.json'), 'output_file should end with .json'
    
    files_pred = subfiles(folder_pred, suffix=file_ending, join=False)
    files_ref = subfiles(folder_ref, suffix=file_ending, join=False)
    
    if not chill:
        present = [isfile(join(folder_pred, i)) for i in files_ref]
        assert all(present), "Not all files in folder_ref exist in folder_pred"
    
    files_ref = [join(folder_ref, i) for i in files_pred]
    files_pred = [join(folder_pred, i) for i in files_pred]
    
    with multiprocessing.get_context("spawn").Pool(num_processes) as pool:
        results = pool.starmap(
            compute_metrics_whole_pancreas,
            list(zip(files_ref, files_pred, [image_reader_writer] * len(files_pred),
                     [ignore_label] * len(files_pred)))
        )

    # mean metric for whole pancreas
    whole_pancreas_key = 'whole_pancreas'
    metric_list = list(results[0]['metrics'][whole_pancreas_key].keys())
    means = {}
    means[whole_pancreas_key] = {}
    
    for m in metric_list:
        means[whole_pancreas_key][m] = np.nanmean([i['metrics'][whole_pancreas_key][m] for i in results])

    # For whole pancreas, the foreground_mean is the same as the mean
    foreground_mean = means[whole_pancreas_key].copy()

    [recursive_fix_for_json_export(i) for i in results]
    recursive_fix_for_json_export(means)
    recursive_fix_for_json_export(foreground_mean)
    
    result = {'metric_per_case': results, 'mean': means, 'foreground_mean': foreground_mean}
    
    if output_file is not None:
        save_summary_json(result, output_file)
    
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate whole pancreas segmentation (all labels > 0)')
    parser.add_argument('gt_folder', type=str, help='folder with gt segmentations')
    parser.add_argument('pred_folder', type=str, help='folder with predicted segmentations')
    parser.add_argument('-djfile', type=str, required=True, help='dataset.json file')
    parser.add_argument('-o', type=str, required=False, default=None,
                        help='Output file. Optional. Default: pred_folder/whole_pancreas_summary.json')
    parser.add_argument('-np', type=int, required=False, default=default_num_processes,
                        help=f'number of processes used. Optional. Default: {default_num_processes}')
    parser.add_argument('--chill', action='store_true', 
                        help='dont crash if folder_pred does not have all files that are present in folder_gt')
    
    args = parser.parse_args()
    
    # Load dataset configuration
    dataset_json = load_json(args.djfile)
    file_ending = dataset_json['file_ending']
    
    # Get reader writer class
    example_file = subfiles(args.gt_folder, suffix=file_ending, join=True)[0]
    rw = determine_reader_writer_from_dataset_json(dataset_json, example_file)()
    
    # Set output file
    if args.o is None:
        output_file = join(args.pred_folder, 'whole_pancreas_summary.json')
    else:
        output_file = args.o
    
    # Run evaluation
    print(f"Evaluating whole pancreas...")
    print(f"GT folder: {args.gt_folder}")
    print(f"Pred folder: {args.pred_folder}")
    print(f"Output file: {output_file}")
    
    result = compute_metrics_on_folder_whole_pancreas(
        args.gt_folder, args.pred_folder, output_file, rw, file_ending,
        ignore_label=None, num_processes=args.np, chill=args.chill
    )
    
    print(f"Whole pancreas Dice: {result['foreground_mean']['Dice']:.4f}")
    print(f"Results saved to: {output_file}")


if __name__ == '__main__':
    main()