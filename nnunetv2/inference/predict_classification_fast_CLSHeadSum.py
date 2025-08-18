"""
Extended nnU-Net predictor that handles both segmentation and classification using encoder features
Modified from nnunetv2.inference.predict_from_raw_data

This creates a new entry point: nnUNetv2_predict_with_classification
"""

# Standard library
import os
import json
import time
import itertools
from json import load, dump
from os.path import join
from queue import Queue
from threading import Thread
from typing import Optional, List, Tuple, Union

# Third-party libraries
import torch
from torch import nn
import numpy as np
from tqdm import tqdm
from batchgenerators.utilities.file_and_folder_operations import (
    load_json, join as bg_join, isfile, maybe_mkdir_p, isdir, subdirs, save_json
)
from acvl_utils.cropping_and_padding.padding import pad_nd_image

# nnU-Net specific
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.utilities.utils import create_lists_from_splitted_dataset_folder
from nnunetv2.inference.export_prediction import export_prediction_from_logits
from nnunetv2.inference.sliding_window_prediction import (
    compute_gaussian, compute_steps_for_sliding_window
)
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from nnunetv2.training.nnUNetTrainer.variants.multi_task.optional_nnUNet_arch.nnUNetWithFeatures import nnUNetWithFeatures
from nnunetv2.preprocessing.resampling.default_resampling import compute_new_shape
from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager

class Classifier3D(nn.Module):
    """3D Classifier for multi-scale nnUNet features"""
    def __init__(self, num_classes, encoder_channels=[320, 320, 256, 128, 64, 32]):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc_layers = nn.ModuleList([
            nn.Linear(ch, num_classes) for ch in encoder_channels
        ])

    def forward(self, *features):
        outputs = []
        for feat, fc in zip(features, self.fc_layers):
            pooled = self.avgpool(feat).flatten(1)
            outputs.append(fc(pooled))
        return sum(outputs)  # sum logits from all scales


class FastPreprocessor(DefaultPreprocessor):
    def run_case_npy(self, data: np.ndarray, seg: Union[np.ndarray, None], properties: dict,
                     plans_manager: PlansManager, configuration_manager: ConfigurationManager,
                     dataset_json: Union[dict, str]):
        # let's not mess up the inputs!


        start_time = time.time()
        data = np.copy(data)
        end_time = time.time()
        print(f"\t Time for copy data: {end_time - start_time} seconds")

        if seg is not None:
            seg = np.copy(seg)

        has_seg = seg is not None

        # apply transpose_forward, this also needs to be applied to the spacing!
        data = data.transpose([0, *[i + 1 for i in plans_manager.transpose_forward]])
        if seg is not None:
            seg = seg.transpose([0, *[i + 1 for i in plans_manager.transpose_forward]])
        original_spacing = [properties['spacing'][i] for i in plans_manager.transpose_forward]

        # crop, remember to store size before cropping!
        shape_before_cropping = data.shape[1:]
        properties['shape_before_cropping'] = shape_before_cropping
        # this command will generate a segmentation. This is important because of the nonzero mask which we may need
        # data, seg, bbox = crop_to_nonzero(data, seg)
        if seg is None:
            seg = np.empty((1,) + data.shape[1:], dtype=np.int8)  # 创建一个与data形状相同的空numpy数组，但只有一个通道并且类型为int8
        data, seg = data, seg
        bbox = [[0, s] for s in data.shape[1:]]

        properties['bbox_used_for_cropping'] = bbox
        # print(data.shape, seg.shape)
        properties['shape_after_cropping_and_before_resampling'] = data.shape[1:]
       
        # resample
        target_spacing = configuration_manager.spacing  # this should already be transposed

        if len(target_spacing) < len(data.shape[1:]):
            # target spacing for 2d has 2 entries but the data and original_spacing have three because everything is 3d
            # in 3d we do not change the spacing between slices
            target_spacing = [original_spacing[0]] + target_spacing
        new_shape = compute_new_shape(data.shape[1:], original_spacing, target_spacing)

        # normalize
        # normalization MUST happen before resampling or we get huge problems with resampled nonzero masks no
        # longer fitting the images perfectly!
        start_time = time.time()
        data = self._normalize(data, seg, configuration_manager,
                               plans_manager.foreground_intensity_properties_per_channel)
        end_time = time.time()
        print(f"\t Time for normalize: {end_time - start_time} seconds")

        old_shape = data.shape[1:]
        start_time = time.time()
        # data = configuration_manager.resampling_fn_data(data, new_shape, original_spacing, target_spacing)
        data = torch.from_numpy(data)
        data = data.unsqueeze(0)
        data = torch.nn.functional.interpolate(data, tuple(new_shape), mode='trilinear', align_corners=False)
        data = data.squeeze(0).numpy()
        end_time = time.time()
        print(f"\t Time for resample image: {end_time - start_time} seconds")

        # seg = configuration_manager.resampling_fn_seg(seg, new_shape, original_spacing, target_spacing)
        if self.verbose:
            print(f'old shape: {old_shape}, new_shape: {new_shape}, old_spacing: {original_spacing}, '
                  f'new_spacing: {target_spacing}, fn_data: {configuration_manager.resampling_fn_data}')

        # if we have a segmentation, sample foreground locations for oversampling and add those to properties
        if has_seg:
            # reinstantiating LabelManager for each case is not ideal. We could replace the dataset_json argument
            # with a LabelManager Instance in this function because that's all its used for. Dunno what's better.
            # LabelManager is pretty light computation-wise.
            label_manager = plans_manager.get_label_manager(dataset_json)
            collect_for_this = label_manager.foreground_regions if label_manager.has_regions \
                else label_manager.foreground_labels

            # when using the ignore label we want to sample only from annotated regions. Therefore we also need to
            # collect samples uniformly from all classes (incl background)
            if label_manager.has_ignore_label:
                collect_for_this.append(label_manager.all_labels)

            # no need to filter background in regions because it is already filtered in handle_labels
            # print(all_labels, regions)
            properties['class_locations'] = self._sample_foreground_locations(seg, collect_for_this,
                                                                                   verbose=self.verbose)
            seg = self.modify_seg_fn(seg, plans_manager, dataset_json, configuration_manager)
        if np.max(seg) > 127:
            seg = seg.astype(np.int16)
        else:
            seg = seg.astype(np.int8)
        
        return data, seg, properties


class nnUNetPredictorWithClassification(nnUNetPredictor):
    """Extended nnU-Net predictor that also performs classification using encoder features"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cls_head = None
        self.mt_num_classes = None
        self.cls_output_folder = None
        
    def initialize_from_trained_model_folder_with_classification(
            self,
            model_training_output_dir: str,
            use_folds: Optional[Union[Tuple[Union[int, str], ...], None]],
            checkpoint_name: str = 'checkpoint_final.pth',
    ):
        """
        Initialize both segmentation and classification models
        """
        # Initialize segmentation model first
        self.initialize_from_trained_model_folder(
            model_training_output_dir, use_folds, checkpoint_name
        )
        
        # Load classification head from first available fold
        if use_folds is not None and len(use_folds) > 0:
            fold_to_use = use_folds[0]
        else:
            # Find available folds
            available_folds = [f for f in os.listdir(model_training_output_dir) if f.startswith('fold_')]
            if available_folds:
                fold_to_use = available_folds[0].replace('fold_', '')
            else:
                print("Warning: No folds found for classification head")
                return
        
        self.checkpoint_name = checkpoint_name
        self._load_classification_head(model_training_output_dir, fold_to_use)
        
    def _load_classification_head(self, model_folder: str, fold: Union[int, str]):
        """Load the classification head from checkpoint"""
        checkpoint_path = join(model_folder, f"fold_{fold}", self.checkpoint_name)
        
        if not os.path.exists(checkpoint_path):
            print(f"Warning: Classification checkpoint not found: {checkpoint_path}")
            print("Classification will be skipped")
            return
            
        try:
            print(f"Loading classification head from: {checkpoint_path}")
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Classification will be skipped")
            return
        
        # Extract classification parameters
        self.mt_num_classes = ckpt.get("mt_num_classes", 3)
        cls_state = ckpt.get("cls_head_state", None)
        
        if cls_state is None:
            print("Warning: No classification head state found in checkpoint")
            print("Classification will be skipped")
            return
            
        print(f"Classification setup: {self.mt_num_classes} classes (single label)")
        
        self.cls_head = Classifier3D(self.mt_num_classes).to(self.device)

        self.cls_head.load_state_dict(cls_state, strict=True)
                
        # Move to device AFTER loading state dict
        self.cls_head.to(self.device)
        self.cls_head.eval()
        # Verify the weights are different from initialization
        print("Verifying classification head weights:")
        for name, param in self.cls_head.named_parameters():
            print(f"  {name}: mean={param.data.mean().item():.6f}, std={param.data.std().item():.6f}")
        print("Classification head loaded successfully!")

    def set_cls_output_folder(self, cls_output_folder: str):
        """Set the output folder for classification results"""
        self.cls_output_folder = cls_output_folder
        maybe_mkdir_p(cls_output_folder)

    def save_classification_results(self, case_identifier: str, cls_results: dict):
        """Save classification results to JSON file"""
        if self.cls_output_folder is not None and cls_results is not None:
            output_file = join(self.cls_output_folder, f"{case_identifier}_classification.json")
            save_json(cls_results, output_file)
    
    @torch.inference_mode()
    def _internal_maybe_mirror_and_predict_with_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Modified version that returns both prediction and encoder features
        """
        mirror_axes = self.allowed_mirroring_axes if self.use_mirroring else None
        
        # Get prediction and features from network (original orientation only for features)
        prediction = self.network(x)
        # features = self.network.encoder(x)[-1]

        encoder_features = self.network.encoder(x)  # list of encoder stage outputs

        # Get bottleneck encoder feature
        decoder_features = []
        bottleneck_feature = encoder_features[-1]
        y = bottleneck_feature
        for i, stage in enumerate(self.network.decoder.stages):

            y = self.network.decoder.transpconvs[i](y)
            y = torch.cat((y, encoder_features[-(i+2)]), 1) #NOTE??
            y = stage(y)
            decoder_features.append(y)

        # Get some decoder features
        dec_feature1 = decoder_features[0]
        dec_feature2 = decoder_features[1]  # e.g., final stage before output  
        dec_feature3 = decoder_features[2]
        dec_feature4 = decoder_features[3]
        dec_feature5 = decoder_features[4]

        cls_logits = self.cls_head(
            bottleneck_feature,
            dec_feature1,
            dec_feature2,
            dec_feature3,
            dec_feature4,
            dec_feature5)

        
        # For TTA, only augment predictions, keep original features
        if mirror_axes is not None:
            assert max(mirror_axes) <= x.ndim - 3, 'mirror_axes does not match the dimension of the input!'

            mirror_axes = [m + 2 for m in mirror_axes]
            axes_combinations = [
                c for i in range(len(mirror_axes)) for c in itertools.combinations(mirror_axes, i + 1)
            ]
            
            for axes in axes_combinations:
                flipped_x = torch.flip(x, axes)
                flipped_output = self.network(flipped_x)
                
                if isinstance(flipped_output, (list, tuple)):
                    flipped_pred = flipped_output[0]
                else:
                    flipped_pred = flipped_output
                
                prediction += torch.flip(flipped_pred, axes)
            
            prediction /= (len(axes_combinations) + 1)
            # Keep original features unchanged
        
        return prediction, cls_logits
        
    @torch.inference_mode()
    def _internal_predict_sliding_window_return_logits_and_features(self,
                                                       data: torch.Tensor,
                                                       slicers,
                                                       do_on_device: bool = True,
                                                       ):
        """
        Modified version that returns both logits and features - FIXED VERSION
        """
        predicted_logits = n_predictions = prediction = gaussian = workon = None
        results_device = self.device if do_on_device else torch.device('cpu')
        
        # Store all features from patches to average them properly
        cls_logits = [] 

        def producer(d, slh, q):
            for s in slh:
                q.put((torch.clone(d[s][None], memory_format=torch.contiguous_format).to(self.device), s))
            q.put('end')

        try:
            empty_cache(self.device)

            # move data to device
            if self.verbose:
                print(f'move image to device {results_device}')
            data = data.to(results_device)
            queue = Queue(maxsize=2)
            t = Thread(target=producer, args=(data, slicers, queue))
            t.start()

            # preallocate arrays
            if self.verbose:
                print(f'preallocating results arrays on device {results_device}')
            predicted_logits = torch.zeros((self.label_manager.num_segmentation_heads, *data.shape[1:]),
                                           dtype=torch.half,
                                           device=results_device)
            n_predictions = torch.zeros(data.shape[1:], dtype=torch.half, device=results_device)

            if self.use_gaussian:
                gaussian = compute_gaussian(tuple(self.configuration_manager.patch_size), sigma_scale=1. / 8,
                                            value_scaling_factor=10,
                                            device=results_device)
            else:
                gaussian = 1

            if not self.allow_tqdm and self.verbose:
                print(f'running prediction: {len(slicers)} steps')

            with tqdm(desc=None, total=len(slicers), disable=not self.allow_tqdm) as pbar:
                while True:
                    item = queue.get()
                    if item == 'end':
                        queue.task_done()
                        break
                    workon, sl = item
                    
                    # Get both prediction and features
                    prediction, cls_logit= self._internal_maybe_mirror_and_predict_with_features(workon)
                    prediction = prediction[0].to(results_device)  # Take first output if multiple

                    if self.use_gaussian:
                        prediction *= gaussian
                    predicted_logits[sl] += prediction
                    n_predictions[sl[1:]] += gaussian

                    cls_logits.append(cls_logit.clone().to(results_device))
                    
                    
                    queue.task_done()
                    pbar.update()
            queue.join()

            # Average logits
            torch.div(predicted_logits, n_predictions, out=predicted_logits)
            
            cls_logits = torch.stack(cls_logits).mean(dim=0)
            # # FIXED: Average all features
            # if cls_logits:
            #     cls_logits = torch.stack(cls_logits).mean(dim=0)
            # else:
            #     cls_logits = None


            # check for infs
            if torch.any(torch.isinf(predicted_logits)):
                raise RuntimeError('Encountered inf in predicted array. Aborting... If this problem persists, '
                                   'reduce value_scaling_factor in compute_gaussian or increase the dtype of '
                                   'predicted_logits to fp32')
        except Exception as e:
            del predicted_logits, n_predictions, prediction, gaussian, workon, cls_logits
            empty_cache(self.device)
            empty_cache(results_device)
            raise e
        
        return predicted_logits, cls_logits

    @torch.inference_mode()
    def predict_sliding_window_return_logits_and_features(self, input_image: torch.Tensor) \
            -> Union[Tuple[np.ndarray, np.ndarray], Tuple[torch.Tensor, torch.Tensor]]:
        """
        Modified version that returns both logits and features
        """
        assert isinstance(input_image, torch.Tensor)
        self.network = self.network.to(self.device)
        self.network.eval()

        empty_cache(self.device)

        with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            assert input_image.ndim == 4, 'input_image must be a 4D np.ndarray or torch.Tensor (c, x, y, z)'

            if self.verbose:
                print(f'Input shape: {input_image.shape}')
                print("step_size:", self.tile_step_size)
                print("mirror_axes:", self.allowed_mirroring_axes if self.use_mirroring else None)

            # if input_image is smaller than tile_size we need to pad it to tile_size.
            data, slicer_revert_padding = pad_nd_image(input_image, self.configuration_manager.patch_size,
                                                       'constant', {'value': 0}, True,
                                                       None)

            slicers = self._internal_get_sliding_window_slicers(data.shape[1:])

            if self.perform_everything_on_device and self.device != 'cpu':
                try:
                    predicted_logits, cls_logits = self._internal_predict_sliding_window_return_logits_and_features(data, slicers,
                                                                                           self.perform_everything_on_device)
                except RuntimeError:
                    print(
                        'Prediction on device was unsuccessful, probably due to a lack of memory. Moving results arrays to CPU')
                    empty_cache(self.device)
                    predicted_logits, cls_logits = self._internal_predict_sliding_window_return_logits_and_features(data, slicers, False)
            else:
                predicted_logits, cls_logits = self._internal_predict_sliding_window_return_logits_and_features(data, slicers,
                                                                                       self.perform_everything_on_device)
            empty_cache(self.device)
            # revert padding
            predicted_logits = predicted_logits[(slice(None), *slicer_revert_padding[1:])]
            
        return predicted_logits, cls_logits

    @torch.inference_mode()
    def predict_logits_from_preprocessed_data(self, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Modified version that returns both logits and encoder features - FIXED VERSION
        """
        n_threads = torch.get_num_threads()
        torch.set_num_threads(self.configuration_manager.default_num_processes if hasattr(self.configuration_manager, 'default_num_processes') else 3)
        prediction = None
        features = None

        for i, params in enumerate(self.list_of_parameters):
            # Load model parameters
            if not isinstance(self.network, torch._dynamo.OptimizedModule):
                self.network.load_state_dict(params)
            else:
                self.network._orig_mod.load_state_dict(params)

            if prediction is None:
                predicted_logits, cls_logits = self.predict_sliding_window_return_logits_and_features(data)
                print(f'Fold {i}: predicted_logits shape: {predicted_logits.size()}')
                print(f'Fold {i}: predicted_cls_logits shape: {cls_logits.size()}')
                prediction = predicted_logits.to('cpu')
                cls_logits= cls_logits.to('cpu') if cls_logits is not None else None
            else:
                predicted_logits, cls_logits= self.predict_sliding_window_return_logits_and_features(data)
                prediction += predicted_logits.to('cpu')
                # FIXED: Only average features if we're actually using multiple folds
                # For classification, we typically only use the first fold's features
                if i == 0:  # Keep features from first fold only
                    cls_logits = cls_logits.to('cpu')

        if len(self.list_of_parameters) > 1:
            prediction /= len(self.list_of_parameters)
            # Don't average features - use first fold's features for classification
            
        print('Final prediction shape:', prediction.size())
        print('cls shape:', cls_logits.size() if cls_logits is not None else None)

        if self.verbose: print('Prediction done')
        torch.set_num_threads(n_threads)
        return prediction, cls_logits   

    def predict_logits_and_features_from_preprocessed_data(self, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Use the new method to get both segmentation logits and encoder features
        """
        seg_logits, cls_logits = self.predict_logits_from_preprocessed_data(data)
        
        return seg_logits, cls_logits

    def predict_from_files_sequential(self,
                           list_of_lists_or_source_folder: Union[str, List[List[str]]],
                           output_folder_or_list_of_truncated_output_files: Union[str, None, List[str]],
                           save_probabilities: bool = False,
                           overwrite: bool = True,
                           folder_with_segs_from_prev_stage: str = None):
        """
        Modified sequential prediction that handles classification using encoder features.
        """
        if isinstance(output_folder_or_list_of_truncated_output_files, str):
            output_folder = output_folder_or_list_of_truncated_output_files
        elif isinstance(output_folder_or_list_of_truncated_output_files, list):
            output_folder = os.path.dirname(output_folder_or_list_of_truncated_output_files[0])
            if len(output_folder) == 0:
                output_folder = os.path.curdir
        else:
            output_folder = None

        # Store input arguments
        if output_folder is not None:
            import inspect
            from copy import deepcopy
            from nnunetv2.utilities.json_export import recursive_fix_for_json_export
            
            my_init_kwargs = {}
            for k in inspect.signature(self.predict_from_files_sequential).parameters.keys():
                my_init_kwargs[k] = locals()[k]
            my_init_kwargs = deepcopy(my_init_kwargs)
            recursive_fix_for_json_export(my_init_kwargs)
            save_json(my_init_kwargs, join(output_folder, 'predict_from_raw_data_args.json'))

            save_json(self.dataset_json, join(output_folder, 'dataset.json'), sort_keys=False)
            save_json(self.plans_manager.plans, join(output_folder, 'plans.json'), sort_keys=False)

        # Check cascaded network requirements
        if self.configuration_manager.previous_stage_name is not None:
            assert folder_with_segs_from_prev_stage is not None, \
                f'The requested configuration is a cascaded network. It requires the segmentations of the previous ' \
                f'stage ({self.configuration_manager.previous_stage_name}) as input. Please provide the folder where' \
                f' they are located via folder_with_segs_from_prev_stage'

        # Sort out input and output filenames
        list_of_lists_or_source_folder, output_filename_truncated, seg_from_prev_stage_files = \
            self._manage_input_and_output_lists(list_of_lists_or_source_folder,
                                                output_folder_or_list_of_truncated_output_files,
                                                folder_with_segs_from_prev_stage, overwrite, 0, 1,
                                                save_probabilities)
        if len(list_of_lists_or_source_folder) == 0:
            return

        # Initialize preprocessor
        label_manager = self.plans_manager.get_label_manager(self.dataset_json)
        preprocessor = FastPreprocessor(verbose=self.verbose)
        
        if output_filename_truncated is None:
            output_filename_truncated = [None] * len(list_of_lists_or_source_folder)
        if seg_from_prev_stage_files is None:
            seg_from_prev_stage_files = [None] * len(list_of_lists_or_source_folder)

        ret = []
        for case_idx, (li, of, sps) in enumerate(zip(list_of_lists_or_source_folder, output_filename_truncated, seg_from_prev_stage_files)):
            print(f"\n=== Processing case {case_idx + 1}/{len(list_of_lists_or_source_folder)}: {os.path.basename(of) if of else 'Unknown'} ===")
            
            # Preprocess
            data, seg, data_properties = preprocessor.run_case(
                li,
                sps,
                self.plans_manager,
                self.configuration_manager,
                self.dataset_json
            )

            if self.verbose:
                print(f'perform_everything_on_device: {self.perform_everything_on_device}')

            # Predict segmentation AND get encoder features
            seg_logits, cls_logits = self.predict_logits_and_features_from_preprocessed_data(
                torch.from_numpy(data)
            )

            case_name = os.path.basename(of) if of else f"case_{case_idx}"
            print(f"Case: {case_name}")

            # Handle batch dimension
            if cls_logits.ndim > 1:
                cls_logits = cls_logits.squeeze(0)
            
            # Single label classification
            probs = torch.softmax(cls_logits, dim=0).tolist()
            pred_label = int(torch.argmax(cls_logits).item())
            
            print(f"  Classification logits: {[f'{x:.4f}' for x in cls_logits.tolist()]}")
            print(f"  Classification probs: {[f'{x:.4f}' for x in probs]}")
            print(f"  Predicted class: {pred_label}")
            
            cls_results = {
                "logits": [float(x) for x in cls_logits.tolist()],
                "probs": probs,
                "pred": pred_label,
                "num_classes": self.mt_num_classes
            }



            # Export segmentation
            if of is not None:
                export_prediction_from_logits(seg_logits, data_properties, self.configuration_manager, self.plans_manager,
                  self.dataset_json, of, save_probabilities)
                
                # Save classification results
                if cls_results is not None:
                    case_identifier = os.path.basename(of)
                    self.save_classification_results(case_identifier, cls_results)
                    
            else:
                from nnunetv2.inference.export_prediction import convert_predicted_logits_to_segmentation_with_correct_shape
                ret.append((convert_predicted_logits_to_segmentation_with_correct_shape(seg_logits, self.plans_manager,
                     self.configuration_manager, self.label_manager,
                     data_properties,
                     save_probabilities), cls_results))
            
            print("-" * 50)

        from nnunetv2.inference.sliding_window_prediction import compute_gaussian
        from nnunetv2.utilities.helpers import empty_cache
        compute_gaussian.cache_clear()
        empty_cache(self.device)
        return ret


def predict_entry_point_with_classification():
    """
    Entry point for nnUNetv2_predict_with_classification command
    """
    import argparse
    from nnunetv2.paths import nnUNet_results
    from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
    
    parser = argparse.ArgumentParser(
        description="Extended nnU-Net prediction with classification support using encoder features"
    )
    parser.add_argument('-i', type=str, required=False, default= "/home/mengwei/Downloads/nnUNet_code/data/nnUNet_raw/Dataset001_3DCT/imagesTr",
                       help='input folder. Remember to use the correct channel numberings for your files (_0000 etc). '
                             'File endings must be the same as the training dataset!')
    parser.add_argument('-o', type=str, required=False, default = "/home/mengwei/Downloads/nnUNet_code/data/nnUNet_results_sum_lr1e-3/segVal_fast",
                       help='Output folder. If it does not exist it will be created. Predicted segmentations will '
                             'have the same name as their source images.')
    parser.add_argument('-co', '--cls_output_folder', type=str, default="/home/mengwei/Downloads/nnUNet_code/data/nnUNet_results_sum_lr1e-3/segVal_fast",
                       help='Output folder for classification JSONs (default: output_folder/classification)')
    parser.add_argument('-d', type=str, required=False, default="Dataset001_3DCT",
                       help='Dataset with which you would like to predict. You can specify either dataset name or id')
    parser.add_argument('-p', type=str, required=False, default='nnUNetPlans',
                       help='Plans identifier. Default: nnUNetPlans')
    parser.add_argument('-tr', type=str, required=False, default='nnUNetTrainer_CLSHeadSum',
                       help='What nnU-Net trainer class was used for training? Default: nnUNetTrainer')
    parser.add_argument('-c', type=str, required=False, default="3d_fullres",
                       help='nnU-Net configuration that should be used for prediction')
    parser.add_argument('-f', nargs='+', type=str, required=False, default="0",
                       help='Specify the folds of the trained model that should be used for prediction. '
                             'Default: (0, 1, 2, 3, 4)')
    parser.add_argument('-step_size', type=float, required=False, default=0.5,
                       help='Step size for sliding window prediction. Default: 0.5')
    parser.add_argument('--disable_tta', action='store_true', required=False, default=False,
                       help='Set this flag to disable test time data augmentation')
    parser.add_argument('--verbose', action='store_true', help="Set this if you like being talked to")
    parser.add_argument('--save_probabilities', action='store_true',default=True,
                       help='Set this to export predicted class "probabilities"')
    parser.add_argument('--continue_prediction', action='store_true',
                       help='Continue an aborted previous prediction (will not overwrite existing files)')
    parser.add_argument('-chk', type=str, required=False, default='checkpoint_best.pth',
                       help='Name of the checkpoint you want to use. Default: checkpoint_final.pth')
    parser.add_argument('-prev_stage_predictions', type=str, required=False, default=None,
                       help='Folder containing the predictions of the previous stage. Required for cascaded models.')
    parser.add_argument('-device', type=str, default='cuda', required=False,
                       help="Use this to set the device the inference should run with. Available options are 'cuda' "
                             "(GPU), 'cpu' (CPU) and 'mps' (Apple M1/M2)")
    parser.add_argument('--disable_progress_bar', action='store_true', required=False, default=False,
                       help='Set this flag to disable progress bar')
    
    args = parser.parse_args()
    args.f = ['0']
    args.f = [i if i == 'all' else int(i) for i in args.f]

    start_time = time.time()
    
    # Set up paths
    model_folder = join(nnUNet_results, maybe_convert_to_dataset_name(args.d),
                       f"{args.tr}__{args.p}__{args.c}")

    if not isdir(args.o):
        maybe_mkdir_p(args.o)

    # Set up classification output folder
    if args.cls_output_folder is None:
        args.cls_output_folder = join(args.o, 'classification')
    if not isdir(args.cls_output_folder):
        maybe_mkdir_p(args.cls_output_folder)

    assert args.device in ['cpu', 'cuda', 'mps'], \
        f'-device must be either cpu, mps or cuda. Got: {args.device}.'
    
    if args.device == 'cpu':
        import multiprocessing
        torch.set_num_threads(multiprocessing.cpu_count())
        device = torch.device('cpu')
    elif args.device == 'cuda':
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        device = torch.device('cuda')
    else:
        device = torch.device('mps')

    # Create predictor
    predictor = nnUNetPredictorWithClassification(
        tile_step_size=args.step_size,
        use_gaussian=True,
        use_mirroring=not args.disable_tta,
        perform_everything_on_device=True,
        device=device,
        verbose=args.verbose,
        verbose_preprocessing=args.verbose,
        allow_tqdm=not args.disable_progress_bar
    )
    
    # Initialize with classification support
    predictor.initialize_from_trained_model_folder_with_classification(
        model_folder,
        args.f,
        checkpoint_name=args.chk
    )
    
    # Set classification output folder
    predictor.set_cls_output_folder(args.cls_output_folder)
    
    print(f"Input folder: {args.i}")
    print(f"Output folder: {args.o}")
    print(f"Classification output folder: {args.cls_output_folder}")
    print(f"Model folder: {model_folder}")
    print(f"Device: {device}")
    print(f"Classification head available: {predictor.cls_head is not None}")
    
    print("Running in sequential mode with classification support using encoder features")
    predictor.predict_from_files_sequential(
        args.i, 
        args.o, 
        save_probabilities=args.save_probabilities,
        overwrite=not args.continue_prediction,
        folder_with_segs_from_prev_stage=args.prev_stage_predictions
    )
    
    print("Prediction completed!")
    if predictor.cls_head is not None:
        print(f"Classification results saved to: {args.cls_output_folder}")
    print(f"Segmentation results saved to: {args.o}")

    end_time = time.time()
    print(f"\nTotal time for the whole prediction procedure: {end_time - start_time} seconds")  


if __name__ == '__main__':
    # import sys
    # sys.argv = [
    #     'predict_classification_fast_CLSHeadSum.py',
    #     '-i', '/nfs/home/mwei/nnUNet_3d_data/nnUNet_raw/Dataset001_3DCT/imagesVal',
    #     '-o', '/nfs/home/mwei/nnUNet_3d_data/nnUNet_results_sum_nopretrain/segreTr_fast',
    #     '-co', '/nfs/home/mwei/nnUNet_3d_data/nnUNet_results_sum_nopretrain/segreTr_fast',
    #     '-d', 'Dataset001_3DCT',
    #     '-c', '3d_fullres',
    #     '-tr', 'nnUNetTrainer_CLSHeadSum',
    #     '-p', 'nnUNetPlans',
    #     '-f', '0',
    #     '-chk', 'checkpoint_best.pth',
    #     '--save_probabilities'
    # ]

    predict_entry_point_with_classification()