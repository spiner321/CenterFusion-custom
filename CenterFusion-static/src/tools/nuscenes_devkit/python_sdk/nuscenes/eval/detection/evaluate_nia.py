# nuScenes dev-kit.
# Code written by Holger Caesar & Oscar Beijbom, 2018.

import argparse
import json
import os
import random
import time
from typing import Tuple, Dict, Any

import numpy as np

from nuscenes import NuScenes
from nuscenes.eval.common.config import config_factory
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.utils.data_classes import Box
from nuscenes.eval.common.loaders import load_prediction, load_gt, add_center_dist, filter_eval_boxes
from nuscenes.eval.detection.algo import accumulate, calc_ap, calc_tp
from nuscenes.eval.detection.constants import TP_METRICS
from nuscenes.eval.detection.data_classes import DetectionConfig, DetectionMetrics, DetectionBox, \
    DetectionMetricDataList
from nuscenes.eval.detection.render import summary_plot, class_pr_curve, class_tp_curve, dist_pr_curve, visualize_sample

from pyquaternion import Quaternion

from nuscenes.eval.common.utils import boxes_to_sensor
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points

from matplotlib import pyplot as plt
import open3d
from scipy.spatial.transform import Rotation as R


# CATS = ['car', 'truck', 'bus', 'bicycle', 'motorcycle', 'pedestrian', 'barrier']
CATS = ['median_strip', 'overpass', 'tunnel', 'sound_barrier', 'street_trees', 'ramp_sect', 'road_sign']

class DetectionEval:
    """
    This is the official nuScenes detection evaluation code.
    Results are written to the provided output_dir.

    nuScenes uses the following detection metrics:
    - Mean Average Precision (mAP): Uses center-distance as matching criterion; averaged over distance thresholds.
    - True Positive (TP) metrics: Average of translation, velocity, scale, orientation and attribute errors.
    - nuScenes Detection Score (NDS): The weighted sum of the above.

    Here is an overview of the functions in this method:
    - init: Loads GT annotations and predictions stored in JSON format and filters the boxes.
    - run: Performs evaluation and dumps the metric data to disk.
    - render: Renders various plots and dumps to disk.

    We assume that:
    - Every sample_token is given in the results, although there may be not predictions for that sample.

    Please see https://www.nuscenes.org/object-detection for more details.
    """
    def __init__(self,
                 nusc,
                 config,
                 result_path,
                 ann_path,
                 eval_set,
                 output_dir = None,
                 verbose = True):
        """
        Initialize a DetectionEval object.
        :param nusc: A NuScenes object.
        :param config: A DetectionConfig object.
        :param result_path: Path of the nuScenes JSON result file.
        :param eval_set: The dataset split to evaluate on, e.g. train, val or test.
        :param output_dir: Folder to save plots and results to.
        :param verbose: Whether to print to stdout.
        """
        self.nusc = nusc
        self.result_path = result_path
        self.ann_path = ann_path
        self.eval_set = eval_set
        self.output_dir = output_dir
        self.verbose = verbose
        self.cfg = config

        # Check result file exists.
        assert os.path.exists(result_path), 'Error: The result file does not exist!'

        # Make dirs.
        self.plot_dir = os.path.join(self.output_dir, 'plots')
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.isdir(self.plot_dir):
            os.makedirs(self.plot_dir)

        # Load data.
        if verbose:
            print('Initializing nuScenes detection evaluation')
        self.pred_boxes, self.meta = load_prediction(self.result_path, self.cfg.max_boxes_per_sample, DetectionBox,
                                                     verbose=verbose)

        self.gt_boxes, self.lidar_filenames = self.load_gts(self.ann_path, DetectionBox)

        # self.gt_boxes = load_gt(self.nusc, self.eval_set, DetectionBox, verbose=verbose)

        assert set(self.pred_boxes.sample_tokens) == set(self.gt_boxes.sample_tokens), \
            "Samples in split doesn't match samples in predictions."

        # Add center distances.
        # self.pred_boxes = add_center_dist(nusc, self.pred_boxes)
        # self.gt_boxes = add_center_dist(nusc, self.gt_boxes)

        # Filter boxes (distance, points per box, etc.).
        # if verbose:
        #     print('Filtering predictions')
        # self.pred_boxes = filter_eval_boxes(nusc, self.pred_boxes, self.cfg.class_range, verbose=verbose)
        # if verbose:
        #     print('Filtering ground truth annotations')
        # self.gt_boxes = filter_eval_boxes(nusc, self.gt_boxes, self.cfg.class_range, verbose=verbose)

        self.sample_tokens = self.gt_boxes.sample_tokens

    def evaluate(self) -> Tuple[DetectionMetrics, DetectionMetricDataList]:
        """
        Performs the actual evaluation.
        :return: A tuple of high-level and the raw metric data.
        """
        start_time = time.time()

        # -----------------------------------
        # Step 1: Accumulate metric data for all classes and distance thresholds.
        # -----------------------------------
        if self.verbose:
            print('Accumulating metric data...')
        metric_data_list = DetectionMetricDataList()
        for class_name in self.cfg.class_names:
            for dist_th in self.cfg.dist_ths:
                md = accumulate(self.gt_boxes, self.pred_boxes, class_name, self.cfg.dist_fcn_callable, dist_th)
                metric_data_list.set(class_name, dist_th, md)

        # -----------------------------------
        # Step 2: Calculate metrics from the data.
        # -----------------------------------
        if self.verbose:
            print('Calculating metrics...')
        metrics = DetectionMetrics(self.cfg)
        for class_name in self.cfg.class_names:
            # Compute APs.
            for dist_th in self.cfg.dist_ths:
                metric_data = metric_data_list[(class_name, dist_th)]
                ap = calc_ap(metric_data, self.cfg.min_recall, self.cfg.min_precision)
                metrics.add_label_ap(class_name, dist_th, ap)

            # Compute TP metrics.
            for metric_name in TP_METRICS:
                metric_data = metric_data_list[(class_name, self.cfg.dist_th_tp)]
                if class_name in ['traffic_cone'] and metric_name in ['attr_err', 'vel_err', 'orient_err']:
                    tp = np.nan
                elif class_name in ['barrier'] and metric_name in ['attr_err', 'vel_err']:
                    tp = np.nan
                else:
                    tp = calc_tp(metric_data, self.cfg.min_recall, metric_name)
                metrics.add_label_tp(class_name, metric_name, tp)

        # Compute evaluation time.
        metrics.add_runtime(time.time() - start_time)

        return metrics, metric_data_list

    def load_gts(self,  ann_path, DetectionBox, verbose=True):
        with open(ann_path) as f:
            data = json.load(f)

        all_annotations = EvalBoxes()
        lidar_filenames = dict()
        for img in data['images']:
            sample_token = img['sample_token']
            id=img['id']
            sample_boxes = []
            lidar_filenames[sample_token] = {'points':img['lidar_filename'],
                                             'rot':img['cs_record_rot'],
                                             'trans':img['cs_record_trans']
                                             }

            for sample_annotation in data['annotations']:
                if sample_annotation['image_id'] != id:
                    continue

                detection_name=CATS[sample_annotation['category_id']-1]
                translation = np.array(sample_annotation['location']) - np.array([0, sample_annotation['dim'][1], 0])
                sample_boxes.append(
                    DetectionBox(
                        sample_token=sample_token,
                        translation=translation[[0, 2, 1]].tolist(),
                        size=[sample_annotation['dim'][1], sample_annotation['dim'][2], sample_annotation['dim'][0]],
                        rotation= Quaternion(axis=[0, 1, 0], angle=sample_annotation['rotation_y']).elements.tolist(),
                        velocity=np.array([0,0]),
                        detection_name=detection_name,
                        detection_score=-1.0,  # GT samples do not have a score.
                        num_pts= sample_annotation['num_lidar_pts'] + sample_annotation['num_radar_pts']
                     )
                )

            all_annotations.add_boxes(sample_token, sample_boxes)

        if verbose:
            print("Loaded ground truth annotations for {} samples.".format(len(all_annotations.sample_tokens)))

        return all_annotations, lidar_filenames

    def render(self, metrics: DetectionMetrics, md_list: DetectionMetricDataList) -> None:
        """
        Renders various PR and TP curves.
        :param metrics: DetectionMetrics instance.
        :param md_list: DetectionMetricDataList instance.
        """
        if self.verbose:
            print('Rendering PR and TP curves')

        def savepath(name):
            return os.path.join(self.plot_dir, name + '.pdf')

        summary_plot(md_list, metrics, min_precision=self.cfg.min_precision, min_recall=self.cfg.min_recall,
                     dist_th_tp=self.cfg.dist_th_tp, savepath=savepath('summary'))

        for detection_name in self.cfg.class_names:
            class_pr_curve(md_list, metrics, detection_name, self.cfg.min_precision, self.cfg.min_recall,
                           savepath=savepath(detection_name + '_pr'))

            class_tp_curve(md_list, metrics, detection_name, self.cfg.min_recall, self.cfg.dist_th_tp,
                           savepath=savepath(detection_name + '_tp'))

        for dist_th in self.cfg.dist_ths:
            dist_pr_curve(md_list, metrics, dist_th, self.cfg.min_precision, self.cfg.min_recall,
                          savepath=savepath('dist_pr_' + str(dist_th)))

    def main(self,
             plot_examples: int = 0,
             render_curves: bool = True) -> Dict[str, Any]:
        """
        Main function that loads the evaluation code, visualizes samples, runs the evaluation and renders stat plots.
        :param plot_examples: How many example visualizations to write to disk.
        :param render_curves: Whether to render PR and TP curves to disk.
        :return: A dict that stores the high-level metrics and meta data.
        """
        if plot_examples > 0:
            # Select a random but fixed subset to plot.
            random.seed(42)
            sample_tokens = list(self.sample_tokens)
            random.shuffle(sample_tokens)
            sample_tokens = sample_tokens[:plot_examples]

            # Visualize samples.
            example_dir = os.path.join(self.output_dir, 'examples')
            if not os.path.isdir(example_dir):
                os.mkdir(example_dir)
            for sample_token in sample_tokens:
                self.visualize_sample(
                                 sample_token,
                                 self.gt_boxes if self.eval_set != 'test' else EvalBoxes(),
                                 # Don't render test GT.
                                 self.pred_boxes,
                                 lidar_filenames=self.lidar_filenames,
                                 eval_range=max(self.cfg.class_range.values()),
                                 savepath=os.path.join(example_dir, '{}.png'.format(sample_token)))

        # Run evaluation.
        metrics, metric_data_list = self.evaluate()

        # Render PR and TP curves.
        if render_curves:
            self.render(metrics, metric_data_list)

        # Dump the metric data, meta and metrics to disk.
        if self.verbose:
            print('Saving metrics to: %s' % self.output_dir)
        metrics_summary = metrics.serialize()
        metrics_summary['meta'] = self.meta.copy()
        with open(os.path.join(self.output_dir, 'metrics_summary.json'), 'w') as f:
            json.dump(metrics_summary, f, indent=2)
        with open(os.path.join(self.output_dir, 'metrics_details.json'), 'w') as f:
            json.dump(metric_data_list.serialize(), f, indent=2)

        # Print high-level metrics.
        print('mAP: %.4f' % (metrics_summary['mean_ap']))
        err_name_mapping = {
            'trans_err': 'mATE',
            'scale_err': 'mASE',
            'orient_err': 'mAOE',
            'vel_err': 'mAVE',
            'attr_err': 'mAAE'
        }
        for tp_name, tp_val in metrics_summary['tp_errors'].items():
            print('%s: %.4f' % (err_name_mapping[tp_name], tp_val))
        print('NDS: %.4f' % (metrics_summary['nd_score']))
        print('Eval time: %.1fs' % metrics_summary['eval_time'])

        # Print per-class metrics.
        print()
        print('Per-class results:')
        print('Object Class\tAP\tATE\tASE\tAOE\tAVE\tAAE')
        class_aps = metrics_summary['mean_dist_aps']
        class_tps = metrics_summary['label_tp_errors']
        for class_name in class_aps.keys():
            print('%s\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
                  % (class_name, class_aps[class_name],
                     class_tps[class_name]['trans_err'],
                     class_tps[class_name]['scale_err'],
                     class_tps[class_name]['orient_err'],
                     class_tps[class_name]['vel_err'],
                     class_tps[class_name]['attr_err']))

        return metrics_summary

    def visualize_sample(self,
                         sample_token: str,
                         gt_boxes: EvalBoxes,
                         pred_boxes: EvalBoxes,
                         lidar_filenames: str,
                         conf_th: float = 0.15,
                         eval_range: float = 50,
                         verbose: bool = True,
                         savepath: str = None) -> None:
        """
        Visualizes a sample from BEV with annotations and detection results.
        :param sample_token: The nuScenes sample token.
        :param gt_boxes: Ground truth boxes grouped by sample.
        :param pred_boxes: Prediction grouped by sample.
        :param nsweeps: Number of sweeps used for lidar visualization.
        :param conf_th: The confidence threshold used to filter negatives.
        :param eval_range: Range in meters beyond which boxes are ignored.
        :param verbose: Whether to print to stdout.
        :param savepath: If given, saves the the rendering here instead of displaying.
        """
        # Retrieve sensor & pose records.
        cs_record = dict()
        cs_record['translation'] = lidar_filenames[sample_token]['trans']
        cs_record['rotation'] = lidar_filenames[sample_token]['rot']

        # Get boxes.
        boxes_gt_global = gt_boxes[sample_token]
        boxes_est_global = pred_boxes[sample_token]

        # Move box to lidar coord system.
        boxes_gt = []
        for box in boxes_gt_global:
            # trans = [box.translation[2], -1.0 * box.translation[0], -1.0 * box.translation[1]]
            box = Box(box.translation, box.size, Quaternion(box.rotation))
            box.rotate(Quaternion(cs_record['rotation']))
            box.translate(np.array(cs_record['translation']))

            boxes_gt.append(box)

        boxes_est = []
        for box in boxes_est_global:
            # trans = [box.translation[2], -1.0 * box.translation[0], -1.0 * box.translation[1]]
            box = Box(box.translation, box.size, Quaternion(box.rotation))
            box.rotate(Quaternion(cs_record['rotation']))
            box.translate(np.array(cs_record['translation']))

            boxes_est.append(box)

        # Add scores to EST boxes.
        for box_est, box_est_global in zip(boxes_est, boxes_est_global):
            box_est.score = box_est_global.detection_score


        # Get point cloud in lidar frame.
        pc = open3d.t.io.read_point_cloud(lidar_filenames[sample_token]['points'])
        points = pc.point["positions"].numpy().transpose(1, 0)

        # Init axes.
        _, ax = plt.subplots(1, 1, figsize=(9, 9))

        # Show point cloud.
        # points = view_points(pc.points[:3, :], np.eye(4), normalize=False)
        # dists = np.sqrt(np.sum(pc.points[:2, :] ** 2, axis=0))
        # colors = np.minimum(1, dists / eval_range)
        # ax.scatter(points[0, :], points[1, :], c=colors, s=0.2)
        # ax.scatter(points[0, :], points[1, :], s=0.2)

        # Show ego vehicle.
        ax.plot(0, 0, 'x', color='black')

        # Show GT boxes.
        for box in boxes_gt:
            box.render(ax, view=np.eye(4), colors=('g', 'g', 'g'), linewidth=2)

        # Show EST boxes.
        for box in boxes_est:
            # Show only predictions with a high score.
            assert not np.isnan(box.score), 'Error: Box score cannot be NaN!'
            if box.score >= conf_th:
                box.render(ax, view=np.eye(4), colors=('b', 'b', 'b'), linewidth=1)

        # Limit visible range.
        axes_limit = eval_range + 3  # Slightly bigger to include boxes that extend beyond the range.
        ax.set_xlim(-axes_limit, axes_limit)
        ax.set_ylim(-axes_limit, axes_limit)

        # Show / save plot.
        if verbose:
            print('Rendering sample token %s' % sample_token)
        plt.title(sample_token)
        if savepath is not None:
            plt.savefig(savepath)
            plt.close()
        else:
            plt.show()


class niaEval(DetectionEval):
    """
    Dummy class for backward-compatibility. Same as DetectionEval.
    """


if __name__ == "__main__":

    # Settings.
    parser = argparse.ArgumentParser(description='Evaluate nuScenes detection results.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--result_path', type=str,
                        default='./exp/ddd/centerfusion/results_nia_det_test_norm.json',
                        help='The submission as a JSON file.')
    parser.add_argument('--ann_path', type=str, default='./data/nia/annotations/test_norm.json',
                        help='annotation as a JSON file.')
    parser.add_argument('--output_dir', type=str, default='./exp/ddd/centerfusion',
                        help='Folder to store result metrics, graphs and example visualizations.')
    parser.add_argument('--eval_set', type=str, default='val',
                        help='Which dataset split to evaluate on, train, val or test.')
    parser.add_argument('--dataroot', type=str, default='./data/nuscenes',
                        help='Default nuScenes data directory.')
    parser.add_argument('--version', type=str, default='v1.0-mini',
                        help='Which version of the nuScenes dataset to evaluate on, e.g. v1.0-trainval.')
    parser.add_argument('--config_path', type=str, default='',
                        help='Path to the configuration file.'
                             'If no path given, the CVPR 2019 configuration will be used.')
    parser.add_argument('--plot_examples', type=int, default=10,
                        help='How many example visualizations to write to disk.')
    parser.add_argument('--render_curves', type=int, default=1,
                        help='Whether to render PR and TP curves to disk.')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Whether to print to stdout.')
    args = parser.parse_args()

    result_path_ = os.path.expanduser(args.result_path)
    ann_path_ = os.path.expanduser(args.ann_path)
    output_dir_ = os.path.expanduser(args.output_dir)
    eval_set_ = args.eval_set
    dataroot_ = args.dataroot
    version_ = args.version
    config_path = args.config_path
    plot_examples_ = args.plot_examples
    render_curves_ = bool(args.render_curves)
    verbose_ = bool(args.verbose)

    if config_path == '':
        cfg_ = config_factory('detection_cvpr_2019')
    else:
        with open(config_path, 'r') as _f:
            cfg_ = DetectionConfig.deserialize(json.load(_f))

    nusc_ = NuScenes(version=version_, verbose=verbose_, dataroot=dataroot_)
    nusc_eval = DetectionEval(nusc_, config=cfg_, result_path=result_path_, ann_path=ann_path_, eval_set=eval_set_,
                              output_dir=output_dir_, verbose=verbose_)
    nusc_eval.main(plot_examples=plot_examples_, render_curves=render_curves_)
