# Copyright (c) Xingyi Zhou. All Rights Reserved
'''
nuScenes pre-processing script.
This file convert the nuScenes annotation into COCO format.
'''
import time
import matplotlib.patches as patches
from matplotlib import pyplot as plt
from functools import reduce
from scipy.spatial.transform import Rotation as R
import open3d
from nuscenes.utils.data_classes import Box
import _init_paths

from utils.pointcloud import RadarPointCloudWithVelocity as RadarPointCloud
from utils.ddd_utils import draw_box_3d, unproject_2d_to_3d
from utils.ddd_utils import compute_box_3d, project_to_image, alpha2rot_y

import sys
import os
import json
import numpy as np
import cv2
import copy
import matplotlib.pyplot as plt
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import BoxVisibility, transform_matrix
# from nuScenes_lib.utils_kitti import KittiDB
from nuscenes.eval.detection.utils import category_to_detection_name
from pyquaternion import Quaternion
from tqdm.auto import tqdm
import glob
from pathlib import Path

sys.path.insert(0, os.getcwd())


# from nuScenes_lib.utils_radar import map_pointcloud_to_image

#// ================ PARAMETERS ================
DATA_PATH = '/data/kimgh/CenterFusion-custom/CenterFusion-static/data/sample'
OUT_PATH = DATA_PATH + '/annotations'
subsample = 10

SPLITS = {
    'train': 'train',
    'val': 'val',
    'test_abnormal': 'test_abnormal',
    'test_normal': 'test_normal'
}

DEBUG = False

CATS = ['median_strip', 'overpass', 'tunnel',
        'sound_barrier', 'street_trees', 'ramp_sect', 'road_sign']

SENSOR_ID = {'RADAR_FRONT': 7, 'RADAR_FRONT_LEFT': 9,
             'RADAR_FRONT_RIGHT': 10, 'RADAR_BACK_LEFT': 11,
             'RADAR_BACK_RIGHT': 12,  'LIDAR_TOP': 8,
             'CAM_FRONT': 1, 'CAM_FRONT_RIGHT': 2,
             'CAM_BACK_RIGHT': 3, 'CAM_BACK': 4, 'CAM_BACK_LEFT': 5,
             'CAM_FRONT_LEFT': 6}

USED_SENSOR = ['CAM_FRONT', 'CAM_FRONT_RIGHT',
               'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT',
               'CAM_FRONT_LEFT']

RADARS_FOR_CAMERA = {
    'CAM_FRONT_LEFT':  ["RADAR_FRONT_LEFT", "RADAR_FRONT"],
    'CAM_FRONT_RIGHT': ["RADAR_FRONT_RIGHT", "RADAR_FRONT"],
    'CAM_FRONT':       ["RADAR_FRONT_RIGHT", "RADAR_FRONT_LEFT", "RADAR_FRONT"],
    'CAM_BACK_LEFT':   ["RADAR_BACK_LEFT", "RADAR_FRONT_LEFT"],
    'CAM_BACK_RIGHT':  ["RADAR_BACK_RIGHT", "RADAR_FRONT_RIGHT"],
    'CAM_BACK':        ["RADAR_BACK_RIGHT", "RADAR_BACK_LEFT"]}

ATTRIBUTE_TO_ID = {
    '': 0, 'cycle.with_rider': 1, 'cycle.without_rider': 2,
    'pedestrian.moving': 3, 'pedestrian.standing': 4,
    'pedestrian.sitting_lying_down': 5,
    'vehicle.moving': 6, 'vehicle.parked': 7,
    'vehicle.stopped': 8}

NUM_SWEEPS = 1

suffix1 = '_{}sweeps'.format(NUM_SWEEPS) if NUM_SWEEPS > 1 else ''
OUT_PATH = OUT_PATH + suffix1 + '/'

CAT_IDS = {v: i + 1 for i, v in enumerate(CATS)}

DISTORTION_COEFFI = [[-0.120864, 0.057409], [-0.141913, 0.059090]]


#// ================ FUNCTIONS ================
def _rot_y2alpha(rot_y, x, cx, fx):
    """
    Get rotation_y by alpha + theta - 180
    alpha : Observation angle of object, ranging [-pi..pi]
    x : Object center x to the camera center (x-W/2), in pixels
    rotation_y : Rotation ry around Y-axis in camera coordinates [-pi..pi]
    """
    alpha = rot_y - np.arctan2(x - cx, fx)
    if alpha > np.pi:
        alpha -= 2 * np.pi
    if alpha < -np.pi:
        alpha += 2 * np.pi
    return alpha


def _bbox_inside(box1, box2):
    return box1[0] > box2[0] and box1[0] + box1[2] < box2[0] + box2[2] and \
        box1[1] > box2[1] and box1[1] + box1[3] < box2[1] + box2[3]

def box_select(box_list, cat): # box_list: "3d_box", cat: "category"
    if cat=="MEDIAN_STRIP" or cat=="SOUND_BARRIER":
        dist = []
        for box in box_list:
            dist.append(box["location"][0])
        idx = dist.index(min(dist))
        # # 중앙분리대, 방음벽의 길이를 15m로 고정
        # new_x = box_list[idx]['location'][0] - box_list[idx]['dimension'][2]/2 + 7.5
        # new_l = 15
        # box_list[idx]['location'][0] = new_x
        # box_list[idx]['dimension'][2] = new_l
        return box_list[idx]
    elif cat=='RAMP_SECT':
        dist = []
        for box in box_list:
            dist.append(box["location"][1])
        idx = dist.index(min(dist))
        return box_list[idx]
    elif cat=="OVERPASS" or cat=="TUNNEL":
        width = []
        for box in box_list:
            width.append(box["dimension"][0])
        idx = width.index(max(width))
        return box_list[idx]
    else:
        return box_list[0]

def gen_token(imgpath):
    fp_parts = Path(imgpath).parts
    weather, catcode, delim, fname = fp_parts[-7],fp_parts[-6],fp_parts[-5][0],fp_parts[-1]
    sample_token = '.'.join([weather,catcode,delim,fname])
    return sample_token

def gen_clippaths(data_rt, split_tvte, slicing=None):
    gp = ""
    tmp = split_tvte.split('_')
    if len(tmp)>1:
        gp = f"{data_rt}/{tmp[0]}/source/{tmp[1]}/??/*Clip*"
    else:
        gp = f"{data_rt}/{split_tvte}/source/*normal/??/*Clip*"
    
    ret = None
    if slicing is None:
        ret = sorted(glob.glob(gp))
    elif type(slicing)==int:
        ret = sorted(glob.glob(gp))[:slicing]
    return ret



def main():
    if not os.path.exists(OUT_PATH):
        os.mkdir(OUT_PATH)

    for split in list(SPLITS.values()):
        data_path = DATA_PATH

        out_path = OUT_PATH + '{}.json'.format(split)  # /data/nia/
        categories_info = [{'name': CATS[i], 'id': i + 1}
                           for i in range(len(CATS))]
        ret = {'images': [], 'annotations': [], 'categories': categories_info,
               'videos': [], 'attributes': ATTRIBUTE_TO_ID, 'pointclouds': []}
        num_images = 0
        num_anns = 0
        num_videos = 0
        
        scenes = gen_clippaths(DATA_PATH, split)
        
        for scene_path in tqdm(scenes, desc=split):
            # print(scene_path)

            condition = Path(scene_path).parts[-3]
            class_ind = Path(scene_path).parts[-2]

            scene = os.path.basename(scene_path)
            data_path = scene_path
            Ann_path = scene_path.replace("source", "label")

            Radar_path = os.path.join(data_path, 'Radar', 'RadarFront')
            Img_path = os.path.join(data_path, 'Camera', 'CameraFront', 'blur')
            Calib_path = os.path.join(data_path, 'calib')
            Lidar_path = os.path.join(data_path, 'Lidar')

            scene_name = scene[-8:-3]  # '00050'
            samples = sorted([name[-7:-4] for name in os.listdir(Radar_path) if os.path.isfile(os.path.join(Radar_path, name))])

            # Lidar-camera calibration file
            LC_calib_filename = os.path.join(
                Calib_path, 'Lidar_camera_calib', '2-048_{}_LCC_CF.txt'.format(scene_name))

            # Lidar-radar calibration file
            LR_calib_filename = os.path.join(
                Calib_path, 'Lidar_radar_calib', '2-048_{}_LRC_RF.txt'.format(scene_name))

            # caemra intricsic matrix

            # samples = [samples[i] for i in range(3,20,4)]
            for sample in (samples)[5::subsample]:

                lidar_filename = os.path.join(
                    Lidar_path, '2-048_{}_LR_{}.pcd'.format(scene_name, sample))
                img_filename = os.path.join(
                    Img_path, '2-048_{}_CF_{}.png'.format(scene_name, sample))
                radar_filename = os.path.join(
                    Radar_path, '2-048_{}_RF_{}.pcd'.format(scene_name, sample))

                sample_token = gen_token(img_filename)

                height = cv2.imread(img_filename, cv2.IMREAD_COLOR).shape[0]
                width = cv2.imread(img_filename, cv2.IMREAD_COLOR).shape[1]
                num_images += 1
                num_videos += 1

                # camera to lidar translation and rotation
                calib_file = open(LC_calib_filename, 'r')
                calib_info = calib_file.readlines()

                # camera intrinsic
                camera_intrinsic = calib_info[8].strip().split(',')
                camera_intrinsic = np.array([[camera_intrinsic[0], 0.0, camera_intrinsic[2]],
                                             [0.0, camera_intrinsic[1],
                                                 camera_intrinsic[3]],
                                             [0.0, 0.0, 1.0]]).astype(np.float32)

                calib = np.eye(4, dtype=np.float32)
                calib[:3, :3] = camera_intrinsic
                calib = calib[:3]

                rotvec = calib_info[4].strip().split(',')
                rotvec = [float(x) for x in rotvec]

                # L2C_trans = calib_info[6].strip().split(',')
                # L2C_trans = [float(x) for x in L2C_trans]

                trans = np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]])
                r1 = np.array([[1, 0, 0], [0, np.cos(
                    rotvec[2]), -np.sin(rotvec[2])], [0, np.sin(rotvec[2]), np.cos(rotvec[2])]])
                r2 = np.array([[np.cos(rotvec[1]), 0, np.sin(rotvec[1])], [
                              0, 1, 0], [-np.sin(rotvec[1]), 0, np.cos(rotvec[1])]])
                r3 = np.array([[np.cos(rotvec[0]), -np.sin(rotvec[0]), 0],
                              [np.sin(rotvec[0]), np.cos(rotvec[0]), 0], [0, 0, 1]])
                L2C_rot = reduce(np.matmul, [trans, r1, r2, r3])
                ############################################################################################
                ############################################################################################

                if scene[0] == 'S':
                    L2C_trans = np.array([0.00156798, -0.221076, -0.0847737])
                else:
                    L2C_trans = np.array([0.0278092, -0.149391, -0.253849])

                trans_l2c = np.eye(4)
                trans_l2c[:3, :3] = L2C_rot
                trans_l2c[:3, 3] = L2C_trans
                ############################################################################################
                ############################################################################################

                C2L_rot = L2C_rot.T
                C2L_trans = -1.0 * L2C_trans

                # cs_record_rot = (R.from_matrix(L2C_rot)).as_quat().tolist()
                # cs_record_rot=[cs_record_rot[-1], *cs_record_rot[:-1]]

                cs_record_rot = (R.from_matrix(C2L_rot)).as_quat().tolist()
                cs_record_rot = [cs_record_rot[-1], *cs_record_rot[:-1]]

                ############################################################################################
                ############################################################################################
                # radar to lidar translation matrix
                calib_file = open(LR_calib_filename, 'r')
                calib_info = calib_file.readlines()
                R2L_mat = calib_info[8:-1]

                R2L_transform = []
                for row in R2L_mat:
                    row = row.strip().split(',')
                    row = [float(x) for x in row]
                    R2L_transform.append(row)

                # radar points in camera coordinates
                radar_pcs = open3d.t.io.read_point_cloud(radar_filename)
                points = radar_pcs.point["positions"].numpy().transpose(1, 0)
                points_vel = radar_pcs.point["curvature"].numpy(
                ).transpose(1, 0)
                points_int = radar_pcs.point["intensity"].numpy(
                ).transpose(1, 0)

                points = np.concatenate(
                    (points, np.ones((1, points.shape[1]))))
                points = np.matmul(R2L_transform, points)
                points = points + L2C_trans[:, None]
                points = np.matmul(L2C_rot, points)

                points = np.concatenate((points, points_vel), axis=0)
                ############################################################################################
                ############################################################################################

                # image information in COCO format
                image_info = {'id': num_images,
                              'file_name': img_filename,
                              'calib': calib.tolist(),
                              'video_id': num_videos,
                              # 'frame_id': frame_ids[sensor_name],
                              'sensor_id': 0,
                              'sample_token': sample_token,#'2-048_{}_CF_{}.png'.format(scene_name, sample),
                              # 'trans_matrix': trans_matrix.tolist(),
                              # 'velocity_trans_matrix': velocity_trans_matrix.tolist(),
                              'width': width,
                              'height': height,
                              # nuscenes /convert_eval_format
                              'pose_record_trans': [0, 0, 0],
                              'pose_record_rot': [1, 0, 0, 0],
                              'cs_record_trans': C2L_trans.tolist(),
                              'cs_record_rot': cs_record_rot,
                              'radar_pc': points.tolist(),
                              "lidar_filename": lidar_filename,  # for viz
                              'camera_intrinsic': camera_intrinsic.tolist(),
                              }
                ret['images'].append(image_info)

                ############################################################################################################
                ############################################################################################################
                # TODO: KYS: uncomment to visualize
                # import matplotlib.pyplot as plt
                # imgpath = image_info['file_name']
                # print(imgpath)
                # img = cv2.imread(imgpath)

                # img_vis = plt.imread(imgpath)
                # plt.figure(figsize=(16, 10))
                # imgname = os.path.basename(image_info['file_name'])
                # plt.savefig(f'{DEMO_PATH}/origin_{imgname}.png')
                # plt.clf()
                ############################################################################################################
                ############################################################################################################

                # annotation - annotation in lidar coordinate

                ann_filename = os.path.join(
                    Ann_path, 'result', '2-048_{}_FC_{}.json'.format(scene_name, sample))
                boxes = json.load(open(ann_filename, 'r'))

                anns = []

                for box3d in boxes['annotation']:

                    track_id = box3d['id']

                    det_name = box3d['category'].lower()
                    if det_name == 'etc':
                        det_name = 'barrier'

                    if det_name is None:
                        continue

                    num_anns += 1

                    # for box_orig in box3d['3d_box']: # all boxes
                    box_orig = box_select(box3d['3d_box'], det_name) # select sub_id box

                    # track_id = 100*( abs(box3d['id']) ) + abs( box_orig["sub_id"] )

                    yaw = box_orig['rotation_y']
                    yaw = float(-(yaw)-np.pi/2)

                    center = np.array(box_orig['location']+[1])
                    center = np.dot(trans_l2c, center)[:3]

                    # box.wlh: w/h/l
                    box_orig['dimension'] = [box_orig['dimension'][0],
                                            box_orig['dimension'][2], box_orig['dimension'][1]]

                    # box = Box(center=center, size=box_orig['dimension'], orientation=Quaternion(w=qw, x=qx, y=qy, z=qz))
                    box = Box(center=center, size=box_orig['dimension'], orientation=Quaternion(
                        axis=[0.0, 1.0, 0.0], radians=yaw))

                    # box.wlh = np.array([box.wlh[0], box.wlh[2], box.wlh[1]]) # wlh
                    box.translate(np.array([0, box.wlh[2] / 2, 0]))

                    category_id = CAT_IDS[det_name]

                    ############## amodel_center #####################
                    amodel_center = project_to_image(
                        np.array([box.center[0], box.center[1] - box.wlh[2] / 2, box.center[2]],
                                np.float32).reshape(1, 3), calib)[0]  # 산민이형

                    if amodel_center[0] > 0 and amodel_center[0] < 1920 and amodel_center[1] > 0 and amodel_center[1] < 1200:
                        # convert to normalized image coordinate
                        amodel_center[0] = (
                            amodel_center[0] - calib[0, 2]) / calib[0, 0]
                        amodel_center[1] = (
                            amodel_center[1] - calib[1, 2]) / calib[1, 1]

                        # calculate parameters

                        if scene[0] == 'S':
                            k1, k2 = DISTORTION_COEFFI[0][0], DISTORTION_COEFFI[0][1]
                        else:
                            k1, k2 = DISTORTION_COEFFI[1][0], DISTORTION_COEFFI[1][1]
                        r2 = (amodel_center[0]**2 + amodel_center[1]**2)
                        r4 = r2**2

                        # undistort
                        amodel_center = amodel_center * (1 + k1 * r2 + k2 * r4)

                        # convert back to image coordinate
                        amodel_center[0] = amodel_center[0] * \
                            calib[0, 0] + calib[0, 2]
                        amodel_center[1] = amodel_center[1] * \
                            calib[1, 1] + calib[1, 2]
                        ###################################################
                    else:
                        # print(scene,sample)
                        continue

                    # instance information in COCO format
                    ann = {
                        'id': num_anns,
                        'image_id': num_images,
                        'category_id': category_id,
                        # h w l           #// 병합 대상
                        'dim': [float(box.wlh[2]), float(box.wlh[0]), float(box.wlh[1])],
                        # // 병합 대상
                        'location':  [float(box.center[0]), float(box.center[1]), float(box.center[2])],
                        'depth': box.center[2],  # // 병합 대상
                        'occluded': 0,
                        'truncated': 0,
                        'rotation_y': yaw,  # // 병합 대상
                        'amodel_center': amodel_center.tolist(),  # // 병합 대상
                        'iscrowd': 0,
                        'track_id': track_id,  # // 병합 대상
                        'attributes': 0,
                        'velocity': 0,
                        'velocity_cam': 0,
                        # // 병합 대상
                        'num_lidar_pts': box_orig['lidar_point_count'],
                        # // 병합 대상
                        "num_radar_pts": box_orig['radar_point_count'],
                    }
                    ############################################################################################################
                    ############################################################################################################
                    # TODO: KYS: uncomment to visualize
                #   bbox_vis = copy.copy(box3d['3d_box'][0]['2d_box'])
                #   bbox_vis = [bbox_vis[0] - bbox_vis[2]/2, bbox_vis[1] - bbox_vis[3]/2, bbox_vis[0] + bbox_vis[2]/2, bbox_vis[1] + bbox_vis[3]/2]
                #   plt.plot([bbox_vis[0], bbox_vis[0]], [bbox_vis[1], bbox_vis[3]], linewidth=0.5, c='g')
                #   plt.plot([bbox_vis[2], bbox_vis[2]], [bbox_vis[1], bbox_vis[3]], linewidth=0.5, c='g')
                #   plt.plot([bbox_vis[0], bbox_vis[2]], [bbox_vis[1], bbox_vis[1]], linewidth=0.5, c='g')
                #   plt.plot([bbox_vis[2], bbox_vis[0]], [bbox_vis[3], bbox_vis[3]], linewidth=0.5, c='g')
                #   plt.scatter(amodel_center[0],height-amodel_center[1])
                #   imgname = os.path.basename(image_info['file_name'])
                #   plt.savefig(f'{DEMO_PATH}/plot_{imgname}.png')
                #   plt.clf()

                    ##################################################################
                    ######################### considering distortion #################
                    # get image coordinate point (using given extrinsic, intrinsic)
                    # box_3d = compute_box_3d(ann['dim'], ann['location'], ann['rotation_y'])
                    # box_2d = project_to_image(box_3d, calib)

                    # if DISTORTION:
                    # # convert to normalized image coordinate
                    #   box_2d[:, 0] = (box_2d[:, 0] - calib[0, 2]) / calib[0, 0]
                    #   box_2d[:, 1] = (box_2d[:, 1] - calib[1, 2]) / calib[1, 1]

                    #   # calculate parameters
                    #   if scene[0] == 'S':
                    #     k1, k2 = DISTORTION_COEFFI[0][0], DISTORTION_COEFFI[0][1]
                    #   else:
                    #     k1, k2 = DISTORTION_COEFFI[1][0], DISTORTION_COEFFI[1][1]
                    #   r2 = (box_2d[:, 0]**2 + box_2d[:, 1]**2)[:, None]
                    #   r4 = r2**2

                    #   # undistort
                    #   box_2d = box_2d * (1 + k1 * r2 + k2 * r4)

                    #   # convert back to image coordinate
                    #   box_2d[:, 0] = box_2d[:, 0] * calib[0, 0] + calib[0, 2]
                    #   box_2d[:, 1] = box_2d[:, 1] * calib[1, 1] + calib[1, 2]
                    #   ##################################################################
                    #   ##################################################################

                    # plt.scatter(box_2d[:, 0], box_2d[:, 1], s=1, c='r')
                    # plt.plot(box_2d[[0, 1], 0], box_2d[[0, 1], 1], linewidth=0.5, c='b')
                    # plt.plot(box_2d[[0, 4], 0], box_2d[[0, 4], 1], linewidth=0.5, c='b')
                    # plt.plot(box_2d[[5, 1], 0], box_2d[[5, 1], 1], linewidth=0.5, c='b')
                    # plt.plot(box_2d[[5, 4], 0], box_2d[[5, 4], 1], linewidth=0.5, c='b')

                    # plt.plot(box_2d[[2, 3], 0], box_2d[[2, 3], 1], linewidth=0.5, c='r')
                    # plt.plot(box_2d[[2, 6], 0], box_2d[[2, 6], 1], linewidth=0.5, c='r')
                    # plt.plot(box_2d[[7, 3], 0], box_2d[[7, 3], 1], linewidth=0.5, c='r')
                    # plt.plot(box_2d[[7, 6], 0], box_2d[[7, 6], 1], linewidth=0.5, c='r')
                    ############################################################################################################
                    ############################################################################################################

                    # box2d = box3d['3d_box'][0]['2d_box']
                    box2d = box_orig['2d_box']

                    if box2d[2] > 1200:
                        continue
                    bbox = tuple([box2d[0], box2d[1], box2d[0] +
                                box2d[2], box2d[1]+box2d[3]])
                    # bbox = KittiDB.project_kitti_box_to_image(
                    #   copy.deepcopy(box), camera_intrinsic, imsize=(1920, 1200))

                    if bbox == None:
                        continue
                    alpha = _rot_y2alpha(yaw, box2d[0],
                                        camera_intrinsic[0, 2], camera_intrinsic[0, 0])
                    # alpha = _rot_y2alpha(yaw, (bbox[0] + bbox[2]) / 2,
                    #                      camera_intrinsic[0, 2], camera_intrinsic[0, 0])

                    ann['bbox'] = [box2d[0]-box2d[2]/2,
                                box2d[1]-box2d[3]/2, box2d[2], box2d[3]]
                    ann['area'] = (box3d['3d_box'][0]['2d_area'])
                    ann['alpha'] = alpha
                    anns.append(ann)

                ############################################################################################################
                ############################################################################################################
                # TODO: KYS: uncomment to visualize
                # plt.axis('off')
                # plt.savefig(f"{DEMO_PATH}/{str(DISTORTION)}_{str(scene)}_{str(sample)}_distortion.png")
                # plt.clf()
                ############################################################################################################
                ############################################################################################################

                # plt.savefig('save.png')
                # Filter out bounding boxes outside the image
                visable_anns = []
                for i in range(len(anns)):
                    vis = True
                    for j in range(len(anns)):
                        if anns[i]['depth'] - min(anns[i]['dim']) / 2 > \
                           anns[j]['depth'] + max(anns[j]['dim']) / 2 and \
                           _bbox_inside(anns[i]['bbox'], anns[j]['bbox']):
                            vis = False

                            break
                    if vis:
                        visable_anns.append(anns[i])
                    else:
                        pass

                for ann in visable_anns:
                    ret['annotations'].append(ann)

                # if DEBUG:
                #   img_path = data_path + image_info['file_name']
                #   img = cv2.imread(img_path)
                #   img_3d = img.copy()
                #   # plot radar point clouds
                #   pc = np.array(image_info['pc_3d'])
                #   cam_intrinsic = np.array(image_info['calib'])[:,:3]
                #   points, coloring, _ = map_pointcloud_to_image(pc, cam_intrinsic)
                #   for i, p in enumerate(points.T):
                #     img = cv2.circle(img, (int(p[0]), int(p[1])), 5, (255,0,0), -1)

                #   for ann in visable_anns:
                #     bbox = ann['bbox']
                #     cv2.rectangle(img, (int(bbox[0]), int(bbox[1])),
                #                   (int(bbox[2] + bbox[0]), int(bbox[3] + bbox[1])),
                #                   (0, 0, 255), 3, lineType=cv2.LINE_AA)
                #     box_3d = compute_box_3d(ann['dim'], ann['location'], ann['rotation_y'])
                #     box_2d = project_to_image(box_3d, calib)
                #     img_3d = draw_box_3d(img_3d, box_2d)

                #     pt_3d = unproject_2d_to_3d(ann['amodel_center'], ann['depth'], calib)
                #     pt_3d[1] += ann['dim'][0] / 2
                #     print('location', ann['location'])
                #     print('loc model', pt_3d)
                #     pt_2d = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2],
                #                       dtype=np.float32)
                #     pt_3d = unproject_2d_to_3d(pt_2d, ann['depth'], calib)
                #     pt_3d[1] += ann['dim'][0] / 2
                #     print('loc      ', pt_3d)
                #   # cv2.imshow('img', img)
                #   # cv2.imshow('img_3d', img_3d)
                #   # cv2.waitKey()

                #   cv2.imwrite('img.jpg', img)
                #   cv2.imwrite('img_3d.jpg', img_3d)
                #   nusc.render_sample_data(image_token, out_path='nusc_img.jpg')
                #   input('press enter to continue')
                #   # plt.show()

        print('reordering images')
        images = ret['images']
        video_sensor_to_images = {}
        for image_info in images:
            tmp_seq_id = image_info['video_id'] * 20 + image_info['sensor_id']
            if tmp_seq_id in video_sensor_to_images:
                video_sensor_to_images[tmp_seq_id].append(image_info)
            else:
                video_sensor_to_images[tmp_seq_id] = [image_info]
        ret['images'] = []
        for tmp_seq_id in sorted(video_sensor_to_images):
            ret['images'] = ret['images'] + video_sensor_to_images[tmp_seq_id]

        print('{} {} images {} boxes'.format(
            split, len(ret['images']), len(ret['annotations'])))
        print('out_path', out_path)
        json.dump(ret, open(out_path, 'w'))


if __name__ == '__main__':
    main()