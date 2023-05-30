# nuScenes dev-kit.
# Code written by Oscar Beijbom and Varun Bankiti, 2019.

# DETECTION_NAMES = ['car', 'truck', 'bus', 'trailer', 'construction_vehicle', 'pedestrian', 'motorcycle', 'bicycle',
#                    'traffic_cone', 'barrier', 'median_strip', 'sound_barrier', 'overpass', 'tunnel', 'street_trees',
#                    'road_sign', 'ramp_sect']

DETECTION_NAMES = ['median_strip', 'sound_barrier', 'overpass', 'tunnel', 'street_trees', 'road_sign', 'ramp_sect']

PRETTY_DETECTION_NAMES = {'car': 'Car',
                          'truck': 'Truck',
                          'bus': 'Bus',
                          'trailer': 'Trailer',
                          'construction_vehicle': 'Constr. Veh.',
                          'pedestrian': 'Pedestrian',
                          'motorcycle': 'Motorcycle',
                          'bicycle': 'Bicycle',
                          'traffic_cone': 'Traffic Cone',
                          'barrier': 'Barrier',
                          'median_strip' : 'median_strip',
                          'sound_barrier' : 'sound_barrier',
                          'overpass' : 'overpass',
                          'tunnel' : 'tunnel',
                          'street_trees' : 'street_trees',
                          'road_sign' : 'road_sign',
                          'ramp_sect' : 'ramp_sect'
                          }

DETECTION_COLORS = {'car': 'C0',
                    'truck': 'C1',
                    'bus': 'C2',
                    'trailer': 'C3',
                    'construction_vehicle': 'C4',
                    'pedestrian': 'C5',
                    'motorcycle': 'C6',
                    'bicycle': 'C7',
                    'traffic_cone': 'C8',
                    'barrier': 'C9',
                    'barrier': 'C0',
                    'median_strip' : 'C1',
                    'sound_barrier' : 'C2',
                    'overpass' : 'C3',
                    'tunnel' : 'C4',
                    'street_trees' : 'C5',
                    'road_sign' : 'C6',
                    'ramp_sect' : 'C7'}

ATTRIBUTE_NAMES = ['pedestrian.moving', 'pedestrian.sitting_lying_down', 'pedestrian.standing', 'cycle.with_rider',
                   'cycle.without_rider', 'vehicle.moving', 'vehicle.parked', 'vehicle.stopped']

PRETTY_ATTRIBUTE_NAMES = {'pedestrian.moving': 'Ped. Moving',
                          'pedestrian.sitting_lying_down': 'Ped. Sitting',
                          'pedestrian.standing': 'Ped. Standing',
                          'cycle.with_rider': 'Cycle w/ Rider',
                          'cycle.without_rider': 'Cycle w/o Rider',
                          'vehicle.moving': 'Veh. Moving',
                          'vehicle.parked': 'Veh. Parked',
                          'vehicle.stopped': 'Veh. Stopped'}

TP_METRICS = ['trans_err', 'scale_err', 'orient_err', 'vel_err', 'attr_err']

PRETTY_TP_METRICS = {'trans_err': 'Trans.', 'scale_err': 'Scale', 'orient_err': 'Orient.', 'vel_err': 'Vel.',
                     'attr_err': 'Attr.'}

TP_METRICS_UNITS = {'trans_err': 'm',
                    'scale_err': '1-IOU',
                    'orient_err': 'rad.',
                    'vel_err': 'm/s',
                    'attr_err': '1-acc.'}
