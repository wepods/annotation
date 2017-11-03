import os
import numpy as np
import matplotlib.pylab as plt
import yaml
import cv2
import glob


str2arr = lambda label, x1, y1, x2, y2, conf, freq: [int(label), int(x1), int(y1), int(x2), int(y2), float(conf),

                                                     int(freq)]
def read_detections_from_file(detection_file):
    with open(detection_file, 'r') as f:
        data = yaml.load(f)

    image_names = data.keys()
    for image_name in image_names:
        if len(data[image_name]):
            bboxs = data[image_name].split(' ')
            dets = list()
            for bbox in bboxs:
                dets.append(str2arr(*bbox.split(',')))
            data[image_name] = np.array(dets)
    return data


arr2str = lambda a: ','.join(map(str, map(int, a[:5])) + [str(a[5])] + [str(int(a[6]))])


def write_detections_to_file(detection_file, result):
    data = result.copy()
    image_names = data.keys()
    for image_name in image_names:
        dets = data[image_name]
        if len(data[image_name]):
            dets_str = str()
            if len(np.shape(dets)) == 1:
                dets = dets[np.newaxis, :]
            for det in dets:
                tmp = arr2str(det) + ' '
                dets_str = dets_str + tmp
            dets_str = dets_str[:-1]
            data[image_name] = dets_str
        else:
            data[image_name] = str()

    with open(detection_file, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)


def get_voc_labels():
    labels_only_name = ['background',
                        'aeroplane', 'bicycle', 'bird', 'boat',
                        'bottle', 'bus', 'car', 'cat', 'chair',
                        'cow', 'diningtable', 'dog', 'horse',
                        'motorbike', 'person', 'plant',
                        'sheep', 'sofa', 'train', 'tvmonitor']
    label_names = dict(zip(np.arange(len(labels_only_name)), labels_only_name))
    return label_names


def find_unique_labels(*args):
    if not len(args):
        raise IOError('detections are empty')
    labels = list()
    for arg in args:
        labels.extend(np.unique(arg[:, 0]))
    return np.unique(labels)


def find_unique_labels_from_detections(detections):
    im_names = detections.keys()
    labels = list()
    for im_name in im_names:
        if len(detections[im_name]):
            labels.extend(np.unique(detections[im_name][:, 0]))
    return list(np.int64(np.unique(labels)))


def make_kitti_format_labels(det_filename, det_dir, save_dir, **kwargs):
    """Generate kitti format labels 
    
    Input
    =====
    - det_filename: detection filenane    
    - det_dir : detection directory
    - save_dir: save directory
    - [optional]label_conversion = dict(zip([2,6,7,14,15],[1,2,3,4,5])). If not
    given the original labels are used.
    
    Output
    ======
    Save all label files in save_dir. Per image one label file is generated. 
    There is a detection per line in the form of "label_id x1 y1 x2 y2", where 
    label_id is the integer, x1,y1,x2,y2 are bounding boxes.
    
    """
    det_data = read_detections_from_file(os.path.join(det_dir, det_filename))
    labels = find_unique_labels_from_detections(det_data)

    if kwargs.has_key('label_conversion'):
        label_conversion = kwargs['label_conversion']
    else:
        label_conversion = dict(zip(labels,labels))
    
    keys = label_conversion.keys()
    for label in labels:
        if label not in keys:
            raise ImportError("no conversion is given for {} label".format(label))

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    im_names = det_data.keys()
    for im_name in im_names:
        if len(det_data[im_name]):
            with open(os.path.join(save_dir, im_name[:-3] + 'txt'), 'w') as f:
                for i in det_data[im_name]:
                    s = '{:d} {:.2f} {:.2f} {:.2f} {:.2f}\n'.format(label_conversion.get(int(i[0])), *out_of_boundry_correction(*i[1:5]))
                    f.write(s)


def make_kitti_format_labels_for_all_dataset(det_filename, det_main_dir, save_main_dir, pattern='*E*', **kwargs):
    """Save kitti type labels for all dataset (image folders). For more details 
    check :func:`make_kitti_format_labels`
    
    Input
    =====
    det_filename: 
        detection filename
    det_main_dir : 
        detection main directory where all detection folders are located
    det_main_save: 
        main saving directory, where per dataset labels are recorded. 
    pattern:
        default is '*E*'
    [optional]label_conversion :
    If not given the original labels are used. e.g.::
    
        dict(zip([2,6,7,14,15],[1,2,3,4,5]))
    
        
    example
    =======
    example::
    
        label_conversion = dict(zip([2,6,7,14,15],[1,2,3,4,5]))
        make_kitti_format_labels_for_all_dataset(det_filename, det_main_dir, save_main_dir, label_conversion)
    """
    if not os.path.exists(save_main_dir):
        os.mkdir(save_main_dir)

    #data_folders = os.listdir(det_main_dir)
    data_paths = glob.glob('{}/{}'.format(det_main_dir, pattern))
    data_folders = map(os.path.basename, data_paths)
    data_folders.sort()
    for data_folder in data_folders:
        det_dir = os.path.join(det_main_dir, data_folder)
        save_dir = os.path.join(save_main_dir, data_folder)
        make_kitti_format_labels(det_filename, det_dir, save_dir,**kwargs)

