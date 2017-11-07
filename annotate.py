import os
import sys
import yaml
import numpy as np
import cv2
import glob
import re
import argparse
import Tkinter
import tkSimpleDialog

project_python_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print project_python_path
if project_python_path not in sys.path:
    sys.path.insert(0, project_python_path)
import common.io as io

os.chdir(project_python_path)


if '2.' in re.findall(r"\d+\.",cv2.__version__)[0]:
    cv2.LINE_AA = cv2.CV_AA
    cv2.FILLED = -1
    cv2.WINDOW_FULLSCREEN = cv2.cv.CV_WINDOW_FULLSCREEN
   
def _str_to_bool(s):
    """Convert string to bool (in argparse context)."""
    if s.lower() not in ['true', 'false']:
        raise ValueError('Need bool; got %r' % s)
    return {'true': True, 'false': False}[s.lower()]

def parse_arguments():
    usage_text='\nYou can stop at any time using ctrl+c\n'  \
    'example commands:\n'  \
    '\t' + __file__ + ' /media/fatemeh/data/images /media/fatemeh/data/labels'

    parser = argparse.ArgumentParser(usage=usage_text, description='With this tool, you can annotate your data.')
    parser.add_argument('image_dir', help='Image directoyr')
    parser.add_argument('label_dir', help='Existing label directory. If not exist, this is the save_dir.')
    parser.add_argument('-s', '--save_dir', help='Save resulting labels. If not provided use the label_dir.')
    parser.add_argument('-f', '--use_fullscreen', type=_str_to_bool, default=True, help='Visualize in fullscreen mode. [default: True]')
    #parser.add_argument('-y', '--use_yaml_detection_file', type=_str_to_bool, default=False, help='Use existing yaml detection file. At the moment is not used! [default: False]')

    args = parser.parse_args()
    return args

def draw_rectangle_callback(event,x,y,flags,param):
    ''' mouse callback function
    '''
    global ix1, iy1, ix2, iy2, drawing, used_roi, draw_during_zoom, image
    if event == cv2.EVENT_LBUTTONDOWN:
        draw_during_zoom = True
        drawing = True
        ix1,iy1 = x,y
        print("left button down")
    elif event == cv2.EVENT_MOUSEMOVE:
        image_copy = image.copy()
        if draw_during_zoom == True:
            cv2.rectangle(image,(ix1,iy1),(x,y),(0,255,0),0)
            cv2.imshow('image',image)
            image = image_copy.copy()
    elif event == cv2.EVENT_LBUTTONUP:
        draw_during_zoom = False
        drawing = False
        used_roi = False
        ix2, iy2 = x, y
        if ix1 > ix2:
            tmp = ix2
            ix2 = ix1
            ix1 = tmp
        if iy1 > iy2:
            tmp = iy2
            iy2 = iy1
            iy1 = tmp
        print("left button up")
        print (ix1,iy1,ix2,iy2)
        print("drawing: {}, used_roi: {}".format(drawing, used_roi))

def correct_zoom_coordinates(resized_points, roi, resized_roi_size):
    if not len(resized_points):
        return resized_points
    roi_x1,roi_y1,roi_x2,roi_y2 = roi
    roi_width, roi_height = roi_x2-roi_x1, roi_y2-roi_y1
    resized_width, resized_height = resized_roi_size
    if len(resized_points.shape) == 1:
        resized_points = np.reshape(resized_points,(1,-1))
    corrected_points = list()
    for resized_point in resized_points:
        x1,y1,x2,y2 = resized_point
        point_x = map( int, np.r_[x1,x2] * roi_width/float(resized_width) + roi_x1)
        point_y = map( int, np.r_[y1,y2] * roi_height/float(resized_height) +roi_y1)
        corrected_points.append([point_x[0],point_y[0],point_x[1],point_y[1]])
    return np.array(corrected_points).squeeze()

def zoom_coordinates(points, roi, resized_roi_size):
    if not len(points):
        return points
    roi_x1, roi_y1, roi_x2, roi_y2 = roi
    roi_width, roi_height = roi_x2-roi_x1, roi_y2-roi_y1
    resized_width, resized_height = resized_roi_size
    if len(points.shape) == 1:
        points = np.reshape(resized_point, (1, -1))
    zoomed_points = list()
    for resized_point in points:
        x1, y1, x2, y2 = resized_point
        point_x = map(int, (np.r_[x1, x2]- roi_x1)/(roi_width/float(resized_width)))
        point_y = map(int, (np.r_[y1, y2]- roi_y1)/(roi_height/float(resized_height)))
        zoomed_points.append([point_x[0], point_y[0], point_x[1], point_y[1]])
    return np.array(zoomed_points).squeeze()

def add_detection(detections, detection):
    class_id, x1, y1, x2, y2 = detection
    if (x2-x1) == 0 or (iy2-iy1) == 0:
        print("no region is added!")
        return detections
    if not len(detections):
        return np.array([[class_id, x1, y1, x2, y2]], dtype=np.int64)
    return np.vstack((detections,detection))


def draw_rectangle(image, detections, class_ids = [1,2,3], rectangle_colors=[(255,0,0),(0,255,0),(0,0,255)], text_color=(255,255,255)):
    class_id_dic = dict(zip(class_ids, np.arange(1, len(class_ids)+1)))
    rectangle_colors = [(255,0,0),(0,255,0),(0,0,255)]
    for i in xrange(len(class_ids)-3):
        rectangle_colors.append( tuple(np.random.randint(256, size=(3,))) )
    if len(detections):
        if len(detections.shape) == 1:
            detections = np.reshape(detections,(1,-1))
    for id, value in enumerate(detections):
        class_id, x1, y1, x2, y2 = value
        cv2.rectangle(image,(x1,y1),(x2,y2),rectangle_colors[class_id_dic[class_id]-1],0)
        number_of_digits = len(str(id))
        cv2.rectangle(image,(x1,y1),(x1+8+5*(number_of_digits -1),y1+8), (0,0,0), cv2.FILLED)
        cv2.putText(image, str(id), (x1+1,y1+7), cv2.FONT_HERSHEY_SIMPLEX, 0.3, text_color, 1, cv2.LINE_AA, bottomLeftOrigin=False)
    cv2.imshow('image',image)


def get_list_of_images(image_dir, expersion='*'):
    image_paths = glob.glob(os.path.join(image_dir,expersion))
    image_paths.sort()
    total_number_of_images = len(image_paths)
    return image_paths, total_number_of_images


def load_each_image(image_idx, image_paths, total_number_of_images):
    if image_idx >= total_number_of_images or image_idx <= -total_number_of_images -1:
        print("End of image sequence. Start from beginning!")
        image_idx = 0
    image_path = image_paths[image_idx]
    image_name_no_extension = os.path.splitext( os.path.basename(image_path) )[0]
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    image_with_help = np.zeros((image.shape[0]+100, image.shape[1],3), dtype=image.dtype)
    image_with_help[:image.shape[0],:,:] = image.copy()
    number_part_of_image = int(re.findall(r"\d+",image_name_no_extension)[0])
    _ = cv2.putText(image_with_help, "image: {}".format(number_part_of_image), (5, image.shape[0]+25), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255), 1, cv2.LINE_AA, bottomLeftOrigin=False)
    _ = cv2.putText(image_with_help, "(1: car (blue), 2: pedestrian(green), 3:cyclist(red)), (r: remove), (z: zoom, h: home), ".format(number_part_of_image), (5, image.shape[0]+55), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255), 1, cv2.LINE_AA, bottomLeftOrigin=False)
    _ = cv2.putText(image_with_help, "(n or ->:next, p or <-:previous), (space bar: play/pause), (s: skip image), (q/Esc: exit)".format(number_part_of_image), (5, image.shape[0]+85), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255), 1, cv2.LINE_AA, bottomLeftOrigin=False)
    image = image_with_help.copy()

    image_clean = image.copy()
    image_orig = image.copy()
    image_height, image_width, _ = image.shape
    print image_name_no_extension
    return image, image_idx, image_name_no_extension, image_height, image_width, image_orig, image_clean


def save_detections(filename, detections):
    np.savetxt(filename, detections, delimiter=' ',fmt='%d')


def load_detections(label_dir, image_name_no_extension):
    filename = os.path.join(label_dir, '{}.txt'.format(image_name_no_extension) )
    if not os.path.isfile(filename):
        return np.array([], dtype=np.int64)
    try:
        detections = list()
        f = open(filename,'r')
        for r in f.readlines():
           r = r.split()
           detections.append(map(np.int64, map(np.float64,r)))
        detections = np.array(detections)
    except:
        return np.array([], dtype=np.int64)
    if len(detections.shape) == 1:
        detections = detections.reshape((1,-1))
    return detections

def reset_booleans():
    draw_during_zoom = False
    drawing = False # true if mouse is pressed
    used_roi = False
    is_zoomed = False
    is_home = True
    return is_zoomed, is_home, used_roi, drawing, draw_during_zoom


def read_meta_data(meta_data_file):
    image_idx = 0
    meta_data = dict()
    if os.path.isfile(meta_data_file):
        try:
            with open(meta_data_file,'r+') as mdf:
                meta_data = yaml.load(mdf)
                image_idx = meta_data['starting_image']
        except:
                print("The meta file is corropt!. Start from begining of image.")
                meta_data = dict()
                image_idx = 0
    if not meta_data.has_key('class_ids'):
        meta_data['class_ids'] = {1: 'car', 2: 'pedestrian', 3: 'bike'}
    return meta_data, image_idx

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# ~~~~~~~~~~~~~~~~~~~~~~
# input
# ~~~~~~~~~~~~~~~~~~~~~~
args = parse_arguments()
image_dir = args.image_dir
label_dir = args.label_dir
is_fullscreen = args.use_fullscreen
is_yaml = False#args.use_yaml_detection_file
save_dir = label_dir

ix1, iy1, ix2, iy2, image = -1, -1, -1, -1, np.array([],np.uint8)
draw_during_zoom = False
drawing = False # true if mouse is pressed
used_roi = False
is_zoomed = False
is_home = True
is_play = False


meta_data_file = os.path.join(save_dir, 'meta_data.txt')
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)
meta_data, image_idx = read_meta_data(meta_data_file)
class_ids = meta_data['class_ids'].keys()

image_paths, total_number_of_images = get_list_of_images(image_dir, expersion='*')

image, image_idx, image_name_no_extension, image_height, image_width, image_orig, image_clean = load_each_image(image_idx, image_paths, total_number_of_images)

'''
UNDER CONSTRUCTION!!!
====================
If objects are previously detected in images. The detections are save in yaml file. 
For each image, all the detections are stored as a line of the yaml file with the following format:
image_name: class_id,x1,y1,x2,y2, confidence, frequency class_id,x1,y1,x2,y2, confidence, frequency ...
e.g. 
image0000000000.png: 7,1,198,127,312,0.870625555515,1 14,688,213,904,329,0.951634645462,1
image0000000005.png: 6,279,142,632,312,0.682528793812,1 7,6,202,148,314,0.970469713211,1
  14,728,210,949,328,0.980623006821,1
'''
if is_yaml == True:
    #[2:'bicycle', 6:'bus', 7:'car', 14:'motorbike', 15:'person', 19:'train']
    #voc_lablels = io.get_voc_labels()
    #yaml_detections = io.read_detections_from_file(detection_yaml_file)
    #class_ids = io.find_unique_labels_from_detections(yaml_detections) #[2, 6, 7, 14, 15, 19]
    parameters = dict()
    parameters['label_conversion'] = dict(zip([2,6,7,14,15,19],[3,4,1,5,2,6]))
    # if not meta_data.has_key('label_conversion'):
    #     label_conversion = dict(zip([2,6,7,14,15,19],[3,4,1,5,2,6]))
    #     parameters['label_conversion'] = label_conversion
    #     meta_data['class_ids'] = label_conversion.keys()

    detection_yaml_file = '/home/fatemeh/detections/test.yaml'
    io.make_kitti_format_labels(os.path.basename(detection_yaml_file), os.path.dirname(detection_yaml_file), label_dir, **parameters)


detections = load_detections(label_dir, image_name_no_extension)
if len(detections):
    if len(detections.shape) == 1:
        detections = detections.reshape((1,-1))


if is_fullscreen == True:
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
else:
    cv2.namedWindow('image')
    cv2.moveWindow('image',50,50)

cv2.setMouseCallback('image',draw_rectangle_callback)

try:
    while True:
        if is_play == True:
            is_zoomed, is_home, used_roi, drawing, draw_during_zoom = reset_booleans()
            image_idx += 1
            image, image_idx, image_name_no_extension, image_height, image_width, image_orig, image_clean = load_each_image(image_idx, image_paths, total_number_of_images)
            detections = load_detections(label_dir, image_name_no_extension)
            draw_rectangle(image, detections, class_ids)
            k = cv2.waitKey(66) & 0xFF
        else:
            draw_rectangle(image, detections, class_ids)
            k = cv2.waitKey(0) & 0xFF


        if k == ord('z') and drawing == False and used_roi == False and is_zoomed == False:
            print('zoomed')
            is_zoomed = True
            is_home = False
            used_roi = True

            if abs(ix2-ix1) == 0:
                print("Region width is zero. Automatically 100 pixel is added to the region.")
                ix2 = ix1 + 100
            if abs(iy2-iy1) == 0:
                print("Region width is zero. Automatically 100 pixel is added to the region.")
                iy2 = iy1 + 100
            dx = ix2-ix1;dy = iy2-iy1
            if dy<dx:
                resized_width = image_width
                resized_height = int(float(dy)/dx* resized_width)
            else:
                resized_height = image_height
                resized_width = int(float(dx)/dy* resized_height)

            image = image_orig.copy()
            image = image[iy1:iy2,ix1:ix2]
            image = cv2.resize(image,(resized_width,resized_height),interpolation=cv2.INTER_NEAREST)
            image_clean = image.copy()
            roi = (ix1,iy1,ix2,iy2)
            resized_roi_size = (resized_width, resized_height)
            if len(detections):
                detections[:,1:] = zoom_coordinates(detections[:,1:], roi, resized_roi_size)
            print("drawing: {}, used_roi: {}, is_zoomed: {}, is_home: {}".format(drawing, used_roi, is_zoomed, is_home))


        if k == ord('h') and drawing == False and is_home == False:
            print('home')
            is_zoomed = False
            used_roi = True
            is_home = True
            print drawing, used_roi
            image = image_orig.copy()
            image_clean = image.copy()
            if len(detections):
                detections[:,1:] = correct_zoom_coordinates(detections[:,1:], roi, resized_roi_size)
            print("drawing: {}, used_roi: {}, is_home: {}".format(drawing, used_roi, is_home))

        if k == ord('1') and drawing == False and used_roi == False:
            print('add car')
            used_roi = True
            print drawing, used_roi
            detections = add_detection(detections, [1,ix1,iy1,ix2,iy2])
        if k == ord('2') and drawing == False and used_roi == False:
            print('add pedestrian')
            used_roi = True
            detections = add_detection(detections, [2,ix1,iy1,ix2,iy2])
        if k == ord('3') and drawing == False and used_roi == False:
            print('add bike')
            used_roi = True
            detections = add_detection(detections, [3,ix1,iy1,ix2,iy2])

        if k == ord('r') and drawing == False:
            print('remove region')
            used_roi = True
            # https://stackoverflow.com/questions/15522336/text-input-in-tkinter
            root = Tkinter.Tk()
            region_idx = tkSimpleDialog.askstring("Remove region", "enter region number to remove")
            root.destroy()
            try:
                region_idx = int(region_idx)
                # accidentally enter negative value
                region_idx *= np.sign(region_idx)
            except:
                region_idx = -1
            total_number_of_regions = detections.shape[0]
            if len(detections) and region_idx > -1 and region_idx < total_number_of_regions:
                detections = np.delete(detections,(region_idx),axis=0)
                print detections
                image = image_clean.copy()
            else:
                print("Please provide region number, which you want to remove.")

        if (k == ord('s')) and drawing == False:
            used_roi = True
            # https://stackoverflow.com/questions/15522336/text-input-in-tkinter
            root = Tkinter.Tk()
            skip_number = tkSimpleDialog.askstring("Skip images", "enter a number of images to skip")
            root.destroy()
            try:
                skip_number = int(skip_number)
            except:
                skip_number = 0
            print("skip {} images".format(skip_number))
            image_idx += skip_number
            image, image_idx, image_name_no_extension, image_height, image_width, image_orig, image_clean = load_each_image(image_idx, image_paths, total_number_of_images)
            detections = load_detections(label_dir, image_name_no_extension)

        if (k == ord('n') or k == 82 or k == 83) and drawing == False: # 82: top arrow, 83: right arrow
            print('next image')
            if is_zoomed == True:
                is_zoomed, is_home, used_roi, drawing, draw_during_zoom = reset_booleans()
                if len(detections):
                    detections[:,1:] = correct_zoom_coordinates(detections[:,1:], roi, resized_roi_size)

            filename = os.path.join(save_dir, '{}.txt'.format(image_name_no_extension) )
            if len(detections):
                save_detections(filename, detections)
            if not len(detections) and os.path.isfile(filename):
                os.remove(filename)

            image_idx += 1
            image, image_idx, image_name_no_extension, image_height, image_width, image_orig, image_clean = load_each_image(image_idx, image_paths, total_number_of_images)
            detections = load_detections(label_dir, image_name_no_extension)


        if (k == ord('p') or k == 81 or k == 84) and drawing == False: # 81: left arrow, 83: down arrow:
            print('previous image')
            if is_zoomed == True:
                is_zoomed, is_home, is_play, used_roi, drawing, draw_during_zoom = reset_booleans()
                if len(detections):
                    detections[:,1:] = correct_zoom_coordinates(detections[:,1:], roi, resized_roi_size)

            filename = os.path.join(save_dir, '{}.txt'.format(image_name_no_extension) )
            if len(detections):
                save_detections(filename, detections)
            if not len(detections) and os.path.isfile(filename):
                os.remove(filename)

            image_idx -= 1
            image, image_idx, image_name_no_extension, image_height, image_width, image_orig, image_clean = load_each_image(image_idx, image_paths, total_number_of_images)
            detections = load_detections(label_dir, image_name_no_extension)


        if (k == 32) and drawing == False: # 32 is space bar
            if is_play == True:
                print("play movie")
            else:
                print("pause movie")
            is_play = not is_play

        if (k == ord('q') or k == 27):
            filename = os.path.join(save_dir, '{}.txt'.format(image_name_no_extension) )
            if len(detections):
                save_detections(filename, detections)
            if not len(detections) and os.path.isfile(filename):
                os.remove(filename)
            cv2.destroyAllWindows()

            with open(meta_data_file,'w') as mdf:
                meta_data['starting_image'] = image_idx+1
                yaml.dump(meta_data, mdf, default_flow_style=False)
            break
except:
    cv2.destroyAllWindows()
    with open(meta_data_file,'w') as mdf:
        meta_data['starting_image'] = image_idx+1
        yaml.dump(meta_data, mdf, default_flow_style=False)
