Annotation tool
==================
Make a bounding boxes for a specific object.

This is a a simple tool to make a bouding box. To execute this program, run the below command.

```
python annotate.py your_saving_path your_label_saving_path -f false
```

Type `python annotate -h` to get more help option.

With this tool, by selecting a bounding box, you can do several operations based on keywords that you are selecting. Here is the list of keywords:
- `1`: car
- `2`: pedestrain
- `3`: bike
- `z`: zoom
- `h`: home
- arrow keys: to move the image forward or backward
- `s`: to skip several images. Just type the number of images. Negative number means backward jump.
- space bar: play and pause option

For each image, the labels are saved separatly in a txt file, in the location that you provided. The format is as follows:
```
class_id x1 y1 x2 y2

```
class_id is the 1,2,3 at the moment. This can be easily changed in the code. x1 and y1 are top left coordinates. x2 and y2 are bottom right coordinate of a boundin box.


Anytime, you start the program, it starts from next image were you left.


At the moment, classes of car, pedestrian and cyclist are given. You can easily change the bounding box to anything you want.
