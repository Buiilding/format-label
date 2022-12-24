1. CVAT
CVAT is used for labeling data for solving computer vision tasks such as:

Image Classification
Object Detection
Object Tracking
Image Segmentation
Pose Estimation

Of course, apart from annotating bounding boxes for object detection, CVAT also allows you to annotate your data for image classification, semantic segmentation, instance segmentation, and object tracking

2. COCO
COCO is large scale images with Common Objects in Context (COCO) for object detection, segmentation, and captioning data set. COCO has 1.5 million object instances for 80 object categories

COCO has 5 annotation types used for

object detection
keypoint detection
stuff segmentation
panoptic segmentation
image captioning
COCO stores annotations in a JSON file. Let’s look at the JSON format for storing the annotation details for the bounding box. This will help to create your own data set using the COCO format.

The basic building blocks for the JSON annotation file is

info: contains high-level information about the dataset.
licenses: contains a list of image licenses that apply to images in the dataset.
categories: contains a list of categories. Categories can belong to a supercategory
images: contains all the image information in the dataset without bounding box or segmentation information. image ids need to be unique
annotations: list of every individual object annotation from every image in the dataset

3.Pascal
Pascal VOC provides standardized image data sets for object detection

Difference between COCO and Pacal VOC data formats will quickly help understand the two data formats

Pascal VOC is an XML file, unlike COCO which has a JSON file.
In Pascal VOC we create a file for each of the image in the dataset. In COCO we have one file each, for entire dataset for training, testing and validation.
The bounding Box in Pascal VOC and COCO data formats are different
COCO Bounding box: (x-top left, y-top left, width, height)

Pascal VOC Bounding box :(xmin-top left, ymin-top left,xmax-bottom right, ymax-bottom right)

Folder:
Folder that contains the images

Filename:
Name of the physical file that exists in the folder

Size:
Contain the size of the image in terms of width, height and depth. If the image is black and white then the depth will be 1. For color images, depth will be 3

Object:
Contains the object details. If you have multiple annotations then the object tag with its contents is repeated. The components of the object tags are

name
pose
truncated
difficult
bndbox
name:
This is the name of the object that we are trying to identify

truncated:
Indicates that the bounding box specified for the object does not correspond to the full extent of the object. For example, if an object is visible partially in the image then we set truncated to 1. If the object is fully visible then set truncated to 0

difficult:
An object is marked as difficult when the object is considered difficult to recognize. If the object is difficult to recognize then we set difficult to 1 else set it to 0

bounding box:
Axis-aligned rectangle specifying the extent of the object visible in the image.

4. YOLO
YOLO: In the YOLO labeling format, a .txt file with the same name is created for each image file in the same directory. Each .txt file contains the annotations for the corresponding image file, including its object class, object coordinates, height, and width.

Conclusion:

-
# format-label
CVAT:
<!--lint disable list-item-indent-->
<!--lint disable list-item-spacing-->
<!--lint disable emphasis-marker-->
<!--lint disable maximum-line-length-->
<!--lint disable list-item-spacing-->

# Dataset and annotation formats

## Contents

- [How to add a format](#how-to-add)
- [Format descriptions](#formats)
  - [CVAT](#cvat)
  - [Datumaro](#datumaro)
  - [LabelMe](#labelme)
  - [MOT](#mot)
  - [MOTS](#mots)
  - [COCO](#coco)
  - [PASCAL VOC and mask](#voc)
  - [YOLO](#yolo)

  

## How to add a new annotation format support<a id="how-to-add"></a>

1. Add a python script to `dataset_manager/formats`
1. Add an import statement to [registry.py](./registry.py).
1. Implement some importers and exporters as the format requires.

Each format is supported by an importer and exporter.

It can be a function or a class decorated with
`importer` or `exporter` from [registry.py](./registry.py). Examples:

```python
@importer(name="MyFormat", version="1.0", ext="ZIP")
def my_importer(file_object, task_data, **options):
  ...

@importer(name="MyFormat", version="2.0", ext="XML")
class my_importer(file_object, task_data, **options):
  def __call__(self, file_object, task_data, **options):
    ...

@exporter(name="MyFormat", version="1.0", ext="ZIP"):
def my_exporter(file_object, task_data, **options):
  ...
```

Each decorator defines format parameters such as:

- _name_

- _version_

- _file extension_. For the `importer` it can be a comma-separated list.
  These parameters are combined to produce a visible name. It can be
  set explicitly by the `display_name` argument.

Importer arguments:

- _file_object_ - a file with annotations or dataset
- _task_data_ - an instance of `TaskData` class.

Exporter arguments:

- _file_object_ - a file for annotations or dataset

- _task_data_ - an instance of `TaskData` class.

- _options_ - format-specific options. `save_images` is the option to
  distinguish if dataset or just annotations are requested.

[`TaskData`](../bindings.py) provides many task properties and interfaces
to add and read task annotations.

Public members:

- **TaskData. Attribute** - class, `namedtuple('Attribute', 'name, value')`

- **TaskData. LabeledShape** - class, `namedtuple('LabeledShape', 'type, frame, label, points, occluded, attributes, group, z_order')`

- **TrackedShape** - `namedtuple('TrackedShape', 'type, points, occluded, frame, attributes, outside, keyframe, z_order')`

- **Track** - class, `namedtuple('Track', 'label, group, shapes')`

- **Tag** - class, `namedtuple('Tag', 'frame, label, attributes, group')`

- **Frame** - class, `namedtuple('Frame', 'frame, name, width, height, labeled_shapes, tags')`

- **TaskData. shapes** - property, an iterator over `LabeledShape` objects

- **TaskData. tracks** - property, an iterator over `Track` objects

- **TaskData. tags** - property, an iterator over `Tag` objects

- **TaskData. meta** - property, a dictionary with task information

- **TaskData. group_by_frame()** - method, returns
  an iterator over `Frame` objects, which groups annotation objects by frame.
  Note that `TrackedShape` s will be represented as `LabeledShape` s.

- **TaskData. add_tag(tag)** - method,
  tag should be an instance of the `Tag` class

- **TaskData. add_shape(shape)** - method,
  shape should be an instance of the `Shape` class

- **TaskData. add_track(track)** - method,
  track should be an instance of the `Track` class

Sample exporter code:

```python
...
# dump meta info if necessary
...
# iterate over all frames
for frame_annotation in task_data.group_by_frame():
  # get frame info
  image_name = frame_annotation.name
  image_width = frame_annotation.width
  image_height = frame_annotation.height
  # iterate over all shapes on the frame
  for shape in frame_annotation.labeled_shapes:
    label = shape.label
    xtl = shape.points[0]
    ytl = shape.points[1]
    xbr = shape.points[2]
    ybr = shape.points[3]
    # iterate over shape attributes
    for attr in shape.attributes:
      attr_name = attr.name
      attr_value = attr.value
...
# dump annotation code
file_object.write(...)
...
```

Sample importer code:

```python
...
#read file_object
...
for parsed_shape in parsed_shapes:
  shape = task_data.LabeledShape(
    type="rectangle",
    points=[0, 0, 100, 100],
    occluded=False,
    attributes=[],
    label="car",
    outside=False,
    frame=99,
  )
task_data.add_shape(shape)
```

## Format specifications<a id="formats" />

### CVAT<a id="cvat" />

This is the native CVAT annotation format. It supports all CVAT annotations
features, so it can be used to make data backups.

- supported annotations: Rectangles, Polygons, Polylines,
  Points, Cuboids, Tags, Tracks

- attributes are supported

- [Format specification](/cvat/apps/documentation/xml_format.md)

#### CVAT for images export

Downloaded file: a ZIP file of the following structure:

```bash
taskname.zip/
├── images/
|   ├── img1.png
|   └── img2.jpg
└── annotations.xml
```

- tracks are split by frames

#### CVAT for videos export

Downloaded file: a ZIP file of the following structure:

```bash
taskname.zip/
├── images/
|   ├── frame_000000.png
|   └── frame_000001.png
└── annotations.xml
```

- shapes are exported as single-frame tracks

#### CVAT loader

Uploaded file: an XML file or a ZIP file of the structures above

### Datumaro format <a id="datumaro" />

[Datumaro](https://github.com/openvinotoolkit/datumaro/) is a tool, which can
help with complex dataset and annotation transformations, format conversions,
dataset statistics, merging, custom formats etc. It is used as a provider
of dataset support in CVAT, so basically, everything possible in CVAT
is possible in Datumaro too, but Datumaro can offer dataset operations.

- supported annotations: any 2D shapes, labels
- supported attributes: any

### [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/)<a id="voc" />

- [Format specification](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/devkit_doc.pdf)

- supported annotations:

  - Rectangles (detection and layout tasks)
  - Tags (action- and classification tasks)
  - Polygons (segmentation task)

- supported attributes:

  - `occluded` (both UI option and a separate attribute)
  - `truncated` and `difficult` (should be defined for labels as `checkbox` -es)
  - action attributes (import only, should be defined as `checkbox` -es)
  - arbitrary attributes (in the `attributes` secion of XML files)

#### Pascal VOC export

Downloaded file: a zip archive of the following structure:

```bash
taskname.zip/
├── JPEGImages/
│   ├── <image_name1>.jpg
│   ├── <image_name2>.jpg
│   └── <image_nameN>.jpg
├── Annotations/
│   ├── <image_name1>.xml
│   ├── <image_name2>.xml
│   └── <image_nameN>.xml
├── ImageSets/
│   └── Main/
│       └── default.txt
└── labelmap.txt

# labelmap.txt
# label : color_rgb : 'body' parts : actions
background:::
aeroplane:::
bicycle:::
bird:::
```

#### Pascal VOC import

Uploaded file: a zip archive of the structure declared above or the following:

```bash
taskname.zip/
├── <image_name1>.xml
├── <image_name2>.xml
└── <image_nameN>.xml
```

It must be possible for CVAT to match the frame name and file name
from annotation `.xml` file (the `filename` tag, e. g.
`<filename>2008_004457.jpg</filename>` ).

There are 2 options:

1. full match between frame name and file name from annotation `.xml`
   (in cases when task was created from images or image archive).

1. match by frame number. File name should be `<number>.jpg`
   or `frame_000000.jpg`. It should be used when task was created from video.

#### Segmentation mask export

Downloaded file: a zip archive of the following structure:

```bash
taskname.zip/
├── labelmap.txt # optional, required for non-VOC labels
├── ImageSets/
│   └── Segmentation/
│       └── default.txt # list of image names without extension
├── SegmentationClass/ # merged class masks
│   ├── image1.png
│   └── image2.png
└── SegmentationObject/ # merged instance masks
    ├── image1.png
    └── image2.png

# labelmap.txt
# label : color (RGB) : 'body' parts : actions
background:0,128,0::
aeroplane:10,10,128::
bicycle:10,128,0::
bird:0,108,128::
boat:108,0,100::
bottle:18,0,8::
bus:12,28,0::
```

Mask is a `png` image with 1 or 3 channels where each pixel
has own color which corresponds to a label.
Colors are generated following to Pascal VOC [algorithm](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/htmldoc/devkit_doc.html#sec:voclabelcolormap).
`(0, 0, 0)` is used for background by default.

- supported shapes: Rectangles, Polygons

#### Segmentation mask import

Uploaded file: a zip archive of the following structure:

```bash
  taskname.zip/
  ├── labelmap.txt # optional, required for non-VOC labels
  ├── ImageSets/
  │   └── Segmentation/
  │       └── <any_subset_name>.txt
  ├── SegmentationClass/
  │   ├── image1.png
  │   └── image2.png
  └── SegmentationObject/
      ├── image1.png
      └── image2.png
```

It is also possible to import grayscale (1-channel) PNG masks.
For grayscale masks provide a list of labels with the number of lines equal
to the maximum color index on images. The lines must be in the right order
so that line index is equal to the color index. Lines can have arbitrary,
but different, colors. If there are gaps in the used color
indices in the annotations, they must be filled with arbitrary dummy labels.
Example:

```
q:0,128,0:: # color index 0
aeroplane:10,10,128:: # color index 1
_dummy2:2,2,2:: # filler for color index 2
_dummy3:3,3,3:: # filler for color index 3
boat:108,0,100:: # color index 3
...
_dummy198:198,198,198:: # filler for color index 198
_dummy199:199,199,199:: # filler for color index 199
...
the last label:12,28,0:: # color index 200
```

- supported shapes: Polygons

#### How to create a task from Pascal VOC dataset

1. Download the Pascal Voc dataset (Can be downloaded from the
   [PASCAL VOC website](http://host.robots.ox.ac.uk/pascal/VOC/))

1. Create a CVAT task with the following labels:

   ```bash
   aeroplane bicycle bird boat bottle bus car cat chair cow diningtable
   dog horse motorbike person pottedplant sheep sofa train tvmonitor
   ```

   You can add `~checkbox=difficult:false ~checkbox=truncated:false`
   attributes for each label if you want to use them.

   Select interesting image files (See [Creating an annotation task](cvat/apps/documentation/user_guide.md#creating-an-annotation-task) guide for details)

1. zip the corresponding annotation files

1. click `Upload annotation` button, choose `Pascal VOC ZIP 1.1`

   and select the zip file with annotations from previous step.
   It may take some time.

### [YOLO](https://pjreddie.com/darknet/yolo/)<a id="yolo" />

- [Format specification](https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects)
- supported annotations: Rectangles

#### YOLO export

Downloaded file: a zip archive with following structure:

```bash
archive.zip/
├── obj.data
├── obj.names
├── obj_<subset>_data
│   ├── image1.txt
│   └── image2.txt
└── train.txt # list of subset image paths

# the only valid subsets are: train, valid
# train.txt and valid.txt:
obj_<subset>_data/image1.jpg
obj_<subset>_data/image2.jpg

# obj.data:
classes = 3 # optional
names = obj.names
train = train.txt
valid = valid.txt # optional
backup = backup/ # optional

# obj.names:
cat
dog
airplane

# image_name.txt:
# label_id - id from obj.names
# cx, cy - relative coordinates of the bbox center
# rw, rh - relative size of the bbox
# label_id cx cy rw rh
1 0.3 0.8 0.1 0.3
2 0.7 0.2 0.3 0.1
```

Each annotation `*.txt` file has a name that corresponds to the name of
the image file (e. g. `frame_000001.txt` is the annotation
for the `frame_000001.jpg` image).
The `*.txt` file structure: each line describes label and bounding box
in the following format `label_id cx cy w h`.
`obj.names` contains the ordered list of label names.

#### YOLO import

Uploaded file: a zip archive of the same structure as above
It must be possible to match the CVAT frame (image name)
and annotation file name. There are 2 options:

1. full match between image name and name of annotation `*.txt` file
   (in cases when a task was created from images or archive of images).

1. match by frame number (if CVAT cannot match by name). File name
   should be in the following format `<number>.jpg` .
   It should be used when task was created from a video.

#### How to create a task from YOLO formatted dataset (from VOC for example)

1. Follow the official [guide](https://pjreddie.com/darknet/yolo/)(see Training YOLO on VOC section)
   and prepare the YOLO formatted annotation files.

1. Zip train images

```bash
zip images.zip -j -@ < train.txt
```

1. Create a CVAT task with the following labels:

   ```bash
   aeroplane bicycle bird boat bottle bus car cat chair cow diningtable dog
   horse motorbike person pottedplant sheep sofa train tvmonitor
   ```

   Select images. zip as data. Most likely you should use `share`
   functionality because size of images. zip is more than 500Mb.
   See [Creating an annotation task](cvat/apps/documentation/user_guide.md#creating-an-annotation-task)
   guide for details.

1. Create `obj.names` with the following content:

   ```bash
   aeroplane
   bicycle
   bird
   boat
   bottle
   bus
   car
   cat
   chair
   cow
   diningtable
   dog
   horse
   motorbike
   person
   pottedplant
   sheep
   sofa
   train
   tvmonitor
   ```

1. Zip all label files together (we need to add only label files that correspond to the train subset)

   ```bash
   cat train.txt | while read p; do echo ${p%/*/*}/labels/${${p##*/}%%.*}.txt; done | zip labels.zip -j -@ obj.names
   ```

1. Click `Upload annotation` button, choose `YOLO 1.1` and select the zip

   file with labels from the previous step.

### [MS COCO Object Detection](http://cocodataset.org/#format-data)<a id="coco" />

- [Format specification](http://cocodataset.org/#format-data)

#### COCO export

Downloaded file: a zip archive with following structure:

```bash
archive.zip/
├── images/
│   ├── <image_name1.ext>
│   ├── <image_name2.ext>
│   └── ...
└── annotations/
    └── instances_default.json
```

- supported annotations: Polygons, Rectangles
- supported attributes:
  - `is_crowd` (checkbox or integer with values 0 and 1) -
    specifies that the instance (an object group) should have an
    RLE-encoded mask in the `segmentation` field. All the grouped shapes
    are merged into a single mask, the largest one defines all
    the object properties
  - `score` (number) - the annotation `score` field
  - arbitrary attributes - will be stored in the `attributes` annotation section


*Note*: there is also a [support for COCO keypoints over Datumaro](https://github.com/openvinotoolkit/cvat/issues/2910#issuecomment-726077582)

1. Install [Datumaro](https://github.com/openvinotoolkit/datumaro)
  `pip install datumaro`
1. Export the task in the `Datumaro` format, unzip
1. Export the Datumaro project in `coco` / `coco_person_keypoints` formats
  `datum export -f coco -p path/to/project [-- --save-images]`

This way, one can export CVAT points as single keypoints or
keypoint lists (without the `visibility` COCO flag).

#### COCO import

Uploaded file: a single unpacked `*.json` or a zip archive with the structure above (without images).

- supported annotations: Polygons, Rectangles (if the `segmentation` field is empty)

#### How to create a task from MS COCO dataset

1. Download the [MS COCO dataset](http://cocodataset.org/#download).

   For example [2017 Val images](http://images.cocodataset.org/zips/val2017.zip)
   and [2017 Train/Val annotations](http://images.cocodataset.org/annotations/annotations_trainval2017.zip).

1. Create a CVAT task with the following labels:

   ```bash
   person bicycle car motorcycle airplane bus train truck boat "traffic light" "fire hydrant" "stop sign" "parking meter" bench bird cat dog horse sheep cow elephant bear zebra giraffe backpack umbrella handbag tie suitcase frisbee skis snowboard "sports ball" kite "baseball bat" "baseball glove" skateboard surfboard "tennis racket" bottle "wine glass" cup fork knife spoon bowl banana apple sandwich orange broccoli carrot "hot dog" pizza donut cake chair couch "potted plant" bed "dining table" toilet tv laptop mouse remote keyboard "cell phone" microwave oven toaster sink refrigerator book clock vase scissors "teddy bear" "hair drier" toothbrush
   ```

1. Select val2017.zip as data
   (See [Creating an annotation task](cvat/apps/documentation/user_guide.md#creating-an-annotation-task)
   guide for details)

1. Unpack `annotations_trainval2017.zip`

1. click `Upload annotation` button,
   choose `COCO 1.1` and select `instances_val2017.json.json`
   annotation file. It can take some time.


