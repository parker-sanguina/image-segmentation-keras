import enum
import os
from re import S
import sys
import json
import datetime
import numpy as np
import skimage.draw
import cv2.cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import Config
import utils
import visualize

import tensorflow as tf
import tensorflow.keras.utils as KU

import multiprocessing

############################################################
#  Configurations
############################################################

class HandConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "hands"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + hands

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

    # Don't use mini masks as default
    USE_MINI_MASK = False

    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 500
    IMAGE_MAX_DIM = 500

    IMAGE_SHAPE = (500, 500, 3)

    EPOCHS = 10

class HandDataset(utils.Dataset):

    def load_hands(self, dataset_dir, subset=None):
        """Load a subset of the Balloon dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("hand", 1, "hand")

        # Train or validation dataset?
        assert subset in ["training", "validation"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        # VGG Image Annotator (up to version 1.6) saves each image in the form:
        # _via_img_metadata: contains all the images
        # {"_via_img_metadata": {
        #       "{filename+size}": {
        #           "filename": "7.2.png,
        #           "size": 14859027,  
        #           "regions": [{
        #               "shape_attributes": {
        #                   "name": "polyline",
        #                   "all_points_x": [...],
        #                   "all_points_y": [...]},
        #               "region_attributes": {}},
        #               ... more regions ...
        #               ],
        #               "file_attributes": {}},
        #       ... more files ...
        #       }
        # }
        
        # We mostly care about the x and y coordinates of each region
        # Note: In VIA 2.0, regions was changed from a dict to a list.
        annotations = json.load(open(os.path.join(dataset_dir, "masks.json")))
        annotations = list(annotations["_via_img_metadata"].values())

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. These are stores in the
            # shape_attributes (see json format above)
            # The if condition is needed to support VIA versions 1.x and 2.x.
            if type(a['regions']) is dict:
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
            else:
                polygons = [r['shape_attributes'] for r in a['regions']] 

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image("hand", 
                           image_id=a['filename'],  # use file name as a unique image id
                           path=image_path,
                           width=width, 
                           height=height,
                           polygons=polygons)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        image_info = self.image_info[image_id]
        if image_info["source"] != "hand":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])], dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            if len(p['all_points_y']) > 0 and len(p['all_points_x']) > 0:
                rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
                mask[rr, cc, i] = 1

            #rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            #mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "hand":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

class DataController():
    """
    This generator doesn't seem to play nice in model training. 
    So it will not be used but will remain here for posterity.
    """
    
    def __init__(self, dataset, config, batch_size=32, shuffle=True, augmentation=None):
        self.image_ids = np.copy(dataset.image_ids)
        self.dataset = dataset
        self.config = config
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.image_shape = config.IMAGE_SHAPE
        self.augmentation = augmentation

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.image_ids)

    def __len__(self):
        return int(np.ceil(len(self.image_ids) / float(self.batch_size)))

    def __get_data(self, batches):
        # Returns a batch of images according to specified batch size
        # Batches is a list of image ids

        batch_images = []
        batch_gt_class_ids = []
        batch_gt_boxes = []
        batch_gt_masks = []

        batch_images = np.zeros((self.batch_size, self.config.IMAGE_SHAPE[0], self.config.IMAGE_SHAPE[1], self.config.IMAGE_SHAPE[2]), dtype=np.float32)
        #batch_gt_class_ids = np.zeros((self.batch_size, self.config.NUM_CLASSES), dtype=np.int32)
        #batch_gt_boxes = np.zeros((self.batch_size, 2, 4), dtype=np.int32)
        batch_gt_masks = np.zeros((self.batch_size, 500 * 500, 4), dtype=np.bool)
        
        for i, image_id in enumerate(batches):
            image, image_meta, gt_class_ids, gt_boxes, gt_masks = load_image_gt(self.dataset, self.config, image_id, augmentation=self.augmentation)

            if not np.any(gt_class_ids > 0):
                continue

            batch_images[i] = image
            #batch_gt_class_ids[i] = gt_class_ids
            #batch_gt_boxes[i] = gt_boxes
            batch_gt_masks[i] = gt_masks.reshape((500 * 500, 4))

        return batch_images, batch_gt_masks

    def __fetch(self):
        # Randomly select data
        while True:
            indexes = np.random.choice(self.image_ids, self.batch_size, replace=True)
            batches = self.image_ids[indexes]
            X, y = self.__get_data(batches)
            yield X, y

    def generate_data(self):
        #NOTE: Modify the output to appropriately match unet output, must first get full sized mask
        
        # Input is 32 batches of RGB images of shape (500, 500)
        # Output is 32 batches of arrays of shape (1, 2) each column denoting a label
        #dataset = tf.data.Dataset.from_generator(self.__fetch, output_signature=(tf.TensorSpec(shape=(32, 500, 500, 3), dtype=tf.float32), 
        #                                                                         tf.TensorSpec(shape=(32, 2), dtype=tf.int32)))

        # Masks
        dataset = tf.data.Dataset.from_generator(self.__fetch, output_signature=(tf.TensorSpec(shape=(self.batch_size, 500, 500, 3), dtype=tf.float32), 
                                                                                 tf.TensorSpec(shape=(self.batch_size, 500 * 500, 4), dtype=tf.int32)))

        # This is the number of batches to be used throught the entirety of training                                                                                 
        dataset = dataset.take(self.config.STEPS_PER_EPOCH * self.config.EPOCHS)

        return dataset

    def split_data(self):
        """
        Return dataset into X and y components for training
        """
        
        dataset_size = self.dataset.image_ids.shape[0]
        mask_size = self.config.IMAGE_SHAPE[0] * self.config.IMAGE_SHAPE[1]
    
        # Initialize arrays for data
        images = np.zeros((dataset_size, self.config.IMAGE_SHAPE[0], self.config.IMAGE_SHAPE[1], self.config.IMAGE_SHAPE[2]), dtype=np.float32)
        #class_ids = np.zeros((dataset_size, self.config.NUM_CLASSES), dtype=np.int32)
        #boxes = np.zeros((dataset_size, 2, 4), dtype=np.int32)
        masks = np.zeros((dataset_size, mask_size, 4), dtype=np.bool)

        # Shuffle ids
        image_ids = np.copy(self.dataset.image_ids)
        np.random.shuffle(image_ids)

        # Loop through image ids in dataset
        for i, image_id in enumerate(tqdm(image_ids)):
            # Load image and info
            image, image_meta, gt_class_ids, gt_boxes, gt_masks = load_image_gt(self.dataset, self.config, image_id, augmentation=self.augmentation)

            if not np.any(gt_class_ids > 0):
                continue

            images[i] = image
            #class_ids[i] = gt_class_ids
            #boxes[i] = gt_boxes
            masks[i] = gt_masks.reshape((mask_size, 4))

        return images, masks


def load_image_gt(dataset, config, image_id, augmentation=None):
    """Load and return ground truth data for an image (image, mask, bounding boxes).

    augmentation: Optional. An imgaug (https://github.com/aleju/imgaug) augmentation.
    For example, passing imgaug.augmenters.Fliplr(0.5) flips images
    right/left 50% of the time.

    Returns:
    image: [height, width, 3]
    shape: the original shape of the image before resizing and cropping.
    class_ids: [instance_count] Integer class IDs
    bbox: [instance_count, (y1, x1, y2, x2)]
    mask: [height, width, instance_count]. The height and width are those
    of the image unless use_mini_mask is True, in which case they are
    defined in MINI_MASK_SHAPE.
    """
    # Load image and mask
    image = dataset.load_image(image_id)
    mask, class_ids = dataset.load_mask(image_id)
    original_shape = image.shape

    image, window, scale, padding, crop = preprocess(image, config, equalize=True)

    #image, window, scale, padding, crop = utils.resize_image(image, min_dim=config.IMAGE_MIN_DIM, min_scale=config.IMAGE_MIN_SCALE, 
    #                                                            max_dim=config.IMAGE_MAX_DIM, mode=config.IMAGE_RESIZE_MODE)

    mask = utils.resize_mask(mask, scale, padding, crop)

    # Augmentation
    # This requires the imgaug lib (https://github.com/aleju/imgaug)
    if augmentation:
        import imgaug

        # Augmenters that are safe to apply to masks
        # Some, such as Affine, have settings that make them unsafe, so always
        # test your augmentation on masks
        MASK_AUGMENTERS = ["Sequential", "SomeOf", "OneOf", "Sometimes",
                        "Fliplr", "Flipud", "CropAndPad",
                        "Affine", "PiecewiseAffine"]

        def hook(images, augmenter, parents, default):
            """Determines which augmenters to apply to masks."""
            return augmenter.__class__.__name__ in MASK_AUGMENTERS

        # Store shapes before augmentation to compare
        image_shape = image.shape
        mask_shape = mask.shape
        # Make augmenters deterministic to apply similarly to images and masks
        det = augmentation.to_deterministic()
        image = det.augment_image(image)
        # Change mask to np.uint8 because imgaug doesn't support np.bool
        mask = det.augment_image(mask.astype(np.uint8), hooks=imgaug.HooksImages(activator=hook))
        # Verify that shapes didn't change
        assert image.shape == image_shape, "Augmentation shouldn't change image size"
        assert mask.shape == mask_shape, "Augmentation shouldn't change mask size"
        # Change mask back to bool
        mask = mask.astype(np.bool)

    # Note that some boxes might be all zeros if the corresponding mask got cropped out.
    # and here is to filter them out
    _idx = np.sum(mask, axis=(0, 1)) > 0
    mask = mask[:, :, _idx]
    class_ids = class_ids[_idx]
    # Bounding boxes. Note that some boxes might be all zeros
    # if the corresponding mask got cropped out.
    # bbox: [num_instances, (y1, x1, y2, x2)]
    bbox = utils.extract_bboxes(mask)

    # Active classes
    # Different datasets have different classes, so track the
    # classes supported in the dataset of this image.
    active_class_ids = np.zeros([dataset.num_classes], dtype=np.int32)
    source_class_ids = dataset.source_class_ids[dataset.image_info[image_id]["source"]]
    active_class_ids[source_class_ids] = 1

    # Resize masks to smaller size to reduce memory usage
    if config.USE_MINI_MASK:
        mask = utils.minimize_mask(bbox, mask, config.MINI_MASK_SHAPE)

    # Image meta data
    image_meta = compose_image_meta(image_id, original_shape, image.shape, window, scale, active_class_ids)

    return image, image_meta, class_ids, bbox, mask

def preprocess(image, config, equalize=True):
    if equalize:
        # Compute CLAHE
        LAB = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        L = LAB[:,:,0]
        L = clahe.apply(L)
        LAB[:,:,0] = L
        image = cv2.cvtColor(LAB, cv2.COLOR_LAB2BGR)

    # Resize
    image, window, scale, padding, crop = utils.resize_image(image, min_dim=config.IMAGE_MIN_DIM, max_dim=config.IMAGE_MAX_DIM, mode=config.IMAGE_RESIZE_MODE)
    return image, window, scale, padding, crop

def mold_image(images, config):
    """Expects an RGB image (or array of images) and subtracts
    the mean pixel and converts it to float. Expects image
    colors in RGB order.
    """
    return images.astype(np.float32) - config.MEAN_PIXEL


def unmold_image(normalized_images, config):
    """Takes a image normalized with mold() and returns the original."""
    return (normalized_images + config.MEAN_PIXEL).astype(np.uint8)

def compose_image_meta(image_id, original_image_shape, image_shape,
                       window, scale, active_class_ids):
    """Takes attributes of an image and puts them in one 1D array.

    image_id: An int ID of the image. Useful for debugging.
    original_image_shape: [H, W, C] before resizing or padding.
    image_shape: [H, W, C] after resizing and padding
    window: (y1, x1, y2, x2) in pixels. The area of the image where the real
            image is (excluding the padding)
    scale: The scaling factor applied to the original image (float32)
    active_class_ids: List of class_ids available in the dataset from which
        the image came. Useful if training on images from multiple datasets
        where not all classes are present in all datasets.
    """
    meta = np.array(
        [image_id] +                  # size=1
        list(original_image_shape) +  # size=3
        list(image_shape) +           # size=3
        list(window) +                # size=4 (y1, x1, y2, x2) in image cooredinates
        [scale] +                     # size=1
        list(active_class_ids)        # size=num_classes
    )
    return meta

def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

def detect_and_draw(model, image_path=None, video_path=None):
    assert image_path or video_path

    CLASS_NAMES = ['BG', 'hand']

    # Image or video?
    if image_path:
        # Run model detection and draw mask on image
        #print("Running on {}".format(args.image))
        print("Running on {}".format(image_path))
        # Read image
        #image = skimage.io.imread(args.image)
        image = skimage.io.imread(image_path)
        image = image[:,:,0:3]
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Draw detected masks
        img_data = visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], r['scores'])
        # Save output
        file_name = "/Users/parker/Desktop/splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        # display_instances used
        plt.imsave(file_name, img_data)
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "/Users/parker/Desktop/splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))
    
        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Draw masks
                # TO DO: change this because it's slow
                img_data = visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], CLASS_NAMES, r['scores'])
                # RGB -> BGR to save image to video
                img_data = img_data[..., ::-1]
                # Add image to video writer
                vwriter.write(img_data)
                count += 1

        vwriter.release()
        vcapture.release()
    print("Saved to ", file_name)