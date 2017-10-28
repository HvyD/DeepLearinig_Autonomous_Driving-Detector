
# coding: utf-8

# # Car detection 
# 
# 

# In[1]:


import argparse
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import scipy.io
import scipy.misc
import numpy as np
import pandas as pd
import PIL
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model
from yolo_utils import read_classes, read_anchors, generate_colors, preprocess_image, draw_boxes, scale_boxes
from yad2k.models.keras_yolo import yolo_head, yolo_boxes_to_corners, preprocess_true_boxes, yolo_loss, yolo_body

get_ipython().magic(u'matplotlib inline')


# In[4]:




def yolo_filter_boxes(boxes, box_confidence, box_class_probs, threshold = .6):
    """Filters YOLO boxes by thresholding on object and class confidence.
    
    Arguments:
    boxes -- tensor of shape (19, 19, 5, 4)
    box_confidence -- tensor of shape (19, 19, 5, 1)
    box_class_probs -- tensor of shape (19, 19, 5, 80)
    threshold -- real value, if [ highest class probability score < threshold], then get rid of the corresponding box
    
    Returns:
    boxes -- tensor of shape (None, 4), containing (b_x, b_y, b_h, b_w) coordinates of selected boxes
    scores -- tensor of shape (None,), containing the class probability score for selected boxes
    classes -- tensor of shape (None,), containing the index of the class detected by the selected boxes
    
    Note: "None" is here because you don't know the exact number of selected boxes, as it depends on the threshold
    """
    
    # Compute box scores
   
    box_scores = box_confidence * box_class_probs
  
    
    # Find the box_classes thanks to the max box_scores, keep track of the corresponding score
  
    box_classes = K.argmax(box_scores, axis=-1)
    box_class_scores = K.max(box_scores, axis=-1)
 
    
    # Create a filtering mask based on the box_class_scores and on a threshold
    
    filtering_mask = (box_class_scores >= threshold)
    
    
    # Apply the mask to box coordinates, scores and classes
    
    boxes = tf.boolean_mask(boxes, filtering_mask)
    scores = tf.boolean_mask(box_class_scores, filtering_mask)
    classes = tf.boolean_mask(box_classes, filtering_mask)
 
    
    return boxes, scores, classes


# In[5]:


with tf.Session() as test_a:
    boxes = tf.random_normal([19, 19, 5, 4], mean=1, stddev=4, seed = 1)
    box_confidence = tf.random_normal([19, 19, 5, 1], mean=1, stddev=4, seed = 1)
    box_class_probs = tf.random_normal([19, 19, 5, 80], mean=1, stddev=4, seed = 1)
    boxes, scores, classes = yolo_filter_boxes(boxes, box_confidence, box_class_probs, threshold = 0.5)
    print("boxes[2] = " + str(boxes[2].eval()))
    print("scores[2] = " + str(scores[2].eval()))
    print("classes[2] = " + str(classes[2].eval()))
    print("boxes.shape = " + str(boxes.shape))
    print("scores.shape = " + str(scores.shape))
    print("classes.shape = " + str(classes.shape))


# In[6]:




def iou(box1, box2):
    """Implement the intersection over union (IoU) between box1 and box2
    
    Arguments:
    box1 -- first box, list object with coordinates (x1, y1, x2, y2)
    box2 -- second box, list object with coordinates (x1, y1, x2, y2)
    """

    # Calculate the (y1, x1, y2, x2) coordinates of the intersection of box1 and box2. Calculate its Area.
   
    xi1 = max(box1[0], box2[0])
    yi1 = min(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = max(box1[3], box2[3])
    inter_area = (xi2 - xi1) * (yi1 - yi2)
    

    # Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
 
    box1_area = (box1[2] - box1[0]) * (box1[1] - box1[3])
    box2_area = (box2[2] - box2[0]) * (box2[1] - box2[3])
    union_area = box1_area + box2_area - inter_area
  
    
    # compute the IoU
   
    iou = inter_area / union_area
   

    return iou


# In[7]:


box1 = (1, 4, 3, 2)
box2 = (2, 3, 4, 1)
print("iou = " + str(iou(box1, box2)))


# In[8]:




def yolo_non_max_suppression(boxes, scores, classes, max_boxes = 10, iou_threshold = 0.5):
    """
    Applies Non-maximum suppression to set of boxes
    
    Arguments:
    boxes -- tensor of shape (None, 4), output of yolo_filter_boxes().
    scores -- tensor of shape (None,), output of yolo_filter_boxes()
    classes -- tensor of shape (None,), output of yolo_filter_boxes()
    max_boxes -- integer, maximum number of predicted boxes you'd like
    iou_threshold -- real value, "intersection over union" threshold used for NMS filtering
    
    Returns:
    boxes -- tensor of shape (None, 4), predicted box coordinates
    scores -- tensor of shape (None,), predicted score for each box
    classes -- tensor of shape (None,), predicted class for each box
    
    Note: The "None" dimension of the output tensors has obviously to be less than max_boxes
    """
    
    max_boxes_tensor = K.variable(max_boxes, dtype='int32')     # tensor to be used in tf.image.non_max_suppression()
    K.get_session().run(tf.variables_initializer([max_boxes_tensor])) # initialize variable max_boxes_tensor
    

  
    nms_indices = tf.image.non_max_suppression(boxes, scores, max_boxes_tensor, iou_threshold=iou_threshold)
   
    boxes = K.gather(boxes, nms_indices)
    scores = K.gather(scores, nms_indices)
    classes = K.gather(classes, nms_indices)
    
    
    return boxes, scores, classes


# In[9]:


with tf.Session() as test_b:
    boxes = tf.random_normal([54, 4], mean=1, stddev=4, seed = 1)
    scores = tf.random_normal([54,], mean=1, stddev=4, seed = 1)
    classes = tf.random_normal([54,], mean=1, stddev=4, seed = 1)
    boxes, scores, classes = yolo_non_max_suppression(boxes, scores, classes)
    print("boxes[2] = " + str(boxes[2].eval()))
    print("scores[2] = " + str(scores[2].eval()))
    print("classes[2] = " + str(classes[2].eval()))
    print("boxes.shape = " + str(boxes.eval().shape))
    print("scores.shape = " + str(scores.eval().shape))
    print("classes.shape = " + str(classes.eval().shape))


# In[10]:




def yolo_eval(yolo_outputs, image_shape = (720., 1280.), max_boxes=10, score_threshold=.6, iou_threshold=.5):
    """
    Converts the output of YOLOv2 encoding (a lot of boxes) to your predicted boxes along with their scores and classes.
    
    Arguments:
    yolo_outputs -- output of the encoding model (for image_shape of (608, 608, 3)), contains 4 tensors:
                    box_xy: tensor of shape (None, 19, 19, 5, 2)
                    box_wh: tensor of shape (None, 19, 19, 5, 2)
                    box_confidence: tensor of shape (None, 19, 19, 5, 1)
                    box_class_probs: tensor of shape (None, 19, 19, 5, 80)
    image_shape -- tensor of shape (2,) containing the input shape, in this notebook we use (608., 608.) (has to be float32 dtype)
    max_boxes -- integer, maximum number of predicted boxes you'd like
    score_threshold -- real value, if [ highest class probability score < threshold], then get rid of the corresponding box
    iou_threshold -- real value, "intersection over union" threshold used for NMS filtering
    
    Returns:
    boxes -- tensor of shape (None, None, 4), predicted box coordinates
    scores -- tensor of shape (None, None), predicted score for each box
    classes -- tensor of shape (None, None), predicted class for each box
    """
    
 
    
    # Retrieve outputs of the YOLOv2 model
    box_xy, box_wh, box_confidence, box_class_probs = yolo_outputs

    # Convert boxes to be ready for filtering functions
    boxes = yolo_boxes_to_corners(box_xy, box_wh)

    # Score-filtering
    boxes, scores, classes = yolo_filter_boxes(boxes, box_confidence, box_class_probs, threshold=score_threshold)
    
    # Scale boxes back to original image shape.
    boxes = scale_boxes(boxes, image_shape)

    # Non-max suppression
    boxes, scores, classes = yolo_non_max_suppression(boxes, scores, classes, max_boxes, iou_threshold)
    
  
    
    return boxes, scores, classes


# In[11]:


with tf.Session() as test_b:
    yolo_outputs = (tf.random_normal([19, 19, 5, 2], mean=1, stddev=4, seed = 1),
                    tf.random_normal([19, 19, 5, 2], mean=1, stddev=4, seed = 1),
                    tf.random_normal([19, 19, 5, 1], mean=1, stddev=4, seed = 1),
                    tf.random_normal([19, 19, 5, 80], mean=1, stddev=4, seed = 1))
    boxes, scores, classes = yolo_eval(yolo_outputs)
    print("boxes[2] = " + str(boxes[2].eval()))
    print("scores[2] = " + str(scores[2].eval()))
    print("classes[2] = " + str(classes[2].eval()))
    print("boxes.shape = " + str(boxes.eval().shape))
    print("scores.shape = " + str(scores.eval().shape))
    print("classes.shape = " + str(classes.eval().shape))


# In[12]:


sess = K.get_session()


# In[13]:


class_names = read_classes("model_data/coco_classes.txt")
anchors = read_anchors("model_data/yolo_anchors.txt")
image_shape = (720., 1280.)


# In[14]:


yolo_model = load_model("model_data/yolo.h5")


# In[15]:


yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))


# In[16]:



boxes, scores, classes = yolo_eval(yolo_outputs, image_shape)


# In[25]:




def predict(sess, image_file):
    """
    Runs the graph stored in "sess" to predict boxes for "image_file". Prints and plots the preditions.
    
    Arguments:
    sess -- your tensorflow/Keras session containing the YOLOv2 graph
    image_file -- name of an image stored in the "images" folder.
    
    Returns:
    out_boxes -- tensor of shape (None, 4), coordinates of the predicted boxes
    out_scores -- tensor of shape (None, ), score of the predicted boxes
    out_classes -- tensor of shape (None, ), class index of the predicted boxes
    
    Note: "None" actually represents the number of predicted boxes, it varies between 0 and max_boxes. 
    """

    # Preprocess your image
    image, image_data = preprocess_image("images/" + image_file, model_image_size = (608, 608))

    # Run the session with the correct tensors and choose the correct placeholders in the feed_dict.
  
    out_boxes, out_scores, out_classes = sess.run([boxes, scores, classes],
                                                  feed_dict={yolo_model.input: image_data, K.learning_phase(): 0})
 

    # Print predictions info
    print('Found {} boxes for {}'.format(len(out_boxes), image_file))
    # Generate colors for drawing bounding boxes.
    colors = generate_colors(class_names)
    # Draw bounding boxes on the image file
    draw_boxes(image, out_boxes, out_scores, out_classes, class_names, colors)
    # Save the predicted bounding box on the image
    image.save(os.path.join("out", image_file), quality=90)
    # Display the results in the notebook
    output_image = scipy.misc.imread(os.path.join("out", image_file))
    imshow(output_image)
    
    return out_boxes, out_scores, out_classes


# In[26]:


out_boxes, out_scores, out_classes = predict(sess, "test.jpg")


# In[19]:


class_names = get_classes("model_data/kian_classes.txt")
anchors = get_anchors("model_data/yolo_anchors.txt")
data = np.load("underwater_data.npz")


# In[22]:


image_data, boxes = process_data(data['images'], data['boxes'])


# In[ ]:


detectors_mask, matching_true_boxes = get_detector_mask(boxes, anchors)


# In[24]:


model_body, model = create_model(anchors, class_names)


# In[ ]:


train(model, class_names, anchors, image_data, boxes, detectors_mask, matching_true_boxes)


# In[ ]:


draw(model_body, class_names, anchors,image_data, image_set='val', weights_name='trained_stage_3_best.h5',save_all=False)


# In[18]:


def _main(args):

    # Arg parsers converted into notebook global variables
    data_path = "data/underwater_data.npz"
    anchor_path = "model_data/yolo_anchors.txt"
    classes_path = "data/underwater_classes.txt"

    class_names = get_classes(classes_path)
    anchors = get_anchors(anchors_path)
    data = np.load(data_path) # custom data saved as a numpy file.
    #  has 2 arrays: an object array 'boxes' (variable length of boxes in each image)
    #  and an array of images 'images'

    image_data, boxes = process_data(data['images'], data['boxes'])
    anchors = YOLO_ANCHORS
    detectors_mask, matching_true_boxes = get_detector_mask(boxes, anchors)

    model_body, model = create_model(anchors, class_names)

    train(model, class_names, anchors, image_data, boxes, detectors_mask, matching_true_boxes)

    # assumes training/validation split is 0.9
    draw(model_body, class_names, anchors,image_data, image_set='val', weights_name='trained_stage_3_best.h5',
        save_all=False)


def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    if os.path.isfile(anchors_path):
        with open(anchors_path) as f:
            anchors = f.readline()
            anchors = [float(x) for x in anchors.split(',')]
            return np.array(anchors).reshape(-1, 2)
    else:
        Warning("Could not open anchors file, using default.")
        return YOLO_ANCHORS

def process_data(images, boxes=None):
    '''processes the data'''
    images = [PIL.Image.fromarray(i) for i in images]
    orig_size = np.array([images[0].width, images[0].height])
    orig_size = np.expand_dims(orig_size, axis=0)

    # Image preprocessing.
    processed_images = [i.resize((416, 416), PIL.Image.BICUBIC) for i in images]
    processed_images = [np.array(image, dtype=np.float) for image in processed_images]
    processed_images = [image/255. for image in processed_images]

    if boxes is not None:
        # Box preprocessing.
        # Original boxes stored as 1D list of class, x_min, y_min, x_max, y_max.
        boxes = [box.reshape((-1, 5)) for box in boxes]
        # Get extents as y_min, x_min, y_max, x_max, class for comparision with
        # model output.
        boxes_extents = [box[:, [2, 1, 4, 3, 0]] for box in boxes]

        # Get box parameters as x_center, y_center, box_width, box_height, class.
        boxes_xy = [0.5 * box[:, 3:5] + box[:, 1:3] for box in boxes]
        boxes_wh = [box[:, 3:5] - box[:, 1:3] for box in boxes]
        boxes_xy = [boxxy / orig_size for boxxy in boxes_xy]
        boxes_wh = [boxwh / orig_size for boxwh in boxes_wh]
        boxes = [np.concatenate((boxes_xy[i], boxes_wh[i], box[:, 0:1]), axis=1) for i, box in enumerate(boxes)]

        # find the max number of boxes
        max_boxes = 0
        for boxz in boxes:
            if boxz.shape[0] > max_boxes:
                max_boxes = boxz.shape[0]

        # add zero pad for training
        for i, boxz in enumerate(boxes):
            if boxz.shape[0]  < max_boxes:
                zero_padding = np.zeros( (max_boxes-boxz.shape[0], 5), dtype=np.float32)
                boxes[i] = np.vstack((boxz, zero_padding))

        return np.array(processed_images), np.array(boxes)
    else:
        return np.array(processed_images)

def get_detector_mask(boxes, anchors):
    '''
    Precompute detectors_mask and matching_true_boxes for training.
    Detectors mask is 1 for each spatial position in the final conv layer and
    anchor that should be active for the given boxes and 0 otherwise.
    Matching true boxes gives the regression targets for the ground truth box
    that caused a detector to be active or 0 otherwise.
    '''
    detectors_mask = [0 for i in range(len(boxes))]
    matching_true_boxes = [0 for i in range(len(boxes))]
    for i, box in enumerate(boxes):
        detectors_mask[i], matching_true_boxes[i] = preprocess_true_boxes(box, anchors, [416, 416])

    return np.array(detectors_mask), np.array(matching_true_boxes)

def create_model(anchors, class_names, load_pretrained=True, freeze_body=True):
    '''
    returns the body of the model and the model

    # Params:

    load_pretrained: whether or not to load the pretrained model or initialize all weights

    freeze_body: whether or not to freeze all weights except for the last layer's

    # Returns:

    model_body: YOLOv2 with new output layer

    model: YOLOv2 with custom loss Lambda layer

    '''

    detectors_mask_shape = (13, 13, 5, 1)
    matching_boxes_shape = (13, 13, 5, 5)

    # Create model input layers.
    image_input = Input(shape=(416, 416, 3))
    boxes_input = Input(shape=(None, 5))
    detectors_mask_input = Input(shape=detectors_mask_shape)
    matching_boxes_input = Input(shape=matching_boxes_shape)

    # Create model body.
    yolo_model = yolo_body(image_input, len(anchors), len(class_names))
    topless_yolo = Model(yolo_model.input, yolo_model.layers[-2].output)

    if load_pretrained:
        # Save topless yolo:
        topless_yolo_path = os.path.join('model_data', 'yolo_topless.h5')
        if not os.path.exists(topless_yolo_path):
            print("CREATING TOPLESS WEIGHTS FILE")
            yolo_path = os.path.join('model_data', 'yolo.h5')
            model_body = load_model(yolo_path)
            model_body = Model(model_body.inputs, model_body.layers[-2].output)
            model_body.save_weights(topless_yolo_path)
        topless_yolo.load_weights(topless_yolo_path)

    if freeze_body:
        for layer in topless_yolo.layers:
            layer.trainable = False
    final_layer = Conv2D(len(anchors)*(5+len(class_names)), (1, 1), activation='linear')(topless_yolo.output)

    model_body = Model(image_input, final_layer)

    # Place model loss on CPU to reduce GPU memory usage.
    with tf.device('/cpu:0'):
        # Replace Lambda with custom Keras layer for loss.
        model_loss = Lambda(
            yolo_loss,
            output_shape=(1, ),
            name='yolo_loss',
            arguments={'anchors': anchors,
                       'num_classes': len(class_names)})([
                           model_body.output, boxes_input,
                           detectors_mask_input, matching_boxes_input
                       ])

    model = Model(
        [model_body.input, boxes_input, detectors_mask_input,
         matching_boxes_input], model_loss)

    return model_body, model

def train(model, class_names, anchors, image_data, boxes, detectors_mask, matching_true_boxes, validation_split=0.1):
    '''
    retrain/fine-tune the model

    logs training with tensorboard

    saves training weights in current directory

    best weights according to val_loss is saved as trained_stage_3_best.h5
    '''
    model.compile(
        optimizer='adam', loss={
            'yolo_loss': lambda y_true, y_pred: y_pred
        })  # This is a hack to use the custom loss function in the last layer.

    model.fit([image_data, boxes, detectors_mask, matching_true_boxes],
              np.zeros(len(image_data)),
              validation_split=validation_split,
              batch_size=32,
              epochs=5)
    model.save_weights('trained_stage_1.h5')

    model_body, model = create_model(anchors, class_names, load_pretrained=False, freeze_body=False)

    model.load_weights('trained_stage_1.h5')

    model.compile(
        optimizer='adam', loss={
            'yolo_loss': lambda y_true, y_pred: y_pred
        })  # This is a hack to use the custom loss function in the last layer.


    model.fit([image_data, boxes, detectors_mask, matching_true_boxes],
              np.zeros(len(image_data)),
              validation_split=0.1,
              batch_size=8,
              epochs=30)

    model.save_weights('trained_stage_2.h5')

    model.fit([image_data, boxes, detectors_mask, matching_true_boxes],
              np.zeros(len(image_data)),
              validation_split=0.1,
              batch_size=8,
              epochs=30)

    model.save_weights('trained_stage_3.h5')

def draw(model_body, class_names, anchors, image_data, image_set='val',
            weights_name='trained_stage_3_best.h5', out_path="output_images", save_all=True):
    '''
    Draw bounding boxes on image data
    '''
    if image_set == 'train':
        image_data = np.array([np.expand_dims(image, axis=0)
            for image in image_data[:int(len(image_data)*.9)]])
    elif image_set == 'val':
        image_data = np.array([np.expand_dims(image, axis=0)
            for image in image_data[int(len(image_data)*.9):]])
    elif image_set == 'all':
        image_data = np.array([np.expand_dims(image, axis=0)
            for image in image_data])
    else:
        ValueError("draw argument image_set must be 'train', 'val', or 'all'")
    # model.load_weights(weights_name)
    print(image_data.shape)
    model_body.load_weights(weights_name)

    # Create output variables for prediction.
    yolo_outputs = yolo_head(model_body.output, anchors, len(class_names))
    input_image_shape = K.placeholder(shape=(2, ))
    boxes, scores, classes = yolo_eval(
        yolo_outputs, input_image_shape, score_threshold=0.07, iou_threshold=0)

    # Run prediction on overfit image.
    sess = K.get_session()  # TODO: Remove dependence on Tensorflow session.

    if  not os.path.exists(out_path):
        os.makedirs(out_path)
    for i in range(len(image_data)):
        out_boxes, out_scores, out_classes = sess.run(
            [boxes, scores, classes],
            feed_dict={
                model_body.input: image_data[i],
                input_image_shape: [image_data.shape[2], image_data.shape[3]],
                K.learning_phase(): 0
            })
        print('Found {} boxes for image.'.format(len(out_boxes)))
        print(out_boxes)

        # Plot image with predicted boxes.
        image_with_boxes = draw_boxes(image_data[i][0], out_boxes, out_classes,
                                    class_names, out_scores)
        # Save the image:
        if save_all or (len(out_boxes) > 0):
            image = PIL.Image.fromarray(image_with_boxes)
            image.save(os.path.join(out_path,str(i)+'.png'))

        # To display (pauses the program):
        # plt.imshow(image_with_boxes, interpolation='nearest')
        # plt.show()



if __name__ == '__main__':
    args = argparser.parse_args()
    _main(args)


# In[ ]:



   
for image_file in os.listdir(test_path):
    
    image_type = imghdr.what(os.path.join(test_path, image_file))

    image = Image.open(os.path.join(test_path, image_file))
    resized_image = image.resize(tuple(reversed(model_image_size)), Image.BICUBIC)
    image_data = np.array(resized_image, dtype='float32')
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

    out_boxes, out_scores, out_classes = sess.run([boxes, scores, classes],
        feed_dict={yolo_model.input: image_data,
                   input_image_shape: [image.size[1], image.size[0]],
                   K.learning_phase(): 0})
    
    print('Found {} boxes for {}'.format(len(out_boxes), image_file))

    font = ImageFont.truetype(font='font/FiraMono-Medium.otf',size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    thickness = (image.size[0] + image.size[1]) // 300

    for i, c in reversed(list(enumerate(out_classes))):
        predicted_class = class_names[c]
        box = out_boxes[i]
        score = out_scores[i]

        label = '{} {:.2f}'.format(predicted_class, score)

        draw = ImageDraw.Draw(image)
        label_size = draw.textsize(label, font)

        top, left, bottom, right = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
        print(label, (left, top), (right, bottom))

        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        # My kingdom for a good redistributable image drawing library.
        for i in range(thickness):
            draw.rectangle(
                [left + i, top + i, right - i, bottom - i],
                outline=colors[c])
        draw.rectangle(
            [tuple(text_origin), tuple(text_origin + label_size)],
            fill=colors[c])
        draw.text(text_origin, label, fill=(0, 0, 0), font=font)
        del draw

    image.save(os.path.join(output_path, image_file), quality=90)
sess.close()

