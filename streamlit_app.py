import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import re, glob
from six import BytesIO
import numpy as np
import os, cv2
import tensorflow as tf
import time
import statistics

"""
# SideHelper Demo
"""

def draw_image(image, results, size):
    result_size = len(results)
    for idx, obj in enumerate(results):
        # Prepare image for drawing
        draw = ImageDraw.Draw(image)

        # Prepare boundary box
        xmin, ymin, xmax, ymax = obj['bounding_box']
        xmin = int(xmin * size[1])
        xmax = int(xmax * size[1])
        ymin = int(ymin * size[0])
        ymax = int(ymax * size[0])

        #print(size)

        # Draw rectangle to desired thickness
        for x in range( 0, 10 ):
            color = tuple(np.random.choice(range(256), size=3))
            draw.rectangle((ymin, xmin, ymax, xmax), outline=color, width=5)
        display_str = labels[obj['class_id']] + ": " + str(round(obj['score']*100, 2)) + "%"
        draw.rectangle((ymin+5, xmin+10, ymin+len(display_str)*10, xmin+35), fill='black')
        draw.text((ymin+10,xmin+15), display_str, font=ImageFont.truetype("./utils/Neusharp-Bold.otf", 16))

    displayImage = np.asarray( image )
    #display(Image.fromarray(displayImage))
    st.image(displayImage)

def load_labels(path):
    """Loads the labels file. Supports files with or without index numbers."""
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        labels = {}
        for row_number, content in enumerate(lines):
            pair = re.split(r'[:\s]+', content.strip(), maxsplit=1)
            if len(pair) == 2 and pair[0].strip().isdigit():
                labels[int(pair[0])] = pair[1].strip()
            else:
                labels[row_number] = pair[0].strip()
    return labels

def set_input_tensor(interpreter, image):
    """Sets the input tensor."""
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image


def get_output_tensor(interpreter, index):
    """Returns the output tensor at the given index."""
    output_details = interpreter.get_output_details()[index]
    tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
    return tensor

def filter_boxes(box_xywh, scores, score_threshold=0.4, input_shape = tf.constant([416,416])):
    scores_max = tf.math.reduce_max(scores, axis=-1)

    mask = scores_max >= score_threshold
    class_boxes = tf.boolean_mask(box_xywh, mask)
    pred_conf = tf.boolean_mask(scores, mask)
    class_boxes = tf.reshape(class_boxes, [tf.shape(scores)[0], -1, tf.shape(class_boxes)[-1]])
    pred_conf = tf.reshape(pred_conf, [tf.shape(scores)[0], -1, tf.shape(pred_conf)[-1]])

    box_xy, box_wh = tf.split(class_boxes, (2, 2), axis=-1)

    input_shape = tf.cast(input_shape, dtype=tf.float32)

    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]

    box_mins = (box_yx - (box_hw / 2.)) / input_shape
    box_maxes = (box_yx + (box_hw / 2.)) / input_shape
    boxes = tf.concat([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ], axis=-1)
    # return tf.concat([boxes, pred_conf], axis=-1)
    return (boxes, pred_conf)

def detect_objects(interpreter, image, threshold, image_path = ""):
    """Returns a list of detection results, each a dictionary of object info."""
    set_input_tensor(interpreter, image)
    interpreter.invoke()

    # Get all output details
    boxes = get_output_tensor(interpreter, 1)
    classes = get_output_tensor(interpreter, 3)
    scores = get_output_tensor(interpreter, 0)
    count = int(get_output_tensor(interpreter, 2))

    #outputMap.put(0, outputScores);
    #outputMap.put(1, outputLocations);
    #outputMap.put(2, numDetections);
    #outputMap.put(3, outputClasses);

    #<class_name> <confidence> <left> <top> <right> <bottom>

    #im = cv2.imread(image_path)
    #size = im.shape

    #file_path, file_extension = os.path.splitext(image_path)
    #file_name = os.path.basename(file_path)
    #save_path = '/content/drive/MyDrive/sidehelper/detection-results/efficientdet-lite2-3/' + file_name + '.txt'

    results = []
    for i in range(count):
      if scores[i] >= threshold:
        result = {
            'bounding_box': boxes[i],
            'class_id': classes[i],
            'score': scores[i]
            }
        results.append(result)

        ymin, xmin, ymax, xmax = boxes[i]
        #print(boxes[i])
        #xmin = str(int(size[1] * xmin))
        #xmax = str(int(size[1] * xmax))
        #ymin = str(int(size[0] * ymin))
        #ymax = str(int(size[0] * ymax))
        
        #print(xmin, ymin, xmax, ymax)
        
        #string = labels[classes[i]] + " " + scores[i].astype(str) + " " + ymin.astype(str) + \
        #        " " + xmax.astype(str) + " " + ymax.astype(str) + " " + xmin.astype(str) + "\n"
        #string = labels[classes[i]] + " " + scores[i].astype(str) + " " + xmin + \
        #        " " + ymin + " " + xmax + " " + ymax + "\n"

        #with open(save_path, "a") as myfile:
        #    myfile.write(string)
      
    return results

input_mean = 127.5
input_std = 127.5
path_to_images = "./images/"
files = os.listdir(path_to_images)

with st.sidebar:
    st.markdown("<b style='color:#FF4B4B; font-size: large;'>Options:</b>", unsafe_allow_html=True)
    st.write('')
    threshold = st.slider('Set Confidence Threshold', 0, 100, 50) / 100
    st.write('')
    multiple = st.checkbox('Run multiple models')

    if multiple:
        option = st.multiselect(
         'Choose multiple models:',
         ['EfficientDet-Lite0', 'EfficientDet-Lite1', 'EfficientDet-Lite2'],
         ['EfficientDet-Lite0', 'EfficientDet-Lite1'])
    else:
        option = st.selectbox(
         'Choose a model:',
         ('EfficientDet-Lite0', 'EfficientDet-Lite1', 'EfficientDet-Lite2', 'EfficientDet-Lite3', 'EfficientDet-Lite4'))

    st.caption("Some models may take long time to infer.")

uploaded_files = st.file_uploader("Choose image file(s):", accept_multiple_files=True)
st.caption("If you do not upload any images, default images will be used.")

elapsed_times = []

efficientdet = ['EfficientDet-Lite0', 'EfficientDet-Lite1', 'EfficientDet-Lite2', 'EfficientDet-Lite3', 'EfficientDet-Lite4']

labels = load_labels('./utils/efficientdet-labels.txt')

if option in efficientdet:
    with st.spinner("Loading model..."):
        interpreter = tf.lite.Interpreter(
            model_path="./utils/" + option.lower() + ".tflite")
        interpreter.allocate_tensors()
        _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']

if uploaded_files is not None:
    i = 1
    if st.button('Run Detection'):
        my_bar = st.progress(0)

        if not uploaded_files:
            for j, image_path in enumerate(files):
                print(image_path)
                uploaded_files.append(path_to_images + image_path)

        print(uploaded_files)

        for uploaded_file in uploaded_files:
            with st.spinner("Running detection..."):
                start_time = time.time()
                #for i, image_path in enumerate(files):
                    #image_path = path_to_images+image_path
                    #print(str(i) + '. ' + image_path)
                image = Image.open(uploaded_file).convert('RGB')
                image_pred = image.resize((input_width ,input_height), Image.ANTIALIAS) #Resampling.LANCZOS
                if interpreter.get_input_details()[0]['dtype'] == np.float32:
                    image_pred = (np.float32(image_pred) - input_mean) / input_std
                results = detect_objects(interpreter, image_pred, threshold)
                draw_image(image, results, image.size)

                end_time = time.time()
                elapsed_time = end_time - start_time
                elapsed_times.append(elapsed_time)
                #print('Done! Took {} seconds'.format(elapsed_time))

                progress_bar_value = 1/len(uploaded_files)*i
                my_bar.progress(progress_bar_value)
                i+=1

        st.info('The model took {:.2f} seconds on average to infer.'.format(statistics.mean(elapsed_times)))