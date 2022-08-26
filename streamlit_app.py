import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import re, glob
from six import BytesIO
import numpy as np
import os, cv2
import tensorflow as tf
import time

"""
# SideHelper
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
            draw.rectangle((ymin, xmin, ymax, xmax), outline=(255, 255, 0), width=5)
        display_str = labels[obj['class_id']] + ": " + str(round(obj['score']*100, 2)) + "%"
        draw.text((ymin+20,xmin), display_str, font=ImageFont.truetype("./maki.ttf", 20))

    displayImage = np.asarray( image )
    #display(Image.fromarray(displayImage))
    st.image(displayImage, caption = 'Detection Result')

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

    im = cv2.imread(image_path)
    size = im.shape

    file_path, file_extension = os.path.splitext(image_path)
    file_name = os.path.basename(file_path)
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
        xmin = str(int(size[1] * xmin))
        xmax = str(int(size[1] * xmax))
        ymin = str(int(size[0] * ymin))
        ymax = str(int(size[0] * ymax))
        
        #print(xmin, ymin, xmax, ymax)
        
        #string = labels[classes[i]] + " " + scores[i].astype(str) + " " + ymin.astype(str) + \
        #        " " + xmax.astype(str) + " " + ymax.astype(str) + " " + xmin.astype(str) + "\n"
        string = labels[classes[i]] + " " + scores[i].astype(str) + " " + xmin + \
                " " + ymin + " " + xmax + " " + ymax + "\n"

        #with open(save_path, "a") as myfile:
        #    myfile.write(string)
      
    return results

labels = load_labels('./efficientdet-labels.txt')

interpreter = tf.lite.Interpreter(
    model_path="./efficientdet-lite0.tflite")
interpreter.allocate_tensors()
_, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']

input_mean = 127.5
input_std = 127.5
path_to_images = "E:/Images/images/"
files = os.listdir(path_to_images)

start_time = time.time()
for i, image_path in enumerate(files):
    image_path = path_to_images+image_path
    #print(str(i) + '. ' + image_path)
    image = Image.open(image_path).convert('RGB')
    image_pred = image.resize((input_width ,input_height), Image.ANTIALIAS) #Resampling.LANCZOS
    if interpreter.get_input_details()[0]['dtype'] == np.float32:
        image_pred = (np.float32(image_pred) - input_mean) / input_std
    results = detect_objects(interpreter, image_pred, 0.5, image_path)
    draw_image(image, results, image.size)

    if (i == 3):
        break

end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))