import os
import pandas as pd
import numpy as np
from numpy import asarray
from matplotlib import pyplot as plt
import matplotlib.image as nping
from scipy import ndimage
from skimage import filters, feature, measure, color
from skimage.segmentation import watershed
from PIL import Image
import cv2 as cv
from flask import Flask, request, url_for, redirect, render_template, send_from_directory, send_file
from werkzeug.utils import secure_filename

app = Flask(__name__)

def graph_reading_processing(file_location_name):
    # load file
    img = cv.imread(file_location_name)
    grayscale_Image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, thresh_img = cv.threshold(grayscale_Image, 120, 255, cv.THRESH_BINARY)
    # show(thresh_img)
    kernel =np.ones((3),np.uint8)
    clear_image = cv.morphologyEx(thresh_img,cv.MORPH_OPEN, kernel, iterations=8)
    # show(clear_image)

    label_image = clear_image.copy()
    label_count = 0
    rows, cols = label_image.shape
    for j in range(rows):
        for i in range(cols):
            pixel = label_image[j, i]
            if 255 == pixel:
                label_count +- 1
                cv.floodFill(label_image, None, (i, j), label_count)
    # print("Number of foreground objects", label_count)

    contours, hierarchy = cv.findContours(clear_image,cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    output_contour = cv.cvtColor(clear_image, cv.COLOR_GRAY2BGR)
    # cv.drawContours(output_contour, contours, -1, (0, 0, 255), 2)
    # print("Number of detected contours", len(contours))
    # plt.imshow(output_contour)

    dist_trans = ndimage.distance_transform_edt(clear_image)
    local_max = feature.peak_local_max(dist_trans, min_distance=23)
    local_max_mask = np.zeros(dist_trans.shape, dtype=bool)
    local_max_mask[tuple(local_max.T)] = True
    labels = watershed(-dist_trans, measure.label(local_max_mask), mask=clear_image)
    # plt.figure(figsize=(10,10))
    # plt.imshow(color.label2rgb(labels, bg_label=0))
    # print("Number of Wheat grains are : %d" % labels.max())
    # save the file

    return labels.max(), color.label2rgb(labels, bg_label=0)

out_put_number, graph_data = graph_reading_processing('S15.JPG')

# Set allowed file extensions
ALLOWED_EXTENSIONS = ['jpg', 'jpeg', 'png','tiff','tif']
app.config['UPLOAD_FOLDER'] = 'input'
# create an app object using the Flask class
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/pred_with_file', methods=['POST'])
def pred_with_file():
    weight_input = [str(x) for x in request.form.values()]
    weight = float(weight_input[0])
    # Delete existing files that are in the 'input' folder
    input_dir = 'input'
    try:
        for f in os.listdir(os.path.join(os.getcwd(), input_dir)):
            os.remove(os.path.join(input_dir, f))
    except Exception as e:
        print('Error deleting files:', e)

    # Check if uploaded file has an allowed extension
    file = request.files['input_image']
    if not '.' in file.filename or file.filename.split('.')[-1].lower() not in ALLOWED_EXTENSIONS:
        return 'Invalid image format, should be jpg, jpeg, png, tiff, tif'

    # Save the uploaded file
    filename = secure_filename(file.filename)
    save_location_input_image = os.path.join(input_dir, filename)
    file.save(save_location_input_image)

    # Process the image and get the output data
    out_put_number, graph_data = graph_reading_processing(save_location_input_image)
    # calculate the 100 kernel weight
    thousand_weight = weight/float(out_put_number)*1000
    hundred_weight = weight/float(out_put_number)*100
    # Save the output image
    save_location_output_image = os.path.join(input_dir, 'output.png')
    plt.imsave(save_location_output_image, graph_data)
    final_output = 'Input weight:'+ str(weight)+'\n'+'Number of grains are:' + str(round(out_put_number,2)) +'\n'+ 'Thousand kernel weight: '+ str(round(thousand_weight,2)) +'\n'+ 'Hundred kernel weight: ' + str(round(hundred_weight,2))
    # Return the output image and the prediction text

    return render_template('index.html', prediction_text=final_output)

@app.route('/download_output', methods=['POST'])
def download_output():
    # Return the output image
    # must add this function then it will return the file as a attachment instead of display it directly
    return send_file('input/output.png', mimetype='image/png',as_attachment=True)

if __name__ == '__main__':
    app.run()
