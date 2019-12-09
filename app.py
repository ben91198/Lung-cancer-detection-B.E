import cv2
from PIL import Image
from flask import Flask, render_template, request, send_from_directory
import os
from .pre_Process import pre_Process

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))


@app.route('/')
def load_home_page():
    return render_template("file_upload.html")


@app.route('/upload', methods=['POST'])
def upload():
    target = os.path.join(APP_ROOT, "images/")
    print(target)
    if not os.path.isdir(target):
        os.mkdir(target)
    for file in request.files.getlist("file"):
        print(file)
        filename = file.filename
        destination = "/".join([target, filename])
        print(destination)
        file.save(destination)
    targets = os.path.join(APP_ROOT, "images/")
    targetd = os.path.join(APP_ROOT, "image_files/")
    # image_png =[]
    if not os.path.isdir(targetd):
        os.mkdir(targetd)
    obj = pre_Process()
    scans, images = obj.transform_images(path=targets, target=targetd)
    resample1, new_spacing = obj.resample(images, scans)
    target = os.path.join(APP_ROOT, "images_resampled/")
    if not os.path.isdir(target):
        os.mkdir(target)
    i = 1
    for image in resample1:
        # print(image.shape)
        pic = Image.fromarray(image)
        # plt.imshow(image)
        # plt.show()
        cv2.imwrite(str(target + str(i) + ".png"), image)
        i += 1

    segmented_lungs = obj.segment_lung_mask(resample1, False)

    # target = os.path.join(APP_ROOT, "images_segmented/")
    # if not os.path.isdir(target):
    #     os.mkdir(target)
    # i = 1
    # for image in segmented_lungs:
    #     # print(image.shape)
    #     pic = Image.fromarray(image)
    #     # plt.imshow(image)
    #     # plt.show()
    #     cv2.imwrite(str(target + str(i) + ".png"), image)
    #     i += 1
    # # segmented_lungs_fill = obj.segment_lung_mask(resample1, True)
    target = os.path.join(APP_ROOT, "images_3D/")
    if not os.path.isdir(target):
        os.mkdir(target)
    obj.plot_3d(segmented_lungs, 0, target=target)
    # cv2.imwrite(str(target + str(i) + ".png"), image_3d)
    return render_template("complete.html", title="ResultsPage")


@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory("image_files", filename=filename)


@app.route('/upload1/<filename>')
def send_image_res(filename):
    return send_from_directory("images_resampled", filename=filename)

@app.route('/upload2/<filename>')
def send_image_3D(filename):
    return send_from_directory("images_3D", filename=filename)

@app.route('/view_raw', methods=['Post'])
def view_raw():
    image_names = os.listdir('./images')
    targetd = os.path.join(APP_ROOT, "image_files/")
    # image_png =[]
    image_names = os.listdir(targetd)
    return render_template("complete.html", title="v_raw", image_names=image_names)


@app.route('/preprocessed', methods=['Post'])
def view_preprocessed():
    targetd = os.path.join(APP_ROOT, "images_resampled/")
    # image_png =[]
    image_names = os.listdir(targetd)
    return render_template("complete.html", title="pre_processed", image_names=image_names, )


if __name__ == '__main__':
    app.run()
