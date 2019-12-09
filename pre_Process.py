import numpy as np
import pandas as pd
import pydicom as dic
import os
import scipy.ndimage
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure, morphology

class pre_Process:
    def __init__(self):
        print("Hi")

    def load_scan(self, path):
        slices = [dic.read_file(path + '/' + s) for s in os.listdir(path)]
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        try:
            slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
        except:
            slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

        for s in slices:
            s.SliceThickness = slice_thickness

        return slices

    def get_pixels_hu(self, slices):
        image = np.stack([s.pixel_array for s in slices])
        # Convert to int16 (from sometimes int16),
        # should be possible as values should always be low enough (<32k)
        image = image.astype(np.int16)

        # Set outside-of-scan pixels to 0
        # The intercept is usually -1024, so air is approximately 0
        image[image == -2000] = 0

        # Convert to Hounsfield units (HU)
        for slice_number in range(len(slices)):

            intercept = slices[slice_number].RescaleIntercept
            slope = slices[slice_number].RescaleSlope

            if slope != 1:
                image[slice_number] = slope * image[slice_number].astype(np.float64)
                image[slice_number] = image[slice_number].astype(np.int16)

            image[slice_number] += np.int16(intercept)

        return np.array(image, dtype=np.int16)

    def transform_images(self, path, target):
        scans = self.load_scan(path)
        images = self.get_pixels_hu(scans)
        # APP_ROOT = os.path.dirname(os.path.abspath(__file__))

        i = 1
        for image in images:
            # print(image.shape)
            pic = Image.fromarray(image)
            # plt.imshow(image)
            # plt.show()
            cv2.imwrite(str(target+str(i)+".png"), image)
            i += 1
        return scans, images

    def resample(self, image, scan, new_spacing=[1, 1, 1]):
        # Determine current pixel spacing
        spacing = np.array([scan[0].SliceThickness] + [scan[0].PixelSpacing[0]]+[scan[0].PixelSpacing[1]], dtype=np.float32)

        resize_factor = spacing / new_spacing
        new_real_shape = image.shape * resize_factor
        new_shape = np.round(new_real_shape)
        real_resize_factor = new_shape / image.shape
        new_spacing = spacing / real_resize_factor

        image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')

        return image, new_spacing

    def plot_3d(self,image, threshold=-300, target = None):

        # Position the scan upright,
        # so the head of the patient would be at the top facing the camera
        p = image.transpose(2, 1, 0)

        verts, faces = measure.marching_cubes_classic(p, threshold)

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Fancy indexing: `verts[faces]` to generate a collection of triangles
        mesh = Poly3DCollection(verts[faces], alpha=0.70)
        face_color = [0.45, 0.45, 0.75]
        mesh.set_facecolor(face_color)
        ax.add_collection3d(mesh)

        ax.set_xlim(0, p.shape[0])
        ax.set_ylim(0, p.shape[1])
        ax.set_zlim(0, p.shape[2])

        if target != None:
            plt.savefig(str(str(target + str('3D_model') + ".png")),dpi = 100)
        plt.show()

    def largest_label_volume(self,im, bg=-1):
        vals, counts = np.unique(im, return_counts=True)

        counts = counts[vals != bg]
        vals = vals[vals != bg]

        if len(counts) > 0:
            return vals[np.argmax(counts)]
        else:
            return None

    def segment_lung_mask(self,image, fill_lung_structures=True):

        # not actually binary, but 1 and 2.
        # 0 is treated as background, which we do not want
        binary_image = np.array(image > -320, dtype=np.int8) + 1
        labels = measure.label(binary_image)

        # Pick the pixel in the very corner to determine which label is air.
        #   Improvement: Pick multiple background labels from around the patient
        #   More resistant to "trays" on which the patient lays cutting the air
        #   around the person in half
        background_label = labels[0, 0, 0]

        # Fill the air around the person
        binary_image[background_label == labels] = 2

        # Method of filling the lung structures (that is superior to something like
        # morphological closing)
        if fill_lung_structures:
            # For every slice we determine the largest solid structure
            for i, axial_slice in enumerate(binary_image):
                axial_slice = axial_slice - 1
                labeling = measure.label(axial_slice)
                l_max = self.largest_label_volume(labeling, bg=0)

                if l_max is not None:  # This slice contains some lung
                    binary_image[i][labeling != l_max] = 1

        binary_image -= 1  # Make the image actual binary
        binary_image = 1 - binary_image  # Invert it, lungs are now 1

        # Remove other air pockets insided body
        labels = measure.label(binary_image, background=0)
        l_max = self.largest_label_volume(labels, bg=0)
        if l_max is not None:  # There are air pockets
            binary_image[labels != l_max] = 0

        return binary_image


# clasd = pre_Process()
# scans, images = clasd.transform_images("/home/archie/Downloads/LIDC-IDRI-0013/01-01-2000-54750/3000551-80786", target="/home/archie/hi/" )
# # spacing = np.array(scans[0].SliceThickness + scans[0].PixelSpacing, dtype=np.float32)
# print([scans[0].SliceThickness]+[scans[0].PixelSpacing[0]]+[scans[0].PixelSpacing[1]])
# print()
# resample, new_space = clasd.resample(images, scans)
#
# plt.imshow(images[0])
# plt.show()
# plt.imshow(resample[0])
# plt.show()
# print("Shape before resampling\t", images.shape)
# print("Shape after resampling\t", resample.shape)
# clasd.plot_3d(resample, 400)
