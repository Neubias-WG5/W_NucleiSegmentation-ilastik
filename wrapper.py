import sys
import os
import numpy as np
from scipy import ndimage
import skimage
import skimage.io
import skimage.feature
import skimage.morphology
import skimage.filters
import skimage.color
import skimage.segmentation
from subprocess import call
from cytomine.models import Job
from biaflows import CLASS_OBJSEG
from biaflows.helpers import BiaflowsJob, prepare_data, upload_data, upload_metrics


def label_objects(img, threshold=0.9, min_size=25):
    """
    Threshold ilastik probability map and convert binary data to objects
    """
    # Do fill holes, watershed, remove small objects
    # Or watershed to probability map, threshold, remove small objects
    img = img[:,:,1]
    img[img>=threshold] = 1.0
    img[img<threshold] = 0.0
    img = skimage.morphology.remove_small_holes(img.astype(np.bool), min_size)
    distance = ndimage.distance_transform_edt(img)
    distance = skimage.filters.gaussian(distance, sigma=3)
    local_maxi = skimage.feature.peak_local_max(distance, indices=False, footprint=np.ones((3, 3)), labels=img)
    markers = skimage.morphology.label(local_maxi)
    labelimg = skimage.morphology.watershed(-distance, markers, mask=img)
    labelimg = labelimg.astype(np.uint16)
    labelimg = skimage.morphology.remove_small_objects(labelimg, min_size)
    labelimg = skimage.segmentation.relabel_sequential(labelimg)[0].astype(np.uint16)
    
    return labelimg

def convert2rgb(in_imgs):
    """
    Convert grayscale images in a path to rgb
    """
    for image in in_imgs:
        img = skimage.io.imread(image.filepath)
        img = skimage.color.gray2rgb(img)
        skimage.io.imsave(image.filepath, img)

def main(argv):
    base_path = "{}".format(os.getenv("HOME")) # Mandatory for Singularity
    problem_cls = CLASS_OBJSEG

    with BiaflowsJob.from_cli(argv) as bj:
        bj.job.update(status=Job.RUNNING, progress=0, statusComment="Initialisation...")
        # 1. Prepare data for workflow
        in_imgs, gt_imgs, in_path, gt_path, out_path, tmp_path = prepare_data(problem_cls, bj, **bj.flags)

        temp_img = skimage.io.imread(os.path.join(in_path,"{}".format(in_imgs[0].filename)))
        if len(temp_img.shape) > 2:
            classification_project = "/app/RGBPixelClassification.ilp"
        else:
            classification_project = "/app/RGBPixelClassification.ilp"
            convert2rgb(in_imgs)

        # 2. Run ilastik prediction
        bj.job.update(progress=25, statusComment="Launching workflow...")
        shArgs = [
            "/app/ilastik/run_ilastik.sh",
            "--headless",
            "--project="+classification_project,
            "--export_source=Probabilities",
            "--output_format=tif",
            '--output_filename_format='+os.path.join(tmp_path,'{nickname}.tif')
            ]
        shArgs += [image.filepath for image in in_imgs]
        
        call_return = call(" ".join(shArgs), shell=True)

        # Threshold probabilites
        for image in in_imgs:
            fn = os.path.join(tmp_path,"{}".format(image.filename))
            outfn = os.path.join(out_path,"{}".format(image.filename))
            img = skimage.io.imread(fn)
            img = label_objects(img, bj.parameters.probability_threshold, bj.parameters.min_size)
            skimage.io.imsave(outfn, img)

        # 3. Upload data to Cytomine
        upload_data(problem_cls, bj, in_imgs, out_path, **bj.flags, monitor_params={
            "start": 60, "end": 90, "period": 0.1,
            "prefix": "Extracting and uploading polygons from masks"})
        
        # 4. Compute and upload metrics
        bj.job.update(progress=90, statusComment="Computing and uploading metrics...")
        upload_metrics(problem_cls, bj, in_imgs, gt_path, out_path, tmp_path, **bj.flags)

        # 5. Pipeline finished
        bj.job.update(progress=100, status=Job.TERMINATED, status_comment="Finished.")


if __name__ == "__main__":
    main(sys.argv[1:])
