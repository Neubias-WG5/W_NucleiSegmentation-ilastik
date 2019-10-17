import sys
import os
import numpy as np
from scipy import ndimage
import skimage
import skimage.morphology
from subprocess import call
from cytomine.models import Job
from neubiaswg5 import CLASS_OBJSEG
from neubiaswg5.helpers import NeubiasJob, prepare_data, upload_data, upload_metrics


def label_objects(img, threshold=0.8, min_radius=5):
    """
    Threshold ilastik probability map and convert binary data to objects
    """
    img = img[:,:,1]
    img[img>=threshold] = 1.0
    img[img<threshold] = 0.0
    selem = skimage.morphology.disk(min_radius)
    img = ndimage.morphology.binary_closing(img, structure=selem).astype(np.int)
    dimg = ndimage.morphology.distance_transform_edt(img)
    idimg = -1.0*dimg + dimg.max()
    h_max = skimage.morphology.h_maxima(dimg, min_radius, selem).astype(np.uint16)
    markers,num_objects = ndimage.label(h_max)
    wimg = skimage.morphology.watershed(idimg, markers, mask=img)
    img = wimg.astype(np.uint16)
    
    return img

def main(argv):
    base_path = "{}".format(os.getenv("HOME")) # Mandatory for Singularity
    problem_cls = CLASS_OBJSEG

    with NeubiasJob.from_cli(argv) as nj:
        nj.job.update(status=Job.RUNNING, progress=0, statusComment="Initialisation...")
        # 1. Prepare data for workflow
        in_imgs, gt_imgs, in_path, gt_path, out_path, tmp_path = prepare_data(problem_cls, nj, **nj.flags)

        temp_img = skimage.io.imread(os.path.join(in_path,"{}".format(in_imgs[0].filename)))
        if len(temp_img.shape) > 2:
            classification_project = "/app/RGBPixelClassification.ilp"
        else:
            classification_project = "/app/PixelClassification.ilp"

        # 2. Run ilastik prediction
        nj.job.update(progress=25, statusComment="Launching workflow...")
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
            img = label_objects(img, nj.parameters.probability_threshold, nj.parameters.min_radius)
            skimage.io.imsave(outfn, img)

        # 3. Upload data to Cytomine
        upload_data(problem_cls, nj, in_imgs, out_path, **nj.flags, monitor_params={
            "start": 60, "end": 90, "period": 0.1,
            "prefix": "Extracting and uploading polygons from masks"})
        
        # 4. Compute and upload metrics
        nj.job.update(progress=90, statusComment="Computing and uploading metrics...")
        upload_metrics(problem_cls, nj, in_imgs, gt_path, out_path, tmp_path, **nj.flags)

        # 5. Pipeline finished
        nj.job.update(progress=100, status=Job.TERMINATED, status_comment="Finished.")


if __name__ == "__main__":
    main(sys.argv[1:])
