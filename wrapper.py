import sys
import os
from subprocess import call
from cytomine.models import Job
from neubiaswg5 import CLASS_OBJSEG
from neubiaswg5.helpers import NeubiasJob, prepare_data, upload_data, upload_metrics


def main(argv):
    base_path = "{}".format(os.getenv("HOME")) # Mandatory for Singularity
    problem_cls = CLASS_OBJSEG

    with NeubiasJob.from_cli(argv) as nj:
        nj.job.update(status=Job.RUNNING, progress=0, statusComment="Initialisation...")
        # 1. Prepare data for workflow
        in_imgs, gt_imgs, in_path, gt_path, out_path, tmp_path = prepare_data(problem_cls, nj, is_2d=True, **nj.flags)
        files = [os.path.join(in_path,"{}.tif".format(image.id)) for image in in_imgs]

        # 2. Run ilastik prediction
        nj.job.update(progress=25, statusComment="Launching workflow...")
        shArgs = [
            "/ilastik/run_ilastik.sh",
            "--headless",
            "--project=PixelObjectClassification.ilp",
            '--export_source="Object Predictions"',
            "--output_format=tif",
            '--output_filename_format='+os.path.join(outpath,'{nickname}.tif')
            ]
        for img in in_imgs:
            shArgs.append("{}.tif".format(img.id))
        
        call_return = call(" ".join(shArgs), shell=True, cwd='ilastik')

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
