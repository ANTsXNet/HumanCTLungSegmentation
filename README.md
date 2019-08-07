# Work in progress

# App:  Lung CT segmentation

Deep learning app made for CT lung segmentation using ANTsRNet

## Model training notes

* Training data: human CT lung data
* Unet model (see ``Scripts/Training/``).
* Template-based data augmentation

## Sample usage

```
#
#  Usage:
#    Rscript doBrainExtraction.R inputImage outputPrefix reorientationTemplate
#

$ Rscript Scripts/doLungSegmentation.R Data/Example/lungct.nii.gz output Data/Template/T_template0.nii.gz

