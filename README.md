# CoastalWQL
CoastalWQL is an open-source software tailored for UAV-based water quality monitoring with a pushbroom hyperspectral imager. It performs the following workflow:

* Interactive selection of regions
* Interactive image alignment with time delay correction
* Producing false-composite images
* Sun glint correction
* Radiometric correction
* Image registration
* Image segmentation and masking
* Extraction of spectral information based on supplied water quality information
* Prediction of water quality map

![alt text](workflow.jpg "Workflow")

*Source*: Pak et al (2022) An open-source CoastalWQL software for stitching, pre-processing and visualization of push-broom hyperspectral imagery for UAV-based coastal water quality monitoring (to be submitted)

# Usage

* Download [anaconda](https://www.anaconda.com/)
* Clone or download this repository into your preferred directory
* Create a virtual environment using `conda env create --file CoastalWQL-env.txt` and `pip install -r CoastalWQL_requirements.txt`
* In your preferred IDE, run `GUI_platform.py`
* For testing image segmentation and masking, users can supply their own model or try out using the supplied segmentation model `xgb_segmentation.json`
* For testing model prediction, users may supply their own model or try out the models in the *turbidity_prediction* folder

*For more details on CoastalWQL's features, do read* `Descriptions of features.pdf`.

