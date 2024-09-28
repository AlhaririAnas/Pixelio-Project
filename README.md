# Pixelio-Project

## Overview

This repository houses a project that finds the top x images similar to any given photo from a large dataset. The software utilizes Python to recommend images based on various similarity metrics such as color schemes, embeddings, and YOLO object detection.

## Getting Started

### Prerequisites

Ensure Python is installed on your system and you have the necessary permissions to execute scripts.

### Installation

Clone the repository to your local machine:

```bash
git clone https://github.com/AlhaririAnas/Pixelio-Project
```

Navigate to the project directory:

```bash
cd Pixelio-Project
```

Install dependencies:

```bash
pip install .
```

## Setting up your dataset

To set up a dataset with all similarities and metadata, use:
```bash
python -m resources.main -m -s -p "../Parent_folder_of_dataset"
```

Use this commands to specify:

|  Command 	|  Explanation 	|   Tip	|
|---	|---	|---	|
|  -m 	|  set up the image_metadata.db 	|   	|
|  -s	|  set up a pickle file with all similarity information 	|   	|
|  -p	|   path to your '/data'	directory with your images|  Default: 'D:/'	|
|  -d	|  device to use: cpu or cuda 	|  If not specified, cuda will be used if available 	|
|  --pkl_file 	|  path to a pickle file, in which the similarity information are stored 	|  Default: 'similarities.pkl' 	|
|  --checkpoint	|  Number of images after which the --pkl_file will be updated | Default: 100  	|

If the program crashes, you can simply restart it and it will continue where it left off.

If neither the `-m` nor the `-s` flag are specified, it will launch the web interface...

## Launching the Web Interface

To start the web-based interface, use the following command:
```bash
python -m resources.main --color
```

Make sure that the metadata.db file is located in the Pixelio project folder. The similarities.pkl should also be located there, or you can refer to it using `--pkl_file`. \
**The complete data set must be located in `/app/static/images`, e.g:**
```bash
/app/
    /static/
        /images/
            /000/
                01.jpg
                02.jpg
            /001/
                03.png
                04.png
    /templates/
        index.html
    app.py
```

In addition, `--image_count` can be used to set how many similar images are to be calculated.
Besides `--color` (color histograms) there are `--embedding` (Inception v3 embeddings) and `--yolo` (YOLO v8 object detection) as selection of the measuring method. An example could be:
```bash
python -m resources.main --yolo --image_count 30
```


After running the command, the system will open a browser window automatically.
This process may take some time, especially if you are performing it for the first time. This is because the pickle file loads.

## Pixelio Demo
![Web interface](public/Web%20Interface.png)

If you click on an image, a sidebar will appear on the right side containing the most similar images to this image.

## Contact

For any inquiries, please contact:

* Jonah Gräfe: jonah.graefe@study.hs-duesseldorf.de
* Joschua Schramm: joschua.schramm@study.hs-duesseldorf.de
* Anas Alhariri: anas.alhariri@study.hs-duesseldorf.de
