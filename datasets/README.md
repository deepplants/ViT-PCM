# datasets
Folder containing scripts to build, download and store datasets.

### 1. Prepare nvidia-docker container
Follow container [README](../container/README.md) instructions.

### 2. Open project in nvidia-docker container
Run container if it is not running:
```bash
cd $HOME/ViT-PCM
bash container/main.sh 
```
### 3. Download and convert datasets
#### - PascalVOC 2012 download, build and convert in tfrecords
Once you ran tensorflow nvidia-docker container as in step 2, inside container type:
```bash
cd datasets
source download_and_convert_voc12.sh
```

#### - MS-COCO 2014 (also 2017) download, build and convert in tfrecords
Once you ran tensorflow nvidia-docker container as in step 2, inside container type:
```bash
cd datasets
source download_and_convert_coco.sh -y 2014 # support also 2017
# use flag '--to-voc12 true' to use only PascalVOC12 category set
```
