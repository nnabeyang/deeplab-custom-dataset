# DeepLab Custom Dataset

This repository contains scripts for training DeepLab using custom dataset.

# Usage
```sh
git clone --recursive https://github.com/nnabeyang/deeplab-custom-dataset.git
cd deeplab-custom-dataset
cat <<EOS > .env
UNITY_DATASET_FILE_ID=<GOOGLE_DRIVE_FILE_ID>
UNITY_DATASET_FILE_NAME=<FILE_NAME>
EOS
cat <<EOS > data_generator.py.patch
diff --git a/models/research/deeplab/datasets/data_generator.py b/models/research/deeplab/datasets/data_generator.py
index d84e66f9c..be19f531f 100644
--- a/models/research/deeplab/datasets/data_generator.py
+++ b/models/research/deeplab/datasets/data_generator.py
@@ -99,10 +99,20 @@ _ADE20K_INFORMATION = DatasetDescriptor(
     ignore_label=0,
 )
 
+_MY_FIRST_PERCEPTION_INFORMATION = DatasetDescriptor(
+    splits_to_sizes={
+        'train': 1000,
+        'val': 100,
+    },
+    num_classes=11,
+    ignore_label=255,
+)
+
 _DATASETS_INFORMATION = {
     'cityscapes': _CITYSCAPES_INFORMATION,
     'pascal_voc_seg': _PASCAL_VOC_SEG_INFORMATION,
     'ade20k': _ADE20K_INFORMATION,
+    'my_first_perception': _MY_FIRST_PERCEPTION_INFORMATION,
 }
 
 # Default file pattern of TFRecord of TensorFlow Example.
EOS
docker-compose up -d
docker exec -it <CONTAINER_ID> /bin/bash -l
bash local_test_my_first_perception.sh
```
# License
MIT

# Author
[Noriaki Watanabe@nnabeyang](https://twitter.com/nnabeyang)
