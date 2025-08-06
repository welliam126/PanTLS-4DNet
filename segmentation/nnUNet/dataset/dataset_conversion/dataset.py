import os
import json
import shutil


base = "/data/pathology_2/TLS/CT_finish/SE_CT_updata/Dataset001_CT/imagesTr"


json_dict = {}
json_dict['name'] = "PAN"  
json_dict['description'] = "kidney and kidney tumor segmentation"
json_dict['tensorImageSize'] = "4D"
json_dict['reference'] = "PAN data for nnunet"
json_dict['licence'] = ""
json_dict['release'] = "0.0"

json_dict['channel_names'] = {
    "0": "CT",
}
json_dict['labels'] = {
    "background": "0",
    "tumor": "1",
    "peritumor": "2",
    
}
json_dict['file_ending'] = ".nii.gz"
files = os.listdir(base)
d = []
for f in files:
    d.append({'image': f'imagesTr/{f}', 'label': f'labelsTr/{f}'})
print(d)
print(len(d))

data = json_dict
data['training'] = d
data['numTraining'] = int(len(d) * 0.8)
print(data)
json.dump(data, open('/data/pathology_2/TLS/CT_finish/SE_CT_updata/Dataset001_CT/dataset.json', 'w'), indent=4)
