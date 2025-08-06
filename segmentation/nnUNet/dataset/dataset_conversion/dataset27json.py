import os
import json
import shutil

def save_json(obj, file, indent=4, sort_keys=True):
    with open(file, 'w') as f:
        json.dump(obj, f, sort_keys=sort_keys, indent=indent)

def maybe_mkdir_p(directory):
    directory = os.path.abspath(directory)
    splits = directory.split("/")
    for i in range(0, len(splits)):
        if not os.path.isdir(os.path.join("/", *splits[:i + 1])):
            try:
                os.mkdir(os.path.join("/", *splits[:i + 1]))
            except FileExistsError:
                print("WARNING: Folder %s already existed and does not need to be created" % directory)

def subdirs(folder, join=True, prefix=None, suffix=None, sort=True):
    if join:
        l = os.path.join
    else:
        l = lambda x, y: y
    res = [l(folder, i) for i in os.listdir(folder) if os.path.isdir(os.path.join(folder, i))
           and (prefix is None or i.startswith(prefix))
           and (suffix is None or i.endswith(suffix))]
    if sort:
        res.sort()
    return res

# base = "/data/pathology_2/TLS/CT_finish/KiTS19_modified0215/origin/"  # 原始数据集路径
# out = "/home/hdc/lwlw/2025/nnUNet/nnUNet/dataset/nnUNet_raw/Dataset040_PAN/"  # 结构化数据集目录

# base = "/data/pathology_2/TLS/CT_finish/PAN_0219_only1_20/orign"  # 原始数据集路径
# out = "/home/hdc/lwlw/2025/nnUNet/nnUNet/dataset/nnUNet_raw/Dataset050_PAN/"  # 结构化数据集目录

base = "/data/pathology_2/TLS/CT_finish/SE_CT_updata/"  # 原始数据集路径
out = "/home/hdc/lwlw/2025/nnUNet/nnUNet/dataset/nnUNet_raw/Dataset01_CT/"  # 结构化数据集目录

cases = subdirs(base, join=False)

maybe_mkdir_p(out)
maybe_mkdir_p(os.path.join(out, "imagesTr"))
maybe_mkdir_p(os.path.join(out, "imagesTs"))
maybe_mkdir_p(os.path.join(out, "labelsTr"))

for c in cases:
    case_id = int(c.split("_")[-1])
    if case_id < 80:
        # 训练集
        shutil.copy(os.path.join(base, c, "imaging.nii.gz"), os.path.join(out, "imagesTr", c + "_0000.nii.gz"))
        shutil.copy(os.path.join(base, c, "segmentation.nii.gz"), os.path.join(out, "labelsTr", c + ".nii.gz"))
    else:
        # 测试集
        shutil.copy(os.path.join(base, c, "imaging.nii.gz"), os.path.join(out, "imagesTs", c + "_0000.nii.gz"))

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
json_dict['numTraining'] = len([c for c in cases if int(c.split("_")[-1]) < 80])  # 训练集数量
json_dict['file_ending'] = ".nii.gz"
json_dict['numTest'] = len([c for c in cases if int(c.split("_")[-1]) >= 80])  # 测试集数量
json_dict['training'] = [{'image': "./imagesTr/%s_0000.nii.gz" % c, "label": "./labelsTr/%s.nii.gz" % c} for c in cases if int(c.split("_")[-1]) < 80]
json_dict['test'] = [{'image': "./imagesTs/%s_0000.nii.gz" % c} for c in cases if int(c.split("_")[-1]) >= 80]

save_json(json_dict, os.path.join(out, "dataset.json"))
