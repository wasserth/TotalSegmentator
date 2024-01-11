import sys
from pathlib import Path
p_dir = str(Path(__file__).absolute().parents[1])
if p_dir not in sys.path: sys.path.insert(0, p_dir)

import subprocess

import nibabel as nib
import numpy as np


def add_label_map_to_nifti(img_in, label_map):
    """
    This will save the information which label in a segmentation mask has which name to the extended header.

    img: nifti image
    label_map: a dictionary with label ids and names | a list of names and a running id will be generated starting at 1

    returns: nifti image
    """
    data = img_in.get_fdata()

    if label_map is None:
        label_map = {idx+1: f"L{val}" for idx, val in enumerate(np.unique(data)[1:])}

    if type(label_map) is not dict:   # can be list or dict_values list
        label_map = {idx+1: val for idx, val in enumerate(label_map)}

    colors = [[255,0,0],[0,255,0],[0,0,255],[255,255,0],[255,0,255],[0,255,255],[255,128,0],[255,0,128],[128,255,128],[0,128,255],[128,128,128],[185,170,155]]
    xmlpre = '<?xml version="1.0" encoding="UTF-8"?> <CaretExtension>  <Date><![CDATA[2013-07-14T05:45:09]]></Date>   <VolumeInformation Index="0">   <LabelTable>'

    body = ''
    for label_id, label_name in label_map.items():
        rgb = colors[label_id%len(colors)]
        body += f'<Label Key="{label_id}" Red="{rgb[0]/255}" Green="{rgb[1]/255}" Blue="{rgb[2]/255}" Alpha="1"><![CDATA[{label_name}]]></Label>\n'

    xmlpost = '  </LabelTable>  <StudyMetaDataLinkSet>  </StudyMetaDataLinkSet>  <VolumeType><![CDATA[Label]]></VolumeType>   </VolumeInformation></CaretExtension>'
    xml = xmlpre + "\n" + body + "\n" + xmlpost + "\n              "

    img_in.header.extensions.append(nib.nifti1.Nifti1Extension(0,bytes(xml,'utf-8')))

    return img_in


def save_multilabel_nifti(img, output_path, label_map, nora_project=None):
    """
    img: nifti image
    output_path: the output path
    label_map: a dictionary with label ids and names | a list of names and a running id will be generated starting at 1
    nora_project: if provided the file will be tagged as 'atlas'
    """
    img = add_label_map_to_nifti(img, label_map)
    nib.save(img, output_path)
    if nora_project is not None:
        subprocess.call(f"/opt/nora/src/node/nora -p {nora_project} --add {str(output_path)} --addtag atlas", shell=True)


def load_multilabel_nifti(img_path):
    """
    img_path: path to the image
    returns:
        img: nifti image
        label_map: a dictionary with label ids and names
    """
    import xmltodict
    img = nib.load(img_path)
    ext_header = img.header.extensions[0].get_content()
    ext_header = xmltodict.parse(ext_header)
    ext_header = ext_header["CaretExtension"]["VolumeInformation"]["LabelTable"]["Label"]
    label_map = {int(e["@Key"]): e["#text"] for e in ext_header}
    return img, label_map

