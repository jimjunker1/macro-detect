# function to convert xml output from CVAT

import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET

def xml_to_csv(path):
  xml_list = []
  for xml_file in glob.glob(path + '/*.xml'):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    if root.find('object'):
      for member in root.findall('object'):
        bbx = member.find('bndbox')
