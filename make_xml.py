import xml.etree.ElementTree as ET
from lxml import etree
DEFAULT_ENCODING = 'utf-8'


data = ET.Element('annotation')

folder = ET.SubElement(data, 'folder')


filename = ET.SubElement(data, 'filename')
path = ET.SubElement(data, 'path')


filename.set('type', 'Accepted')
path.set('type', 'Declined')


folder.text = 'img'
filename.text = "WhatsApp Image 2022-07-05 at 2.46.28 PM.jpeg"
path.text = "/home/naseem/Downloads/img/WhatsApp Image 2022-07-05 at 2.46.28 PM.jpeg"


sample_xml = ET.tostring(data, 'utf8')
root = etree.fromstring(sample_xml)

xml_str = etree.tostring(root, pretty_print=True, encoding=DEFAULT_ENCODING).replace("  ".encode(), "\t".encode())

with open('file.xml', 'w') as file:
    file.write(xml_str.decode('utf8'))

