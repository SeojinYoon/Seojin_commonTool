
import os
from pathlib import Path
import xml.etree.ElementTree as ET

script_dir = os.path.dirname(os.path.realpath(__file__))
xml_graph_sc_path = os.path.join(script_dir, "build-xml-tree.sh")

def make_graph_image(xml_path, output_dir_path):
    xmltree_path = os.path.join(output_dir_path, "_".join([Path(xml_path).stem, "treeData"]) + ".dot")
    xmlfig_path = os.path.join(output_dir_path, "_".join([Path(xml_path).stem, "treeFig"]) + ".png")
        
    # Make xml tree data
    os.system(f"{xml_graph_sc_path} {xml_path} > {xmltree_path}")

    os.system(f"dot -Tpng -Goverlap=false -Grankdir=LR {xmltree_path} > {xmlfig_path}")

    # Make graph image
    if os.path.exists(xmltree_path):
        print(f"{xmltree_path} is made completely!")
        
    if os.path.exists(xmlfig_path):
        print(f"{xmlfig_path} is made completely!")

def search_tags_in_xml(parent, tags):
    """
    This function searches for tags in an XML file, given a list of tags to traverse.
    
    :param parent(Element): Element
    :param tags(list - string): Variable number of tag names to search for, in the desired order.
    
    return: List of elements matching the specified tag path(list)
    """

    # Function to search for elements recursively
    def find_elements(parent, tag):
        elements = []
        for child in parent:
            if child.tag == tag:
                elements.append(child)
                elements += find_elements(child, tag)
        return elements

    # Start searching from the parent element, iterating through the provided tags
    current_elements = [parent]
    for tag in tags:
        next_elements = []
        for element in current_elements:
            next_elements += find_elements(element, tag)
        current_elements = next_elements

    return current_elements

def parse_xml_with_includes(xml_path: str) -> ET.ElementTree:
    """
    Read xml with include tag

    :param xml_path: xml path
    
    return root tag
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    base_dir = os.path.dirname(xml_path)
    
    while True:
        include_elem = root.find(".//include")
        if include_elem is None:
            break
            
        file_attr = include_elem.attrib.get("file")
        if file_attr:
            sub_xml_path = os.path.normpath(os.path.join(base_dir, file_attr))
            
            if os.path.exists(sub_xml_path):
                sub_root = parse_xml_with_includes(sub_xml_path)
                
                parent = root.find(f".//{include_elem.tag}/..")
                if parent is None:
                    parent = root
                    
                for child in list(sub_root):
                    parent.append(child)
                
                parent.remove(include_elem)
            else:
                parent = root.find(f".//{include_elem.tag}/..") or root
                parent.remove(include_elem)
        else:
            parent = root.find(f".//{include_elem.tag}/..") or root
            parent.remove(include_elem)
                
    return root
    
if __name__ == "__main__":
    # target xml file path
    setup_scale_path = "/mnt/sdb2/DeepDraw/OpenSim/setup_scale.xml"
    
    # Make graph
    make_graph_image(xml_path = setup_scale_xml_path, 
                     output_dir_path = "/mnt/ext1/seojin/temp")
    
    # search tags
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    search_tags_in_xml(root, ["ScaleTool", "GenericModelMaker"])
