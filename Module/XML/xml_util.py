
import os
import xml.etree.ElementTree as ET

script_dir = os.path.dirname(os.path.realpath(__file__))
xml_graph_sc_path = script_dir + "/build-xml-tree.sh"

def make_graph_image(xml_path, save_xmltreeData_path, save_xmlfig_path):
    # Make xml tree data
    os.system(f"{xml_graph_sc_path} {xml_path} > {save_xmltreeData_path}")

    os.system(f"dot -Tpng -Goverlap=false -Grankdir=LR {save_xmltreeData_path} > {save_xmlfig_path}")

    # Make graph image
    if os.path.exists(save_xmltreeData_path):
        print(f"{save_xmltreeData_path} is made completely!")
        
    if os.path.exists(save_xmlfig_path):
        print(f"{save_xmlfig_path} is made completely!")

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

if __name__ == "__main__":
    # target xml file path
    setup_scale_path = "/mnt/sdb2/DeepDraw/OpenSim/setup_scale.xml"
    
    # paths for saving result
    setup_xmltree_path = os.path.join(save_dir_path, Path(setup_scale_xml_path).stem + "0" + "_treeData" + ".dot")
    setup_xmlfig_path = os.path.join(save_dir_path, Path(setup_scale_xml_path).stem + "0" + "_treeFig" + ".png")
    
    # Make graph
    make_graph_image(xml_path = setup_scale_xml_path, 
                     save_xmltreeData_path = setup_xmltree_path, 
                     save_xmlfig_path= setup_xmlfig_path)
    
    # search tags
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    search_tags_in_xml(root, ["ScaleTool", "GenericModelMaker"])