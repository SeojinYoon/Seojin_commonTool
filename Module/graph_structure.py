
# Common Libraries
import inspect
import numpy as np
import pygraphviz as pgv

# Custom Libraries
from sj_enum import ConnectionType, GraphLayoutType
from sj_linux import exec_command, make_command
from sj_higher_function import flatten

# Classes
class WorkingTreeCursor:
    """
    Tree execution manager
    """
    
    def __init__(self):
        """
        Manager for executing working tree
        """

    def work_fromSup(self, work_tree):
        """
        Execute work from superior tree to current tree
        
        :param working_tree(WorkingTree): current tree
        """
        for suptree in work_tree.suptree:
            if suptree.result == None:
                self.work_fromSup(suptree)
        
        work_tree.work()
        
    def work_toSub(self, work_tree):
        """
        Execute work from current tree to sub tree
        
        :param working_tree(WorkingTree): current tree
        """
        if work_tree.result == None:
            work_tree.work()
        
        for tree in work_tree.subtree:
            self.work_toSub(tree)
        
    def work_all(self, work_tree):
        """
        Execute work from superior tree to sub tree
        
        :param working_tree(WorkingTree): current tree
        """
        for suptree in work_tree.suptree:
            if suptree.result == None:
                self.work_all(suptree)
        
        if work_tree.result == None:
            self.work_fromSup(work_tree)
        
        for tree in work_tree.subtree:
            self.work_all(tree)
            
            """
            print(tree)
            if tree.result == None:
                self.work_all(tree)
            """
            
class WorkingTree:
    """
    This class make tree structure including function to work.
    
    It has super tree and sub tree.
    """
    def __init__(self, 
                 func, 
                 name, 
                 pre_func = None,
                 post_func = None,
                 suptree = [],
                 is_command = False, 
                 arg_info = {}, 
                 pipeline_info = {},
                 check_result = None):
        """
        :param func(function or string): function to work, ex) lambda a: a + 3
        :param name(str): tree name, ex) "Tree1"
        :param pre_func(function): function to work previously
        :param post_func(function): function to work afterwards
        :param suptree(list - WorkingTree): super tree list
        :param is_command(boolean): True: execute command line, False: execute function
        :param arg_info(dictionary): argument info, ex) { "a" : 3 }
        :param pipeline_info(dictinary): command pipeline
        :param check_result(function): check function's result ex) labmda result: result == True
        """
        self.pre_func = pre_func
        self.func = func
        self.post_func = post_func
        self.name = name
        self.arg_info = arg_info
        self.pipeline_info = pipeline_info
        self.is_command = is_command
        self.subtree = []
        self.suptree = []
        self.result = None
        self.check_result = check_result
        self.is_valid_result = None
        if type(self.check_result) == type(None):
            if self.is_command == True:
                def check_(result):
                    return True if result == 0 else False
                
                self.check_result = check_

                
    def work(self):
        """
        execute work
        """
        if type(self.pre_func) != type(None):
            self.pre_func()
        
        if self.is_command:
            if self.arg_info == {}:
                result = os.system(self.func)
            else:
                result = exec_command(self.func, self.arg_info, self.pipeline_info)
        else:
            result = self.func(**self.arg_info)
        
        self.result = result
        
        if type(self.check_result) != type(None):
            self.is_valid_result = self.check_result(self.result)
            
            comm = self.command()
            assert self.is_valid_result, f"{self.name} is not working!!, {comm}, result: {self.result}"
        
        if type(self.post_func) != type(None):
            self.post_func()
            
        return result
    
    def command(self):
        """
        Make command line
        """
        if self.is_command:
            return make_command(command = self.func, parameter_info = self.arg_info, pipeline_info = self.pipeline_info)
        else:
            return str(self.func)
        
    def structure(self):
        """
        Return tree structure including all sub-tress
        
        return(list)
        """
        if len(self.subtree) == 0:
            return self
        
        subs = []
        for tree in self.subtree:
            subs.append(tree.structure())
        
        return [self, subs]
        
    @staticmethod
    def parse(structures, sup_tree = None):
        """
        Parsing list structure to working tree
        
        - Not list: super node
        - list: sub nodes

        Example:
            [root1, root2]: two roots
            [root1, [l1_1] ]: one root and one sub node
            [root1, [l1_1, l1_2] ]: one root and two sub nodes
            [root1, [l1_1, [l2_1]] ]: 
                root1
                    l1_1
                        l2_1
            [root1, [l1_1, [l2_1], l1_2, [l2_2]]]
                root1
                    l1_1
                        l2_1
                    l1_2
                        l2_2
        """
        trees = []
        for e in structures:
            if type(e) is list:
                # super tree
                last_tree = trees[len(trees) - 1]
                
                # Set subtree
                WorkingTree.parse(e, sup_tree = last_tree)
            else:
                if sup_tree == None:
                    pass
                else:
                    e.add_suptree(sup_tree)
                trees.append(e)
        
        return trees
    
    def __repr__(self):
        return self.name

    def visualize_structure(self):
        """
        Print structure of this tree 
        """
        structs = self.structure()
        nodes = flatten(structs)
        
        for node in nodes:
            print(node.name)
    
    def visualize_result(self, direction = "sup"):
        """
        Print result of this tree including all subtree
        """
        
        """
        structs = self.structure()
        nodes = flatten(structs)
        
        for node in nodes:
            print(str(node.name) + ", result: " + str(node.result))
        """

        if direction == "sub":
            print(str(self.name) + ", result: " + str(self.result))

            for tree in self.subtree:
                tree.visualize_result(direction = direction)
        elif direction == "sup":
            print(str(self.name) + ", result: " + str(self.result))
            
            for tree in self.suptree:
                tree.visualize_result(direction = direction)
    
    def visualize_command(self, tap = "  "):
        """
        Visualize command lines including all subtree
        """
        structs = self.structure()
        nodes = flatten(structs)
        
        for node in nodes:
            print()
            print(str(node.name) + ", command: " + node.command())
        
    def add_subtree(self, tree):
        """
        Add sub tree
        
        :param tree(WorkingTree): sub tree
        """
        is_dup = tree.name in [sub_tree.name for sub_tree in self.subtree]
        
        assert is_dup == False , "Duplicated " + tree.name
        
        self.subtree += [tree]
        tree.suptree += [self]
    
    def add_suptree(self, tree):
        """
        Add super tree
        
        :param tree(WorkingTree): super tree
        """
        is_dup = tree.name in [sub_tree.name for sub_tree in self.subtree]
        
        assert is_dup == False, "Duplicated " + tree.name
        
        self.suptree += [tree]
        tree.subtree += [self]
        
    def find(self, names):
        """
        Find tree iterating over all sub trees
        """
        if len(names) == 1:
            if self.name == names[0]:
                return self
            else:
                for node in self.subtree:
                    result = node.find(names)
                    if result != None:
                        return result
        else:
            if self.name == names[0]:
                return self.find(names[1:])
            elif len(self.subtree) == 0:
                return None
            else:
                for node in self.subtree:
                    result = node.find(names)
                    if result != None:
                        return result
                
        return None
    
    def node_json(self, direction = "sub", node_info = {}):
        """
        Get node information
        """
        if direction == "sub":
            node_info[id(self)] = {
                "name" : self.name,
                "is_valid_result" : self.is_valid_result
            }
            
            for tree in self.subtree:
                tree.node_json(direction = direction, node_info = node_info)
        elif direction == "sup":
            node_info[id(self)] = {
                "name" : self.name,
                "is_valid_result" : self.is_valid_result
            }
            
            for tree in self.suptree:
                tree.node_json(direction = direction, node_info = node_info)
            
    def link_json(self, direction = "sub"):
        """
        Get link infomration
        
        :param direction(str): Direction to draw graph
            - "sub": Draw sub trees 
            - "sub": Draw super trees 
        
        return (list - dictionary):
            [
                { source: id, target: id }
            ]
        """
        if direction == "sub":
            links = []
            for tree in self.subtree:
                links.append({"source" : id(self), "target" : id(tree)})

            for tree in self.subtree:
                links += tree.link_json(direction = direction)
        elif direction == "sup":
            links = []
            for tree in self.suptree:
                links.append({"source" : id(tree), "target" : id(self)})

            for tree in self.suptree:
                links += tree.link_json(direction = direction)
                
        return links
    
    def graph_json(self, direction = "sub"):
        """
        Get node and link infomration
        
        return 
            json(dictionary):
                -k, nodes,
                    [
                        { "id" : id }
                    ]
                -k, links,
                    [
                        { "source" : id, "target" : id }
                    ]

            label_info(dictioanry)
                -k id : name
        """
        node_info = {}
        self.node_json(direction = direction, node_info = node_info)
        
        id_lists = []
        for key in node_info.keys():
            id_lists.append({ "id" : key })
        
        json_d = {}
        json_d["nodes"] = id_lists
        json_d["links"] = self.link_json(direction = direction)
        
        return json_d, node_info
    
    def draw_graph(self, 
                   prog = GraphLayoutType.dot.value,
                   direction = "sub"):
        """
        Draw graph strcture
        """
        
        graph_info, label_info = self.graph_json(direction)
        
        G = pgv.AGraph(directed=True)
        for key in label_info:
            node_info = label_info[key]
            
            if node_info["is_valid_result"] == None:
                G.add_node(key, label = node_info["name"])
            elif node_info["is_valid_result"] == True:
                G.add_node(key, label = node_info["name"], fillcolor = "green", style = "filled", fontcolor = "white")
            elif node_info["is_valid_result"] == False:
                G.add_node(key, label = node_info["name"], fillcolor = "red", style = "filled", fontcolor = "white")
            
        for link in graph_info["links"]:
            src = link["source"]
            target = link["target"]

            G.add_edge(src, target)
            
        G.layout(prog=prog)
        return G

class WorkingForest:
    """
    This class make forest using trees
    """
    def __init__(self, root_trees):
        """
        :param root_trees(list - WorkingTree): Working tree list
        """
        self.root_trees = root_trees
    
    def find(self, names):
        """
        find tree using names
        
        :param names(list - str): name list
        """
        for tree in self.root_trees:
            result = tree.find(names)
            if result != None:
                return result
            
    def make_connection(self, root1_name, root2_name, tree_names1, tree_names2):
        """
        Make connection tree1 -> tree2
        
        :param root1_name(str): root1 name
        :param root2_name(str): root2 name
        :param tree_names1(list - str): name list
        :param tree_names2(list - str): name list
        """
        tree1 = self.find([root1_name] + tree_names1)
        tree2 = self.find([root2_name] + tree_names2)
        
        tree1.add_subtree(tree2)
    
    def graph_json(self):
        """
        Get node and link infomration
        
        return 
            json(dictionary):
                -k, nodes,
                    [
                        { "id" : id }
                    ]
                -k, links,
                    [
                        { "source" : id, "target" : id }
                    ]

            label_info(dictioanry)
                -k id : name
        """
        graph_infos = []
        label_infos = []
        for tree in self.root_trees:
            graph_info, label_info = tree.graph_json()
            graph_infos.append(graph_info)
            label_infos.append(label_info)
            
        node_graphs = []
        link_graphs = []
        
        merged_label = {}
        for i in range(len(graph_infos)):
            node_graphs.append(graph_infos[i]["nodes"])
            link_graphs.append(graph_infos[i]["links"])
            
            merged_label = {**merged_label, **label_infos[i]}
        
        merged_graph = {"nodes" : flatten(node_graphs), "links" : flatten(link_graphs) }
            
        return merged_graph, merged_label
    
    def draw_graph(self, 
                   prog = GraphLayoutType.dot.value):
        """
        Draw graph strcture
        """
        graph_info, label_info = self.graph_json()

        G = pgv.AGraph(directed=True)
        for key in label_info:
            node_info = label_info[key]
            G.add_node(key, label = node_info["name"])
        for link in graph_info["links"]:
            src = link["source"]
            target = link["target"]

            G.add_edge(src, target)
            
        G.layout(prog=prog)
        
        return G
        
class InstanceNode:
    """
    Node for representing instance
    """
    def __init__(self, 
                 instance, 
                 name):
        """
        :param instance(object): instance
        :param name(str): name
        """
        self.instance = instance
        self.name = name
        
    def call(self, func_name, arg_info):
        """
        Call function
        
        :param func_name(str): function name
        :param arg_info(dictionary): arg - value
        """
        func = getattr(self.instance, func_name)
        return func(**arg_info)
    
    def func_list(self):
        """
        function list of this instance
        
        return(list): name list
        """
        instance_informations = inspect.getmembers(self.instance, predicate = inspect.ismethod)
        func_names = [e[0] for e in instance_informations]
        func_names = list(filter(lambda x: not x.startswith("__"), func_names))
        
        return func_names
    
    def key_func(self, funcName):
        """
        key of function
        
        return(str)
        """
        sep = " / "
        return self.name + sep + funcName
    
    def key_func_list(self):
        func_names = self.func_list()
        func_name_keys = [self.key_func(func_name) for func_name in func_names]
        
        return func_name_keys
    
    def prop_list(self):
        return list(filter(lambda name: name != "world", vars(self.instance).keys()))
    
    def key_prop_list(self):
        prop_names = self.prop_list()
        names = [self.key_func(name) for name in prop_names]
        
        return names
    
    def link_json(self, type_ = "func"):
        func_names = self.func_list()
        func_name_keys = self.key_func_list()
        
        prop_names = self.func_list()
        prop_name_keys = self.key_prop_list()
            
        if type_ == "func":
            links = [{"source" : k_func_name, 
                      "target" : self.name, 
                      "connection_type" : "func"} for k_func_name in func_name_keys]
        elif type_ == "prop":
            links = [{"source" : name, "target" : self.name, "connection_type" : "prop"} for name in prop_name_keys]
        else:
            f_links = [{"source" : k_func_name, 
                      "target" : self.name, 
                      "connection_type" : "func"} for k_func_name in func_name_keys]
            p_links = [{"source" : name, "target" : self.name, "connection_type" : "prop"} for name in prop_name_keys]
            links = f_links + p_links
        return links
    
    def node_json(self, type_ = "func"):
        f_names = [self.name] + self.func_list()
        f_keys = [self.name] + self.key_func_list()
        
        p_names = [self.name] + self.prop_list()
        p_keys = [self.name] + self.key_prop_list()
        
        label_info = {}
        if type_ == "func":
            names = f_names
            keys = f_keys
        elif type_ == "prop":
            names = p_names
            keys = p_keys
        else:
            names = f_names + p_names
            keys = f_keys + p_keys
        for name, key in zip(names, keys):
            label_info[key] = name    
        return label_info
    
    def graph_json(self, type_ = "func"):
        node_info = self.node_json(type_ = type_)
        
        id_lists = []
        for key in node_info.keys():
            id_lists.append({ "id" : key })
            
        json_d = {}
        json_d["nodes"] = id_lists
        json_d["links"] = self.link_json(type_ = type_)
        
        return json_d, node_info
    
    def draw_graph(self, G = None, prog = GraphLayoutType.twopi.value, type_ = "func"):
        graph_info, label_info = self.graph_json(type_ = type_)

        if type(G) == type(None):
            G = pgv.AGraph(directed=True)

        f_keys = [self.name] + self.key_func_list()
        p_keys = [self.name] + self.key_prop_list()
        for key in label_info:
            if key == self.name:
                G.add_node(key, label = label_info[key], fillcolor = "black", style = "filled", fontcolor = "white")
            elif key in f_keys:
                G.add_node(key, label = label_info[key])
            elif key in p_keys:
                G.add_node(key, label = label_info[key], fillcolor = "blue")
            
        for link in graph_info["links"]:
            src = link["source"]
            target = link["target"]
            c_type = link["connection_type"]
            
            G.add_edge(src, target, arrowhead='diamond')
            
        G.layout(prog=prog)
        return G
        
class InstanceWorld():
    def __init__(self):
        self.instance_info = {}
        self.instance_connections = []
        
    def add_instance(self, instance, instance_name):
        """
        Add instance node
        
        :param instance(InstanceNode): instance node
        :param instance_name(string): name
        """
        is_exist = id(instance) in self.instance_info
        
        self.instance_info[id(instance)] = InstanceNode(instance, instance_name)
    
    def find_instance(self, instance_name):
        """
        find instance in instance node
        
        :param instance_name(string): name of instance

        return (python native instance)
        """
        result = self.find_instanceNode(instance_name)
        if type(result) != type(None):
            return result.instance
        else:
            return None
    
    def find_instanceNode(self, instance_name):
        """
        find instance node
        
        :param instance_name(string): name of instance

        return (InstanceNode)
        """
        instance_ids = np.array([i_id for i_id in self.instance_info])
        names = np.array([self.instance_info[i_id].name for i_id in self.instance_info])
        flags = [instance_name == name for name in names]
        
        n_find = sum(flags)
        if n_find == 0:
            return None
        
        assert n_find == 1, f"Multiple instances using {instance_name}"
        
        return self.instance_info[instance_ids[flags][0]]
    
    def call(self, 
             toIName, 
             to_funcName, 
             arg_info = {}, 
             connection_type = ConnectionType.ret,
             label = ""):
        cf = inspect.currentframe()
        f_back = cf.f_back
        return self.call2(fromI_ = f_back.f_locals["self"], 
                          from_funcName = f_back.f_code.co_name, 
                          toI_ = self.find_instance(toIName), 
                          to_funcName = to_funcName, 
                          arg_info = arg_info, 
                          connection_type = connection_type)
        
    
    def call2(self, 
              fromI_, 
              toI_, 
              from_funcName, 
              to_funcName, 
              arg_info = {}, 
              connection_type = ConnectionType.ret,
              label = ""):
        self.add_connection(fromI_, 
                            toI_, 
                            from_funcName,
                            to_funcName, 
                            connection_type, 
                            label)
        return self.instance_info[id(toI_)].call(to_funcName, arg_info)
        
    def graph_json(self):
        """
        Get node and link infomration
        
        return 
            json(dictionary):
                -k, nodes,
                    [
                        { "id" : id }
                    ]
                -k, links,
                    [
                        { "source" : id, "target" : id }
                    ]

            label_info(dictioanry)
                -k id : name
        """
        graph_infos = []
        label_infos = []
        for key in self.instance_info.keys():
            graph_info, label_info = self.instance_info[key].graph_json()
            graph_infos.append(graph_info)
            label_infos.append(label_info)
        
        node_graphs = []
        link_graphs = []
        
        merged_label = {}
        for i in range(len(graph_infos)):
            node_graphs.append(graph_infos[i]["nodes"])
            link_graphs.append(graph_infos[i]["links"])
            
            merged_label = {**merged_label, **label_infos[i]}
        
        merged_graph = {"nodes" : flatten(node_graphs), "links" : flatten(link_graphs) }
            
        return merged_graph, merged_label
    
    def draw_graph(self, prog = GraphLayoutType.twopi.value, is_interactionOnly = False):
        G = pgv.AGraph(directed=True)
        
        # instance connection info
        sources = [conn["source"] for conn in self.instance_connections]
        targets = [conn["target"] for conn in self.instance_connections]
        
        instance_names = [self.instance_info[key].name for key in self.instance_info]
        i_connection_nodeNames = list(set(sources + targets + instance_names))
        
        # Each instance info
        graph_info, label_info = self.graph_json()
        
        # Add instance's node
        for name in label_info:
            if name in instance_names:
                G.add_node(name, 
                           label = label_info[name], 
                           fillcolor = "black", 
                           style = "filled", 
                           fontcolor = "white")
            else:
                if is_interactionOnly:
                    if name in i_connection_nodeNames:
                        G.add_node(name, 
                                   label = label_info[name])
                else:
                    G.add_node(name, 
                               label = label_info[name])

        # Add instance's link
        for link in graph_info["links"]:
            src = link["source"]
            target = link["target"]

            if is_interactionOnly:
                if src in i_connection_nodeNames and target in i_connection_nodeNames:
                    G.add_edge(src, target, arrowhead='diamond')
            else:
                G.add_edge(src, target, arrowhead='diamond')
                
        # instance connection
        for e in self.instance_connections:
            connection_type = e.get("connection_type", ConnectionType.evoke)
            if connection_type == ConnectionType.ret:
                G.add_edge(e["target"], e["source"], style = "dashed", label = e["label"])
            else:
                G.add_edge(e["source"], e["target"], label = e["label"])
                
        G.layout(prog=prog)
        
        return G
    
    def add_connection(self, fromI_, toI_, fromI_funcName, toI_funcName, connection_type, label = ""):
        fromfunc_ = self.instance_info[id(fromI_)].key_func(fromI_funcName)
        tofunc_ = self.instance_info[id(toI_)].key_func(toI_funcName)
        
        self.instance_connections.append(
            {
                "source" : fromfunc_, 
                "target" : tofunc_, 
                "connection_type" : connection_type,
                "label" : label,
            })
        
class KnowledgeNode:
    """
    This class make knowledge structure including function to work.
    
    It has super knowledge and sub knowledge.
    """
    def __init__(self, 
                 name, 
                 sup = [],
                 sub = [],
                 links = []):
        """
        :param name(str): name, ex) "know1"
        :param sup(list - KnowledgeNode): super knowledge
        :param sub(list - KnowledgeNode): sub knowledge
        """
        self.name = name
        self.properties = [] # property
        self.sup = [] # super-ordinate concept
        self.sub = [] # sub-ordinate concept
        self.links = []
        
    def __repr__(self):
        return self.name
    
    def add_prop(self, property_, relation_label = ""):
        is_dup = property_.name in [prop.name for prop in self.properties]
        
        assert is_dup == False , "Duplicated " + property_.name
        
        self.properties += [property_]
        self.links += [ { "source" : id(self), 
                         "target" : id(property_), 
                         "connection_type" : "property", 
                         "label" : relation_label} 
                      ]
        
    def add_sub(self, knowledge, relation_label = ""):
        """
        Add sub knowledge
        
        :param knowledge(KnowledgeNode): sub knowledge
        """
        is_dup = knowledge.name in [sub.name for sub in self.sub]
        
        if is_dup:
            return
        
        self.sub += [knowledge]
        knowledge.add_sup(knowledge = self, relation_label = relation_label)
        self.links += [ { "source" : id(self), 
                         "target" : id(knowledge), 
                         "connection_type" : "inheritance", 
                         "label" : relation_label} ]
        
    def add_sup(self, knowledge, relation_label = ""):
        """
        Add super knowledge
        
        :param knowledge(KnowledgeNode): super knowledge
        """
        is_dup = knowledge.name in [sub.name for sub in self.sub]
        
        if is_dup:
            return
        
        self.sup += [knowledge]
        knowledge.add_sub(knowledge = self, relation_label = relation_label)
        self.links += [ { "source" : id(knowledge), 
                         "target" : id(self), 
                         "connection_type" : "inheritance", 
                         "label" : relation_label} ]
        
    def find(self, names):
        """
        Find knowledge iterating over all knowleges
        """
        if len(names) == 1:
            if self.name == names[0]:
                return self
            else:
                for node in self.sub:
                    result = node.find(names)
                    if result != None:
                        return result
        else:
            if self.name == names[0]:
                return self.find(names[1:])
            elif len(self.sub) == 0:
                return None
            else:
                for node in self.sub:
                    result = node.find(names)
                    if result != None:
                        return result
                
        return None
    
    def node_json(self, direction = "sub", node_info = {}):
        """
        Get node information
        """
        # Property
        for prop in self.properties:
            node_info[id(prop)] = {
                "name" : prop.name,
                "type" : "property",
            }
        
        if direction == "sub":
            node_info[id(self)] = {
                "name" : self.name,
                "type" : "knowledge",
            }

            for sub in self.sub:
                sub.node_json(direction = direction, node_info = node_info)
        elif direction == "sup":
            node_info[id(self)] = {
                "name" : self.name,
                "type" : "knowledge",
            }
            
            for sup in self.sup:
                sup.node_json(direction = direction, node_info = node_info)
            
    def link_json(self, direction = "sub"):
        """
        Get link infomration
        
        :param direction(str): Direction to draw graph
            - "sub": Draw sub knowledges 
            - "sub": Draw super knowledges 
        
        return (list - dictionary):
            [
                { source: id, target: id }
            ]
        """
        links = []
        for l_info in self.links:
            source = l_info["source"]
            target = l_info["target"]
            conn = l_info["connection_type"]

            if direction == "sub" and conn != "property":
                if source == id(self):
                    links.append(l_info)
            elif direction == "sup" and conn != "property":
                if target == id(self):
                    links.append(l_info)
            else:
                links.append(l_info)
                
        if direction == "sub":
            for sup in self.sub:
                links += sup.link_json(direction = direction)
        elif direction == "sup":
            for sup in self.sup:
                links += sup.link_json(direction = direction)
                
        return links
    
    def graph_json(self, direction = "sub"):
        """
        Get node and link infomration
        
        return 
            json(dictionary):
                -k, nodes,
                    [
                        { "id" : id }
                    ]
                -k, links,
                    [
                        { "source" : id, "target" : id }
                    ]

            label_info(dictioanry)
                -k id : name
        """
        node_info = {}
        self.node_json(direction = direction, node_info = node_info)
        
        id_lists = []
        for key in node_info.keys():
            id_lists.append({ "id" : key })
        
        json_d = {}
        json_d["nodes"] = id_lists
        json_d["links"] = self.link_json(direction = direction)
        
        return json_d, node_info
    
    def draw_graph(self, 
                   prog = GraphLayoutType.dot.value,
                   direction = "sub"):
        """
        Draw graph strcture
        """
        
        graph_info, label_info = self.graph_json(direction)
        
        G = pgv.AGraph(directed=True)
        for key in label_info:

            type_ = label_info[key]["type"]
            if type_ == "knowledge":
                G.add_node(key, label = label_info[key]["name"])
            elif type_ == "property":
                G.add_node(key, label = label_info[key]["name"], fillcolor = "black", style = "filled", fontcolor = "white")
            
        for link in graph_info["links"]:
            src = link["source"]
            target = link["target"]
            connection_type = link["connection_type"]
            
            if connection_type == "inheritance":
                G.add_edge(src, target, fillcolor = "white", label = link["label"])
            elif connection_type == "property":
                G.add_edge(src, target, fillcolor = "black", style = "filled", fontcolor = "white", arrowhead='diamond', label = link["label"])
                
        G.layout(prog=prog)
        return G

# Examples
if __name__=="__main__":
    # Working tree
    def clean(a):
        return a
    
    root = WorkingTree(func = clean, name = "root", is_command = False, arg_info = {"a" : 999})
    l1_1 = WorkingTree(func = clean, name = "l1_1", is_command = False, arg_info = {"a" : 1})
    l1_2 = WorkingTree(func = clean, name = "l1_2", is_command = False, arg_info = {"a" : 2})

    l2_1 = WorkingTree(func = clean, name = "l2_1", is_command = False, arg_info = {"a" : 3})
    l2_2 = WorkingTree(func = clean, name = "l2_2", is_command = False, arg_info = {"a" : 4})
    
    s = WorkingTree.parse([root, 
                           [l1_1, 
                            [l1_2], 

                            l2_1, 
                            [l2_2],
                           ]])
    a = Cursor(s[0])
    a.work_byDepth()

    # InstanceWorld
    class Adder:
        def __init__(self, world):
            self.world = world
            
        def add(self, a:int, b:int):
            s = a + b
            
            # 다른 인스턴스도 부를 수 있음 (예: Logger.log)
            count = self.world.call("logger", "log", {"msg": f"add({a},{b})={s}"})
            return s, count
    
    class Logger:
        def __init__(self, world):
            self.world = world
            self.count = 0
        def log(self, msg:str):
            self.count += 1
            print(f"[LOG {self.count}] {msg}")
            return self.count
    
    # 세계 생성/등록
    world = InstanceWorld()
    logger = Logger(world)
    adder  = Adder(world)
    world.add_instance(logger, "logger")
    world.add_instance(adder,  "adder")
    
    adder.add(3, 4)
    world.draw_graph(GraphLayoutType.circo.value)
    
    # RL
    class Environment:
        def __init__(self, world):
            self.world = world
    
        def observe(self):
            pass
        
        def step(self):
            pass
    
    class Agent:
        def __init__(self, world):
            self.world = world
    
        def select_action():
            pass
    
        def update():
            pass
            
    world = InstanceWorld()
    env = Environment(world)
    world.add_instance(env, "Environment")
    agent  = Agent(world)
    world.add_instance(agent,  "Agent")
    world.call2(fromI_ = agent, toI_ = env, from_funcName = "select_action", to_funcName = "observe", connection_type = ConnectionType.ret)
    world.call2(fromI_ = agent, toI_ = env, from_funcName = "select_action", to_funcName = "step", connection_type = ConnectionType.evoke)
    world.draw_graph(GraphLayoutType.circo.value)
    
