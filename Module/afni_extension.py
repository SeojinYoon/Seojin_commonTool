
# Common Libraries
from re import I
import subprocess
import pandas as pd
import numpy as np
import os
from functools import reduce
import nibabel as nb

# Custom Libraries
from sj_string import search_stringAcrossTarget, str_join
from stat_module import t_critical_value

def set_afni_abin(abin_path):
    """
    set abin path
    
    :param abin_path: path of afni binnary ex) "/Users/clmn/abin/afni"
    """
    os.environ['PATH'] = os.environ['PATH'] + ":" + abin_path
    
def clusterize(file_path, 
               threshold,
               testing_method = "2-sided",
               NN_level = 1, 
               cluster_size = 40, 
               is_show_command = False,
               is_parsing = True,
               orientation = "LPI",
               is_positive = False,
               pref_map = None,
               pref_dat = None,
               stat_index = 1):
    """
    clusetring analysis based on stat map
    https://afni.nimh.nih.gov/pub/dist/doc/program_help/3dClusterize.html

    :param file_path: path of statmap(string)
    :param threshold: threshold(float)
    :param testing_method: (string) ex) 1-sided_right, 1-sided_left, 2-sided
    :param NN_level: NN level(int)
    :param cluster_size: cluster size(int)
    :param is_show_command: whether showing command(boolean)
    :param is_parsing: whether parsing output(boolean)
    :param orientation: (string) ex) LPI, RAI
    :param is_positive: filter positive cluster only
    :param pref_map: cluster map cluster label result path(string)
    :param pref_dat: cluster map intensity result path(string)
    :param stat_index: statmap index of file_path

    return: (pd.DataFrame)
        -row: cluster info
    """
    # Testing method
    test_str = ""
    if testing_method == "1-sided_right":
        test_str = f"-1sided RIGHT_TAIL {threshold}"
    elif testing_method == "1-sided_left":
        test_str = f"-1sided LEFT_TAIL {threshold}"
    else:
        # testing_method == "2-sided":
        threshold = np.abs(threshold)
        l_threshold = -threshold
        u_threshold = threshold
        
        test_str = f"-2sided {l_threshold} {u_threshold}"
        
    # command
    command = str_join(strs = [
        "3dClusterize",
        f"-inset {file_path}",
        f"-idat {stat_index}",
        f"-NN {NN_level}",
        f"-clust_nvox {cluster_size}",
        f"-ithr {stat_index}",
        f"-orient {orientation}",
        test_str,
        f"-pref_map {pref_map}" if pref_map != None else "",
        f"-pref_dat {pref_dat}" if pref_dat != None else "",
    ], delimiter = " ")
    
    output = subprocess.check_output(command, shell=True)
    output = output.decode('utf-8').split("\n")
    
    if is_show_command:
        print("command: ", command)
    
    if not is_parsing:
        return output
    else:
        # cluster lines
        split_line_is = search_stringAcrossTarget(output, ["#-----"], return_type = "index")

        if len(split_line_is) == 0:
            return None
        
        # header
        report_header_i = search_stringAcrossTarget(output, ["#Volume"], return_type = "index")[0]
        header = output[report_header_i]
        header = list(filter(lambda x: len(x) > 0, header.split("  ")))
        
        # trim
        if len(split_line_is) > 1:
            raw_lines = [line.strip() for line in output[split_line_is[0] + 1:split_line_is[1]]]
            cluster_lines = []
            for line in raw_lines:
                cluster_lines.append(list(filter(lambda x: len(x) > 0, line.split(" "))))

        elif len(split_line_is) == 1:
            raw_lines = [line.strip() for line in output[split_line_is[0] + 1: split_line_is[0] + 2]]
            cluster_lines = []
            for line in raw_lines:
                cluster_lines.append(list(filter(lambda x: len(x) > 0, line.split(" "))))

        df = pd.DataFrame(cluster_lines)
        df.columns = header            

        if is_positive:
            return df[df["Mean"].astype(float) > 0]
        else:            
            return df
    
def whereami(x, 
             y, 
             z, 
             coord = "spm", 
             atlas = None, 
             is_show_command = False, 
             is_parsing = True):
    """
    Where is the location?

    https://afni.nimh.nih.gov/pub/dist/doc/program_help/whereami.html

    :param x: (int)
    :param y: (int)
    :param z: (int)
    :param coord: (string) spm, dicom
        -meaning spm: is equal to RAS+, lpi coords
        -meaning dicom: is equal to LPS+ rai coords
        
    :param atlas: (string)
        -Haskins_Pediatric_Nonlinear_1.0
        -CA_ML_18_MNI
        -and so on...
        
    return (pd.DataFrame)
        -row: atals info
    """
    
    if atlas != None:
        command = f"whereami {x} {y} {z} -{coord} -atlas {atlas}"
    else:
        command = f"whereami {x} {y} {z} -{coord}"
        
    if is_show_command:
        print(command)
        
    # command
    output = subprocess.check_output(command, shell=True)
    output = output.decode('utf-8').split("\n")
    
    if is_parsing == False:
        return output
    else:
        atlas_lines = search_stringAcrossTarget(output, 
                                                search_keys = ["Atlas"], 
                                                exclude_keys=["nearby"], 
                                                return_type = "index")
        
        search_atlas = []
        search_infos = []
        search_names = []
        for i in range(len(atlas_lines)):
            if len(atlas_lines) == 1:
                start_line_i = atlas_lines[0]
                end_line_i = search_stringAcrossTarget(output, 
                                                       search_keys = ["Please", "caution"], 
                                                       return_type = "index")[0]
            else:
                if i + 1 < len(atlas_lines):
                    # Search region name based on each atlas
                    start_line_i = atlas_lines[i]
                    end_line_i = atlas_lines[i+1] - 1
                else:
                    continue

            atlas_name = output[start_line_i].split(": ")[0].replace("Atlas ", "")
            selected_output = output[start_line_i:end_line_i]

            # search result
            search_results = search_stringAcrossTarget(selected_output, 
                                                       search_keys = ["Focus", "Within"], 
                                                       search_type = "any")

            for result in search_results:
                sp_result = result.split(":")

                info = sp_result[0].strip()
                name = sp_result[1].strip()

                search_atlas.append(atlas_name)
                search_infos.append(info)
                search_names.append(name)
            
            
        result_df = pd.DataFrame({
            "atlas" : search_atlas,
            "info" : search_infos,
            "name" : search_names,
        })

        return result_df


def LPSp_toRASp(xyz):
    """
    Convert LPS+(Dicom) coordinate to RAS+(spm, mni, lpi) coordinate
    
    :param xyz: LPS+ coord(list)
    
    return xyz(list)
    """
    x = xyz[0]
    y = xyz[1]
    z = xyz[2]
    
    return -x, -y, z

def RASp_toLPSp(xyz):
    """
    Convert RAS+(spm, mni, lpi) coordinate to LPS+(Dicom) coordinate
    
    :param xyz: RAS+ coord(list)
    
    return xyz(list)
    """
    x = xyz[0]
    y = xyz[1]
    z = xyz[2]
    
    return -x, -y, z

def reference2imageCoord(xyz, affine):
    """
    change reference coordinate to image coordinate
    reference coordinate can be scanner coordinate or MNI coordinate...
    
    :param xyz: anatomical coordinate(np.array)
    :param affine: affine matrix(np.array)
    
    return image coordinate(np.array)
    """
    
    result = np.matmul(np.linalg.inv(affine), [xyz[0], xyz[1], xyz[2], 1])[0:3]
    result = np.ceil(result).astype(int) # note: This is ad-hoc process - (np.ceil)
    return result

def cluster_infos(stat_map_paths, 
                  thresholds, 
                  cluster_sizes,
                  NN_level = 1,
                  atlas_name = "Haskins_Pediatric_Nonlinear_1.0",
                  atlas_query_method = "center",
                  orientation = "LPI",
                  is_positive = False,
                  pref_maps = None,
                  stat_indexes = None):
    """
    Get cluster infos
    
    :param stat_map_paths: statmap paths(list - string) 
    :param thresholds: stat threshold(list - float)
    :param cluster_sizes: cluster size threshold(list - int)
    :param NN_level: NN level(int)
    :param atlas_name: atlas name(string)
    :param atlas_query_method: atlas_query_method(string) ex) center, peak
    :param stat_indexes: statmap index of file_paths(list - int)

    return cluster_infos(list)
        -element: (pd.DataFrame)
    """
    if type(pref_maps) == type(None):
        pref_maps = np.repeat(None, len(stat_map_paths))
    if type(stat_indexes) == type(None):
        stat_indexes = np.repeat(1, len(stat_map_paths))

    clusterize_dfs = []
    for stat_map_path, threshold, cluster_size, pref_map, stat_index in zip(stat_map_paths, thresholds, cluster_sizes, pref_maps, stat_indexes):
        if type(threshold) == type(None):
            clusterize_dfs.append(None)
            continue

        clusterize_df = clusterize(file_path = stat_map_path,
                                   threshold = threshold,
                                   NN_level = NN_level,
                                   cluster_size = cluster_size,
                                   is_show_command = False,
                                   orientation = orientation,
                                   is_positive = is_positive,
                                   pref_map = pref_map,
                                   stat_index = stat_index)

        if type(clusterize_df) == type(None):
            clusterize_dfs.append(None)
            continue

        # Location of cluster
        if atlas_query_method == "center":
            cluster_locs = clusterize_df[["CM LR", "CM PA", "CM IS"]]
        elif atlas_query_method == "peak":
            cluster_locs = clusterize_df[["MI LR", "MI PA", "MI IS"]]

        # Query name of area based on atlas
        cluster_mms = []
        cluster_names = []
        for row_i in range(len(cluster_locs)):
            x, y, z = cluster_locs.iloc[row_i]

            loc_where = whereami(x = x, 
                                 y = y, 
                                 z = z, 
                                 is_parsing=True, 
                                 atlas = atlas_name,
                                 coord = "spm")

            if len(loc_where) > 0:
                cluster_mms.append(loc_where.iloc[0]["info"])
                cluster_names.append(loc_where.iloc[0]["name"]) # select first name of area
            else:
                cluster_mms.append("")
                cluster_names.append("")

        clusterize_df["mm"] = cluster_mms
        clusterize_df["name"] = cluster_names
        clusterize_dfs.append(clusterize_df)
        
    return clusterize_dfs

def find_thresholds_1samp(stat_path,
                          criteria_n_cluster, 
                          n_data, 
                          candidate_p_values = [0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001],
                          cluster_size = 40, 
                          NN_level = 1, 
                          is_positive = True,
                          stat_index = 1,
                          upper_limit_voxel_size = 200):
    """
    Find t-stat threshold from stat-map
    
    :param stat_path: stat-map path(t-stat)
    :param criteria_n_cluster: contraint of n_cluster
    :param n_data: #data (to calculate stat)
    :param candidate_p_values: p-values(list)
    :param cluster_size: voxel cluster size(int)
    :param NN_level: NN level(int)
    :param is_positive: Whether cluster is positive only(boolean)
    :param stat_index: statmap index of file_path(int)

    return (dictionary) {
        t_thres: t-stat threshold
        p_value: p-value
        n_cluster : number of cluster
    }
    """
    target_t_thres = None
    n_cluster = None
    p_value = None
    for c_p_value in candidate_p_values:
        # Take t-stat corresponding p-value
        t_thres = t_critical_value(c_p_value, df = n_data - 1)
        
        # Query clusterize analysis
        cluster_d = clusterize(file_path = stat_path,
                               threshold = t_thres, 
                               cluster_size = cluster_size,
                               NN_level = NN_level,
                               is_positive = is_positive,
                               stat_index = stat_index)

        # Validation check
        if type(cluster_d) == type(None) or len(cluster_d) == 0:
            continue
        
        # Check cluster which has too many voxels
        if sum(cluster_d["#Volume"].astype(int) > upper_limit_voxel_size) > 1:
            continue
        
        # Check the results exceeding maximum #cluster
        if len(cluster_d) <= criteria_n_cluster:
            n_cluster = len(cluster_d)
            target_t_thres = t_thres
            p_value = c_p_value
            
            break
            
    return {
        "t_thres" : target_t_thres,
        "p_value" : p_value,
        "n_cluster" : n_cluster
    }

def afni_to_nifti(afni_brain):
    # Get the AFNI data and header information
    data = afni_brain.get_fdata()
    afni_header = afni_brain.header

    # Create a new NIfTI image
    nifti_image = nb.Nifti1Image(data, afni_brain.affine, header=afni_header)
    
    return nifti_image

def read_1d(file_path):
    # Read data
    with open(file_path, 'r') as file:
        data = file.readlines()    
        
        headers = []
        header_idx = -1
        for i, d in enumerate(data):
            if "#" in d:
                header_idx = i
                headers.append(d)
        if header_idx == -1:
            pass
        else:
            data = data[header_idx + 1:]
    
    # Concat string
    data = " ".join(data)
    
    # Unify white space
    white_spaces = [" ", "\t", "\n"]
    for w_s in white_spaces:
        data = data.replace(w_s, " ")
    data = data.strip()
    
    # Remove white space
    def red(acc, cur):
        if len(acc) > 0 and acc[-1] == cur and cur in white_spaces:
            return acc
        else:
            return acc + cur
    data = reduce(lambda acc, cur: red(acc, cur), data, "")
    
    # Convert string to float
    data = [float(e) for e in data.split(" ")]
    
    return data, headers

if __name__ == "__main__":
    set_afni_abin("/Users/clmn/abin/afni")

    stat_map_paths = ["/Users/clmn/Downloads/vedo_vis/test_stat.nii"]
    stat_map_path = "/Users/clmn/Downloads/vedo_vis/test_stat.nii"

    thresholds = [3.3]
    cluster_sizes = [40]

    clusterize(file_path = stat_map_path, threshold = 3.13)
    whereami(10, 10, 10)
    cluster_infos(stat_map_paths = stat_map_paths,
                  thresholds = thresholds,
                  cluster_sizes = cluster_sizes)

    find_thresholds_1samp(stat_path = stat_map_path,
                      criteria_n_cluster = 8,
                      n_data = 6,
                      is_positive = True)