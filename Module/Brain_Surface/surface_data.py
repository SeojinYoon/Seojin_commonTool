
# Common Libraries
import os
import warnings
import numpy as np
import nitools as nt
import nibabel as nb

# Functions
def surf_paths(surf_hemisphere: str, 
               surf_dir_path: str = "/home/seojin/Seojin_commonTool/Module/Brain_Surface/Datas", 
               surf_resolution: int = 32,
               sulcus_dummy_name: str = "sulcus",
               atlas: str = "Brodmann"):
    """
    load paths related to surface map

    :param surf_hemisphere: orientation of hemisphere ex) "L", "R"
    :param surf_dir_path: directory containing surface-related data
    :param surf_resolution: resolution of surface map
    :param sulcus_dummy_name: sulcus data name
    :param atlas: atlas name
    """
    
    # Template
    pial_surf_path = os.path.join(surf_dir_path, "Template", f"fs_LR.{surf_resolution}k.{surf_hemisphere}.pial.surf.gii")
    white_surf_path = os.path.join(surf_dir_path, "Template", f"fs_LR.{surf_resolution}k.{surf_hemisphere}.white.surf.gii")
    template_surface_path = os.path.join(surf_dir_path, "Template", f"fs_LR.{surf_resolution}k.{surf_hemisphere}.flat.surf.gii")
    inflated_brain_path = os.path.join(surf_dir_path, "Template", f"fs_LR.{surf_resolution}k.{surf_hemisphere}.inflated.surf.gii")
    shape_gii_path = os.path.join(surf_dir_path, "Template", f"fs_LR.32k.{surf_hemisphere}.shape.gii")
    
    # Sulcus
    sulcus_path = os.path.join(surf_dir_path, "Sulcus", f"{surf_hemisphere}_{sulcus_dummy_name}.json")
    
    # ROI
    roi_label_path = os.path.join(surf_dir_path, "ROI", atlas, f"{surf_hemisphere}_rois.npy")

    return {
        f"{surf_hemisphere}_pial_surf_path" : pial_surf_path,
        f"{surf_hemisphere}_white_surf_path" : white_surf_path,
        f"{surf_hemisphere}_template_surface_path" : template_surface_path,
        f"{surf_hemisphere}_inflated_brain_path" : inflated_brain_path,
        f"{surf_hemisphere}_shape_gii_path" : shape_gii_path,
        f"{surf_hemisphere}_sulcus_path" : sulcus_path,
        f"{surf_hemisphere}_roi_label_path" : roi_label_path,
    }

def vol_to_surf(volume_data_path, 
                pial_surf_path, 
                white_surf_path,
                ignoreZeros = False,
                depths = [0,0.2,0.4,0.6,0.8,1.0],
                stats = "nanmean"):
    """
    Adapted from https://github.com/DiedrichsenLab/surfAnalysisPy
    
    Maps volume data onto a surface, defined by white and pial surface.
    Function enables mapping of volume-based data onto the vertices of a
    surface. For each vertex, the function samples the volume along the line
    connecting the white and gray matter surfaces. The points along the line
    are specified in the variable 'depths'. default is to sample at 5
    locations between white an gray matter surface. Set 'depths' to 0 to
    sample only along the white matter surface, and to 0.5 to sample along
    the mid-gray surface.

    The averaging across the sampled points for each vertex is dictated by
    the variable 'stats'. For functional activation, use 'mean' or
    'nanmean'. For discrete label data, use 'mode'.

    If 'exclude_thres' is set to a value >0, the function will exclude voxels that
    touch the surface at multiple locations - i.e. voxels within a sulcus
    that touch both banks. Set this option, if you strongly want to prevent
    spill-over of activation across sulci. Not recommended for voxels sizes
    larger than 3mm, as it leads to exclusion of much data.

    For alternative functionality see wb_command volumne-to-surface-mapping
    https://www.humanconnectome.org/software/workbench-command/-volume-to-surface-mapping

    @author joern.diedrichsen@googlemail.com, Feb 2019 (Python conversion: switt)

    INPUTS:
        volume_data_path (string): nifti image path
        whiteSurfGifti (string or nibabel.GiftiImage): White surface, filename or loaded gifti object
        pialSurfGifti (string or nibabel.GiftiImage): Pial surface, filename or loaded gifti object
    OPTIONAL:
        ignoreZeros (bool):
            Should zeros be ignored in mapping? DEFAULT:  False
        depths (array-like):
            Depths of points along line at which to map (0=white/gray, 1=pial).
            DEFAULT: [0.0,0.2,0.4,0.6,0.8,1.0]
        stats (str or lambda function):
            function that calculates the Statistics to be evaluated.
            lambda X: np.nanmean(X,axis=0) default and used for activation data
            lambda X: scipy.stats.mode(X,axis=0) used when discrete labels are sampled. The most frequent label is assigned.
    OUTPUT:
        mapped_data (numpy.array):
            A Data array for the mapped data
    """
    # Stack datas
    depths = np.array(depths)
    
    # Load datas
    volume_img = nb.load(volume_data_path)
    whiteSurfGiftiImage = nb.load(white_surf_path)
    pialSurfGiftiImage = nb.load(pial_surf_path)
    
    whiteSurf_vertices = whiteSurfGiftiImage.darrays[0].data
    pialSurf_vertices = pialSurfGiftiImage.darrays[0].data
    
    assert whiteSurf_vertices.shape[0] == pialSurf_vertices.shape[0], "White and pial surfaces should have same number of vertices"
    
    # Informations
    n_vertex = whiteSurf_vertices.shape[0]
    n_point = len(depths)
    
    # 2D vertex location -> 3D voxel index with considering depth of graymatter
    voxel_indices = np.zeros((n_point, n_vertex, 3), dtype=int)
    for i in range(n_point):
        coeff_whiteMatter = 1 - depths[i]
        coeff_grayMatter = depths[i]
    
        weight_sum_vertex_2d = coeff_whiteMatter * whiteSurf_vertices.T + coeff_grayMatter * pialSurf_vertices.T
        voxel_indices[i] = nt.coords_to_voxelidxs(weight_sum_vertex_2d, volume_img).T
    
    # Read the data and map it
    data_consideringGraymatterDepth = np.zeros((n_point, n_vertex))
    
    ## Load volume array
    volume_array = volume_img.get_fdata()
    if ignoreZeros == True:
        volume_array[volume_array==0] = np.nan
    
    ## volume data without outside
    for i in range(n_point):
        data_consideringGraymatterDepth[i,:] = volume_array[voxel_indices[i,:,0], voxel_indices[i,:,1], voxel_indices[i,:,2]]
        outside = (voxel_indices[i,:,:]<0).any(axis=1) # These are vertices outside the volume
        data_consideringGraymatterDepth[i, outside] = np.nan
    
    # Determine the right statistics - if function - call it
    if stats == "nanmean":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mapped_data = np.nanmean(data_consideringGraymatterDepth,axis=0)
    elif callable(stats):
        mapped_data  = stats(data_consideringGraymatterDepth)
        
    return mapped_data

def load_surfData_fromVolume(volume_data_paths: list, 
                             hemisphere: str, 
                             depths: list = [0,0.2,0.4,0.6,0.8,1.0]):
    """
    Load surface data from volume data

    :param volume_data_paths: volume data path(.nii)
    :param hemisphere(string): "L" or "R"
    :param depths(list): Depths of points along line at which to map (0=white/gray, 1=pial). ex) [0.0,0.2,0.4,0.6,0.8,1.0]
    """
    surf_info = surf_paths(hemisphere)
    
    surface_datas = []
    for path in volume_data_paths:
        surface_data = vol_to_surf(volume_data_path = path,
                                   pial_surf_path = surf_info[f"{hemisphere}_pial_surf_path"],
                                   white_surf_path = surf_info[f"{hemisphere}_white_surf_path"],
                                   depths = depths)
        surface_datas.append(surface_data)
    surface_datas = np.array(surface_datas).T

    return surface_datas

def map_2d_to3d(volume_data_path: str,
                pial_surf_path: str,
                white_surf_path: str,
                depths: list = [0,0.2,0.4,0.6,0.8,1.0]) -> np.ndarray:
    """
    Map from 2D coord into 3D coords
    
    :param volume_data_path: nii file path
    :param pial_surf_path: gii file path of storing pial info
    :param white_surf_path: gii file path of storing white matter info
    :param depths: depths of points along line at which to map (0=white/gray, 1=pial).

    return 3d voxel indices per 2D coordinate
    """
    volume_img = nb.load(volume_data_path)
    whiteSurfGiftiImage = nb.load(white_surf_path)
    pialSurfGiftiImage = nb.load(pial_surf_path)
    
    # Stack datas
    depths = np.array(depths)
    
    # Load datas
    volume_img = nb.load(volume_data_path)
    whiteSurfGiftiImage = nb.load(white_surf_path)
    pialSurfGiftiImage = nb.load(pial_surf_path)
    
    whiteSurf_vertices = whiteSurfGiftiImage.darrays[0].data
    pialSurf_vertices = pialSurfGiftiImage.darrays[0].data
    
    assert whiteSurf_vertices.shape[0] == pialSurf_vertices.shape[0], "White and pial surfaces should have same number of vertices"
    
    # Informations
    n_vertex = whiteSurf_vertices.shape[0]
    n_point = len(depths)
    
    # 2D vertex location -> 3D voxel index with considering depth of graymatter
    voxel_indices = np.zeros((n_point, n_vertex, 3), dtype=int)
    for i in range(n_point):
        coeff_whiteMatter = 1 - depths[i]
        coeff_grayMatter = depths[i]
    
        weight_sum_vertex_2d = coeff_whiteMatter * whiteSurf_vertices.T + coeff_grayMatter * pialSurf_vertices.T
        voxel_indices[i] = nt.coords_to_voxelidxs(weight_sum_vertex_2d, volume_img).T
    return voxel_indices
    
if __name__ == "__main__":
    surf_paths("L")

    nii_path = "/mnt/ext1/seojin/temp/stat.nii"
    l_pial_surf_path = "/mnt/sda2/Common_dir/Atlas/Surface/fs_LR_32/fs_LR.32k.L.pial.surf.gii"
    l_white_surf_path = "/mnt/sda2/Common_dir/Atlas/Surface/fs_LR_32/fs_LR.32k.L.white.surf.gii"
    vol_to_surf(volume_data_path = nii_path,
                pial_surf_path = l_pial_surf_path,
                white_surf_path = l_white_surf_path)

    l_img_coords = map_2d_to3d(volume_data_path = nii_path,
                               pial_surf_path = l_pial_surf_path,
                               white_surf_path = l_white_surf_path,
                               depths = [0,0.2,0.4,0.6,0.8,1.0])
    