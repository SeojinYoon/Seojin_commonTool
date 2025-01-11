
import numpy as np
import nibabel as nb
import nitools as nt
import warnings

def surface_cross_section(template_surface_path, 
                          surface_data, 
                          from_point, 
                          to_point, 
                          width,
                          n_sampling = None):
    """
    Do sampling using virtual strip

    :param template_surface_path(string): template gii file ex) '/home/seojin/single-finger-planning/data/surf/fs_LR.164k.L.flat.surf.gii'
    :param surface_data(string or np.array - shape: (#vertex, #data): data gii file path or data array ex) '/home/seojin/single-finger-planning/data/surf/group.psc.L.Planning.func.gii'
    :param from_point(list): location of start virtual strip - xy coord ex) [-43, 86]
    :param to_point(list): location of end virtual strip - xy coord ex) [87, 58]
    :param width(int): width of virtual strip ex) 20
    :param n_sampling(int): the number of sampling across virtual strip

    return 
        -k virtual_stip_mask(np.array - #vertex): mask
        -k sampling_datas(np.array - #sampling, #data): sampling datas based on virtual strip
        -k sampling_coverages(np.array - #sampling, #vertex): spatial coverage per sampling point
        -k sampling_center_coords(np.array - #sampling, #coord): sampling center coordinates
    """
    if n_sampling == None:
        n_sampling = abs(from_point[0] - to_point[0])
    
    # Load data metric file
    surface_gii = nb.load(template_surface_path)
    flat_coord = surface_gii.darrays[0].data
    
    if type(surface_data) == str:
        data_gii = nb.load(surface_data_path)
    
        # Check - all data has same vertex shape
        darrays = data_gii.darrays
        is_valid = np.all([darray.dims[0] == darrays[0].dims[0] for darray in darrays])
        assert is_valid, "Please check data shape"
    
        data_arrays = np.array([e.data for e in darrays]).T
    else:
        data_arrays = surface_data
        
    # Data information
    n_vertex = data_arrays.shape[0]
    n_data = data_arrays.shape[1]
    
    # Check - surface and data have same vertex
    assert flat_coord.shape[0] == n_vertex, "Data vertex must be matched with surface"
    
    # Extract vertices (x, y)
    vertex_2d = flat_coord[:, :2]
    
    # Move vertex origin
    points = vertex_2d - from_point
    
    # Set virtual vector(orientation of virtual strip)
    virtual_vec = to_point - from_point
    
    # Values for explaining vertex relative to virtual vector
    project = (np.dot(points, virtual_vec)) / np.dot(virtual_vec, virtual_vec)
    
    # Difference between vertex and projection vector
    residual = points - np.outer(project, virtual_vec)
    
    # Distance between vertex and virtual vector
    distance = np.sqrt(np.sum(residual**2, axis=1))

    ## Dummy for sampling result
    sampling_datas = np.zeros((n_sampling, n_data))
    virtual_stip_mask = np.zeros(n_vertex)
    sampling_center_coords = np.zeros((n_sampling, flat_coord.shape[1]))
    sampling_coverages = np.zeros((n_sampling, n_vertex))

    # Find points on the strip
    graduation_onVirtualVec = np.linspace(0, 1, n_sampling + 1)
    for i in range(n_sampling):
        # Filter only the vertices that are inside the virtual strip from all vertices
        start_grad = graduation_onVirtualVec[i]
        next_grad = graduation_onVirtualVec[i + 1]
    
        within_distance = distance < width
        upper_start = (project >= start_grad)
        lower_end = (project <= next_grad)
        no_origin = (np.sum(vertex_2d ** 2, axis=1) > 0)
        
        is_virtual_strip = within_distance & upper_start & lower_end & no_origin
        indx = np.where(is_virtual_strip)[0]

        sampling_coverages[i, indx] = 1
        
        # Perform cross-section
        sampling_datas[i, :] = np.nanmean(data_arrays[indx, :], axis=0) if len(indx) > 0 else 0
        virtual_stip_mask[indx] = 1
        sampling_center_coords[i, :] = np.nanmean(flat_coord[indx, :], axis=0) if len(indx) > 0 else 0

    result_info = {}
    result_info["sampling_datas"] = sampling_datas
    result_info["virtual_stip_mask"] = virtual_stip_mask
    result_info["sampling_center_coords"] = sampling_center_coords
    result_info["sampling_coverages"] = sampling_coverages
    return result_info

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
    voxel_indices = np.zeros((n_point, n_vertex, 3),dtype=int)
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
    
if __name__ == "__main__":
    # Parameters
    template_surface_path = '/home/seojin/single-finger-planning/data/surf/fs_LR.164k.L.flat.surf.gii'
    surface_data_path = '/home/seojin/single-finger-planning/data/surf/group.psc.L.Planning.func.gii'
    from_point = np.array([-43, 86])  # x_start, y_start
    to_point = np.array([87, 58])    # x_end, y_end
    width = 20
    
    values, mask, coords = surface_cross_section(template_surface_path = template_surface_path, 
                                                 urface_data_path = surface_data_path, 
                                                 from_point = from_point, 
                                                 to_point = to_point, 
                                                 width = width)

    vol_to_surf(volume_data_path = "/mnt/ext1/seojin/temp/stat.nii",
                pial_surf_path = "/mnt/sda2/Common_dir/Atlas/Surface/fs_LR_32/fs_LR.32k.L.pial.surf.gii",
                white_surf_path = "/mnt/sda2/Common_dir/Atlas/Surface/fs_LR_32/fs_LR.32k.L.white.surf.gii")

    