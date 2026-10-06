
import cv2
import numpy as np

def render_skin_on2D(body_model,
                     renderer,
                     smpl_pose,
                     smpl_body_shape,
                     smpl_Rh,
                     smpl_Th,
                     camera_img,
                     camera_K,
                     camera_dist,
                     camera_R,
                     camera_T):
    """
    Render SMPL skin mesh on a 2D camera image.

    :param body_model: smpl body model
    :param renderer: renderer of easymocap
    :param smpl_pose: SMPL pose parameters (shape: #pos)
    :param smpl_body_shape: SMPL body shape parameters (shape: #body param).
    :param smpl_Rh: SMPL global rotation parameters (shape: xyz).
    :param smpl_Th: SMPL global translation parameters (shape: xyz).
    :param camera_img: Original camera image (height, width, 3).
    :param camera_K: Camera intrinsic matrix (shape: 3x3).
    :param camera_dist: Camera distortion coefficients (shape: 5x1).
    :param camera_R: Camera rotation matrix (shape: 3x3).
    :param camera_T: Camera translation vector (shape 3x1).
    
    :return: Rendered image with the SMPL mesh overlaid.
    """
    # Undistortion
    img = cv2.undistort(camera_img, camera_K, camera_dist, None)

    # Mesh
    smpl_param = {
        "poses": np.expand_dims(smpl_pose, 0),
        "shapes": np.expand_dims(smpl_body_shape, 0),
        "Rh": np.expand_dims(smpl_Rh, 0),
        "Th": np.expand_dims(smpl_Th, 0),
    }

    vertices = body_model(return_verts=True, 
                          return_tensor=False,
                          **smpl_param)[0]

    render_data = {
        0: {
            "vertices": vertices,
            "faces": body_model.faces,
            "vid": 0,
            "name": "body",
        }
    }

    camera_info = {
        "K": camera_K[None, :, :],
        "R": camera_R[None, :, :],
        "T": camera_T[None, :, :],
    }

    rendered = renderer.render(render_data,
                               camera_info,
                               [img],
                               add_back=True)[0]

    rendered = cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB)
    return rendered