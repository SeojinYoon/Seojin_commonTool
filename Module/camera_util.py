
# Common Libraries
import cv2
import numpy as np
from tqdm import tqdm

# Functions
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

def render_skin_video(body_model,
                      renderer,
                      pose_df,
                      pose_all,
                      shape_all,
                      Rh_all,
                      Gt_all,
                      video_path,
                      camera_K,
                      camera_dist,
                      camera_R,
                      camera_T,
                      frame_indices=None,
                      output_path="smpl_render.mp4",
                      fps=30):
    """
    Render SMPL skin meshes on camera frames and save them as a video.

    :param body_model: SMPL body model.
    :param renderer: Renderer of EasyMocap.
    :param pose_df: DataFrame containing the corresponding original video frame IDs.
    :param pose_all: SMPL pose parameters for all frames.
    :param shape_all: SMPL body shape parameters for all frames.
    :param Rh_all: SMPL global rotation parameters for all frames.
    :param Gt_all: SMPL global translation parameters for all frames.
    :param video_path: Path to the original camera video.
    :param camera_K: Camera intrinsic matrix (shape: 3x3).
    :param camera_dist: Camera distortion coefficients.
    :param camera_R: Camera rotation matrix (shape: 3x3).
    :param camera_T: Camera translation vector (shape: 3x1).
    :param frame_indices: Data indices to render. If None, render all available frames.
    :param output_path: Path for the output MP4 video.
    :param fps: Frames per second of the output video.
    :return: Output video path.
    """

    cap = cv2.VideoCapture(video_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_indices is None:
        frame_indices = np.arange(n_frames)
    else:
        frame_indices = np.asarray(frame_indices)
        
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    pose_frame_idx = pose_df[("3D", ".", "Frame_i")].values
    frame_to_pose_i = { int(frame_id): pose_i for pose_i, frame_id in enumerate(pose_frame_idx) }
    
    writer = None
    try:
        for i in tqdm(frame_indices, desc="Rendering", unit="frame"):
            i = int(i)
    
            # Load camera image
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, camera_img = cap.read()
    
            if not ret:
                print(f"Failed to read frame {i}")
                continue
    
            # Find corresponding SMPL pose
            pose_i = frame_to_pose_i.get(i)
    
            if pose_i is None:
                # No SMPL data -> keep original frame
                rendered_bgr = cv2.undistort(
                    camera_img,
                    camera_K,
                    camera_dist,
                    None
                )
    
            else:
                # Render SMPL
                rendered = render_skin_on2D(
                    body_model=body_model,
                    renderer=renderer,
                    smpl_pose=pose_all[pose_i],
                    smpl_body_shape=shape_all[pose_i],
                    smpl_Rh=Rh_all[pose_i],
                    smpl_Th=Gt_all[pose_i],
                    camera_img=camera_img,
                    camera_K=camera_K,
                    camera_dist=camera_dist,
                    camera_R=camera_R,
                    camera_T=camera_T,
                )
    
                rendered_bgr = cv2.cvtColor(
                    rendered,
                    cv2.COLOR_RGB2BGR
                )
    
            # Initialize writer
            if writer is None:
                height, width = rendered_bgr.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    
                writer = cv2.VideoWriter(
                    output_path,
                    fourcc,
                    fps,
                    (width, height)
                )
    
                if not writer.isOpened():
                    raise RuntimeError(
                        f"Could not create video: {output_path}"
                    )
    
            writer.write(rendered_bgr)
    
    finally:
        cap.release()
    
        if writer is not None:
            writer.release()

    return output_path