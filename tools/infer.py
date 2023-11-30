import hydra
from mmpose.apis import MMPoseInferencer
import os
from urllib.parse import urlparse
import cv2

@hydra.main(config_path='../configs', config_name='infer')
def run_inference(cfg):
    inferencer = MMPoseInferencer(
        pose2d=cfg.infer.pose2d,
        pose2d_weights=cfg.infer.pose2d_weights,
        pose3d=cfg.infer.pose3d,
        pose3d_weights=cfg.infer.pose3d_weights,
        device=cfg.infer.device,
        det_model=cfg.infer.det_model,
        det_weights=cfg.infer.det_weights,
        det_cat_ids=cfg.infer.det_cat_ids,
    )
  
    if "http" in cfg.input.path or cfg.input.path == "webcam":
        input_path = cfg.input.path
    else:
        input_path = hydra.utils.to_absolute_path(cfg.input.path)
    print("input: ", input_path)

    input_type = infer_input_type(input_path)
    # output = cfg.output
    print('input_type:', input_type)
    result_generator = inferencer(
        input_path,
        show=cfg.show,
        draw_heatmap=cfg.output.draw_heatmap,
        out_dir=cfg.output.out_dir
    )
    
    print("result_generator:")
    if input_type == 'image':
        print("infer image")
        result = next(result_generator)
        # Process the result for images
    elif input_type == 'video':
        print("infer video")
        results = [result for result in result_generator]
    elif input_type == 'folder':
        print("infer folder")
        files_in_folder = get_files_in_folder(input_path)
        for file_path in files_in_folder:
            result = next(result_generator)
    elif input_type == 'webcam':
        results = [result for result in result_generator]
            
    
def infer_input_type(path):
    if os.path.isfile(path):
        filename, file_extension = os.path.splitext(path)
        print("extension:", file_extension.lower())
        if file_extension.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
            return 'image'
        elif file_extension.lower() in ['.mp4', '.avi', '.mov']:
            return 'video'
    elif os.path.isdir(path):
        return 'folder'
    elif urlparse(path).scheme in ['http', 'https']:
        filename, file_extension = os.path.splitext(path)
        print("extension:", file_extension.lower())
        if file_extension.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
            return 'image'
        elif file_extension.lower() in ['.mp4', '.avi', '.mov']:
            return 'video'
    elif 'webcam' in path:
        return 'webcam'

def get_files_in_folder(folder_path):
    file_list = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_list.append(file_path)
    return file_list

if __name__ == "__main__":
    run_inference()