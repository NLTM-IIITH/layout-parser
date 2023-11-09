from pydantic import BaseSettings

class OCRConfig(BaseSettings):
    # enter the model faster_rcnn_model_path here 
    #faster_rcnn_model_path: str = 'server\\modules\\layout_detection\\checkpoints\\model700.pth'
    det_threshold: float = 0.5
    table_recognition_language: str = "eng"
    row_determining_threshold: float = 0.6667
    col_determining_threshold: float = 0.5
    nms_table_threshold: float = 0.1
    nms_cell_threshold: float = 0.0001

class ColorConfig(BaseSettings):
    class_color_cells: tuple = (255, 165, 0)
    class_color_equations: tuple = (255, 0, 0)
    class_color_tables: tuple = (0, 0, 255)
    class_color_figures: tuple = (0, 255, 0)
