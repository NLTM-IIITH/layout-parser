import layoutparser as lp
from server.modules.layout_detection.classes.utilities import process_blocks_to_bboxes


def get_equation_detection(image):
    #MFD Model Equations
    model = lp.Detectron2LayoutModel(
        config_path='https://www.dropbox.com/s/ld9izb95f19369w/config.yaml?dl=1',  # In model catalog
        label_map={1: "Equation"},  # In model`label_map`
        extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8]  # Optional
    )
    layout = model.detect(image)
    equation_blocks = lp.Layout([b for b in layout if b.type == 'Equation'])
    equation_bboxes = process_blocks_to_bboxes(equation_blocks)
    return equation_bboxes
