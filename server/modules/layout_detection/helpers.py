import cv2
from server.modules.layout_detection.classes.equations import get_equation_detection
from server.modules.layout_detection.classes.figures import get_figure_detection
from server.modules.layout_detection.classes.tables import get_tables_cells_detection
from server.modules.layout_detection.classes.utilities import mask_image

def get_layout_from_single_image(image_name):
    layout = {}
    image = cv2.imread(image_name)
    height, width, _ = image.shape
    table_bboxes, cell_bboxes = get_tables_cells_detection(image_name)
    masked_image = mask_image(image, table_bboxes)
    equation_bboxes = get_equation_detection(masked_image)
    masked_image = mask_image(image, equation_bboxes)
    figure_bboxes = get_figure_detection(masked_image)
    layout["image-name"] = image_name
    layout["height"] = height
    layout["width"] = width
    layout["tables"] = table_bboxes
    layout["cells"] = cell_bboxes
    layout['equations'] = equation_bboxes
    layout["figures"] = figure_bboxes
    #layout["masked-image"] = masked_image.tolist()

    #converting the numpy arrays to python lists
    layout["tables"] = [table.tolist() for table in layout["tables"]]
    layout["cells"] = [cell.tolist() for cell in layout["cells"]]
    layout["equations"] = [equation.tolist() if not isinstance(equation, list) else equation for equation in layout["equations"]]
    layout["figures"] = [figure.tolist() if not isinstance(figure, list) else figure for figure in layout["figures"]]


    return layout