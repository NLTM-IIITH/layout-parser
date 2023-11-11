from Models.script_identification import AlexNet
from server.modules.script_identification.models import SIResponse
import json
from os.path import join
def process_output(path: str = "server/modules/script_identification/output.json"):
    """Processes output.json and returns in response format

    Args:
        path (str, optional): Path to output.json. Defaults to "server/modules/script_identification/output.json".

    Returns:
        List[SIResponse]: Processed output
    """
    try:
        with open(join(path, "output.json"), 'r') as json_file:
            loaded=json.load(json_file)
            ret = [SIResponse(text=i) for i in loaded]
            return ret
    except:
        print("Error while trying to open output file")