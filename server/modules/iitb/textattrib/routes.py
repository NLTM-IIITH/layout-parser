from subprocess import check_output
from tempfile import TemporaryDirectory

from fastapi import APIRouter

from .helper import process_images, process_output
from .models import PostprocessRequest, SIResponse

router = APIRouter(
    prefix='/script',
)

@router.post('', response_model=list[SIResponse])
async def identify_script(
    si_request: PostprocessRequest,
) -> list[SIResponse]:
    """
    API endpoint for calling the layout parser
    """
    temp = TemporaryDirectory()
    folder = temp.name

    process_images(si_request.images, folder)
    check_output([
        'docker',
        'run',
        '--rm',
        '--net',
        'host',
        '-v',
        f'{folder}:/model/data',
        'layout:iitb-scriptiden'
    ])
    return process_output(folder)