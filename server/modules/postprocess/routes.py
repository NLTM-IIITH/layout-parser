from subprocess import call
from tempfile import TemporaryDirectory

from fastapi import APIRouter

from .helper import process_images, process_layout_output
from .models import MIResponse, PostprocessRequest, SIResponse

router = APIRouter(
    prefix='/layout/postprocess',
    tags=['Postprocess'],
)

@router.post(
    '/language/printed',
    response_model=list[SIResponse],
    response_model_exclude_none=True,
)
def identify_printed_language(si_request: PostprocessRequest) -> list[SIResponse]:
    """
    This is the endpoint for classifying the language of the **printed** images.
    this model works for all the 14 language (13 Indian + english)

    API inputs a list of images in base64 encoded string and outputs a list
    of objects containing **"text"** as key and **language** as value
    """
    tmp = TemporaryDirectory(prefix='language_classify')
    process_images(si_request.images, tmp.name)
    call(f'./lang_iden_printed_v1.sh {tmp.name}', shell=True)
    return process_layout_output(tmp.name)

@router.post(
    '/language/handwritten',
    response_model=list[SIResponse],
    response_model_exclude_none=True,
)
def identify_handwritten_language(si_request: PostprocessRequest) -> list[SIResponse]:
    """
    This is the endpoint for classifying the language of the **handwritten** images.
    this model works for all the 14 language (13 Indian + english)

    API inputs a list of images in base64 encoded string and outputs a list
    of objects containing **"text"** as key and **language** as value
    """
    tmp = TemporaryDirectory(prefix='language_classify')
    process_images(si_request.images, tmp.name)
    call(f'./lang_iden_handwritten_v1.sh {tmp.name}', shell=True)
    return process_layout_output(tmp.name)

@router.post(
    '/language/pola/en_hi',
    response_model=list[SIResponse],
    response_model_exclude_none=True,
)
def identify_2class_hindi_english_language(si_request: PostprocessRequest) -> list[SIResponse]:
    """
    This is the endpoint for 2 class classification of the printed word images.
    this model works for all the 2 languages (hindi & english)

    API inputs a list of images in base64 encoded string and outputs a list
    of objects containing **"text"** as key and **language** as value
    """
    print('calling')
    tmp = TemporaryDirectory(prefix='language_classify')
    process_images(si_request.images, tmp.name)
    call(f'./lang_iden_2class_enhi.sh {tmp.name}', shell=True)
    return process_layout_output(tmp.name)

@router.post(
    '/language/pola/en_pa',
    response_model=list[SIResponse],
    response_model_exclude_none=True,
)
def identify_2class_hindi_english_language(si_request: PostprocessRequest) -> list[SIResponse]:
    """
    This is the endpoint for 2 class classification of the printed word images.
    this model works for all the 2 languages (punjabi & english)

    API inputs a list of images in base64 encoded string and outputs a list
    of objects containing **"text"** as key and **language** as value
    """
    print('calling')
    tmp = TemporaryDirectory(prefix='language_classify')
    process_images(si_request.images, tmp.name)
    call(f'./lang_iden_2class_enpa.sh {tmp.name}', shell=True)
    return process_layout_output(tmp.name)

@router.post(
    '/language/pola/hi_pa',
    response_model=list[SIResponse],
    response_model_exclude_none=True,
)
def identify_2class_hindi_punjabi_language(si_request: PostprocessRequest) -> list[SIResponse]:
    """
    This is the endpoint for 2 class classification of the printed word images.
    this model works for all the 2 languages (punjabi & hindi)

    API inputs a list of images in base64 encoded string and outputs a list
    of objects containing **"text"** as key and **language** as value
    """
    print('calling')
    tmp = TemporaryDirectory(prefix='language_classify')
    process_images(si_request.images, tmp.name)
    call(f'./lang_iden_2class_hipa.sh {tmp.name}', shell=True)
    return process_layout_output(tmp.name)

@router.post(
    '/language/pola/en_hi_pa',
    response_model=list[SIResponse],
    response_model_exclude_none=True,
)
def identify_3class_english_hindi_punjabi_language(si_request: PostprocessRequest) -> list[SIResponse]:
    """
    This is the endpoint for 3 class classification of the printed word images.
    this model works for all the 3 languages (english & punjabi & hindi)

    API inputs a list of images in base64 encoded string and outputs a list
    of objects containing **"text"** as key and **language** as value
    """
    tmp = TemporaryDirectory(prefix='language_classify')
    process_images(si_request.images, tmp.name)
    call(f'./lang_iden_3class_enhipa.sh {tmp.name}', shell=True)
    return process_layout_output(tmp.name)

@router.post(
    '/language/pola/{language}',
    response_model=list[SIResponse],
    response_model_exclude_none=True,
)
def identify_handwritten_language(si_request: PostprocessRequest, language: str) -> list[SIResponse]:
    """
    This is the endpoint for classifying the language of the **handwritten** images.
    this model works for all the 14 language (13 Indian + english)

    API inputs a list of images in base64 encoded string and outputs a list
    of objects containing **"text"** as key and **language** as value
    """
    print(language)
    tmp = TemporaryDirectory(prefix='language_classify')
    process_images(si_request.images, tmp.name)
    call(f'./lang_iden_pola.sh {tmp.name} {language}', shell=True)
    return process_layout_output(tmp.name)

@router.post(
    '/language/scenetext',
    response_model=list[SIResponse],
    response_model_exclude_none=True,
)
def identify_scenetext_language(si_request: PostprocessRequest) -> list[SIResponse]:
    """
    This is the endpoint for classifying the language of the **REAL** Scenetext images.
    this model works for all the 14 language (13 Indian + english)

    API inputs a list of images in base64 encoded string and outputs a list
    of objects containing **"text"** as key and **language** as value
    """
    tmp = TemporaryDirectory(prefix='language_classify')
    process_images(si_request.images, tmp.name)
    call(f'./lang_iden_scenetext_v1.sh {tmp.name}', shell=True)
    return process_layout_output(tmp.name)


@router.post(
    '/script',
    response_model=list[SIResponse],
    response_model_exclude_none=True
)
def identify_script(si_request: PostprocessRequest) -> list[SIResponse]:
    """
    This is an endpoint for identifying the script of the word images.
    this model was contributed by **Punjab university (@Ankur)** on 07-10-2022
    The endpoint takes a list of images in base64 format and outputs the
    identified script for each image in the same order.

    Currently 8 recognized languages are [**hindi, telugu, tamil, gujarati,
    punjabi, urdu, bengali, english**]

    API inputs a list of images in base64 encoded string and outputs a list
    of objects containing **"text"** as key and **language** as value
    """
    tmp = TemporaryDirectory(prefix='st_script')
    process_images(si_request.images, tmp.name)
    call(f'./script_iden_v1.sh {tmp.name}', shell=True)
    return process_layout_output(tmp.name)


@router.post(
    '/modality/v0',
    response_model=list[MIResponse],
    response_model_exclude_none=True,
)
def identify_modality(si_request: PostprocessRequest) -> list[MIResponse]:
    """
    This is the endpoint for classifying the modality of the images.
    this model works for all the 14 language (13 Indian + english) and
    outputs among 3 classes ["**printed**", "**handwritten**", "**scenetext**"]

    API inputs a list of images in base64 encoded string and outputs a list
    of objects containing **"text"** as key and **modality** as value
    """
    tmp = TemporaryDirectory(prefix='modality_classify')
    process_images(si_request.images, tmp.name)
    call(f'./modality_iden_v0.sh {tmp.name}', shell=True)
    return process_layout_output(tmp.name)

@router.post(
    '/modality/v1',
    response_model=list[MIResponse],
    response_model_exclude_none=True,
)
def detect_modality_apoorva_v1(si_request: PostprocessRequest) -> list[MIResponse]:
    """
    This is the endpoint for classifying the modality of the images.
    this model works for all the 14 language (13 Indian + english) and
    outputs among 3 classes ["**printed**", "**handwritten**", "**scenetext**"]

    API inputs a list of images in base64 encoded string and outputs a list
    of objects containing **"text"** as key and **modality** as value
    """
    tmp = TemporaryDirectory(prefix='modality_classify')
    process_images(si_request.images, tmp.name)
    call(f'./modality_iden_v1.sh {tmp.name}', shell=True)
    return process_layout_output(tmp.name)

@router.post(
    '/modality/v2',
    response_model=list[MIResponse],
    response_model_exclude_none=True,
)
def detect_modality_apoorva_v2(si_request: PostprocessRequest) -> list[MIResponse]:
    """
    This is the endpoint for classifying the modality of the images.
    this model works for all the 14 language (13 Indian + english) and
    outputs among 3 classes ["**printed**", "**handwritten**", "**scenetext**"]

    API inputs a list of images in base64 encoded string and outputs a list
    of objects containing **"text"** as key and **modality** as value
    """
    tmp = TemporaryDirectory(prefix='modality_classify')
    process_images(si_request.images, tmp.name)
    call(f'./modality_iden_v2.sh {tmp.name}', shell=True)
    return process_layout_output(tmp.name)



@router.post(
    '/language/iitj/{language}',
    response_model=list[SIResponse],
    response_model_exclude_none=True,
)
def identify_iitj_script(si_request: PostprocessRequest, language: str) -> list[SIResponse]:
    """
    API inputs a list of images in base64 encoded string and outputs a list
    of objects containing **"text"** as key and **language** as value
    """
    print(language)
    tmp = TemporaryDirectory(prefix='language_classify')
    process_images(si_request.images, tmp.name)
    call(f'./lang_iden_pola.sh {tmp.name} {language}', shell=True)
    return process_layout_output(tmp.name)
