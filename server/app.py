from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .modules.cegis.routes import router as cegis_router
from .modules.main.routes import router as main_router
from .modules.preprocess.routes import router as preprocess_router

app = FastAPI(
	title='Layout Parser API',
	description='',
	docs_url='/layout/docs',
	openapi_url='/layout/openapi.json'
)

app.add_middleware(
	CORSMiddleware,
	allow_origins=['*'],
	allow_methods=['*'],
	allow_headers=['*'],
	allow_credentials=True,
)

app.include_router(preprocess_router)
app.include_router(main_router)
app.include_router(cegis_router)
