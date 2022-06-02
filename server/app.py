from fastapi import FastAPI

from .modules.main.routes import router as main_router
from .modules.preprocess.routes import router as preprocess_router

app = FastAPI(
	title='Layout Parser API',
	description='',
	docs_url='/',
)

app.include_router(preprocess_router)
app.include_router(main_router)
