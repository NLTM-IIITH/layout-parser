from datetime import datetime

from dateutil.tz import gettz
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from .modules.textron_api.routes import router as textron_router
from .modules.doctr_api.routes import router as doctr_router

from .modules.cegis.routes import router as cegis_router
from .modules.main.routes import router as main_router
from .modules.postprocess.routes import router as postprocess_router
from .modules.preprocess.routes import router as preprocess_router

app = FastAPI(
	title='Layout API',
	description='',
	docs_url='/layout/docs',
	openapi_url='/layout/openapi.json'
)

@app.middleware('http')
async def log_request_timestamp(request: Request, call_next):
	local_tz = gettz('Asia/Kolkata')
	print(f'Received request at: {datetime.now(tz=local_tz).isoformat()}')
	return await call_next(request)

app.add_middleware(
	CORSMiddleware,
	allow_origins=['*'],
	allow_methods=['*'],
	allow_headers=['*'],
	allow_credentials=True,
)

app.include_router(textron_router)
app.include_router(doctr_router)
app.include_router(preprocess_router)
app.include_router(main_router)
app.include_router(cegis_router)
app.include_router(postprocess_router)
