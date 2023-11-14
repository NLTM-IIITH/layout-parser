from datetime import datetime

from dateutil.tz import gettz
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

import os
from .modules.core.config import *

from .modules.cegis.routes import router as cegis_router
from .modules.main.routes import router as main_router
from .modules.postprocess.routes import router as postprocess_router
from .modules.preprocess.routes import router as preprocess_router

app = FastAPI(
	title='Layout Parser API',
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

if not os.path.exists(IMAGE_FOLDER):
	os.makedirs(IMAGE_FOLDER)


app.include_router(preprocess_router)
app.include_router(main_router)
app.include_router(cegis_router)
app.include_router(postprocess_router)
