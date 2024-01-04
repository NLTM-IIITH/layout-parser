from datetime import datetime

from dateutil.tz import gettz
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from .modules.cegis.routes import router as cegis_router
from .modules.main.routes import router as main_router
from .modules.table.routes import router as table_router
from .modules.postprocess.routes import router as postprocess_router
from .modules.preprocess.routes import router as preprocess_router
from .modules.layout_detection.routes import router as layout_detection_router
from .modules.preprocessv2.routes import router as preprocess_router_v2

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

app.include_router(preprocess_router)
app.include_router(preprocess_router_v2)
app.include_router(main_router)
app.include_router(cegis_router)
app.include_router(postprocess_router)
app.include_router(layout_detection_router)
app.include_router(table_router)
