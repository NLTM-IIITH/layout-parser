from fastapi import APIRouter

from .scriptiden.routes import router as scriptiden_router
from .table.routes import router as table_router
from .textron.routes import router as textron_router

router = APIRouter(
    prefix='/layout/iitb',
    tags=['IITB APIs'],
)

router.include_router(table_router)
router.include_router(scriptiden_router)
router.include_router(textron_router)