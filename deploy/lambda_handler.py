from __future__ import annotations

from mangum import Mangum

from src.api.main import app, load_dependencies

load_dependencies()

handler = Mangum(app, lifespan="off")
