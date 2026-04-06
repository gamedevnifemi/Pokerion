"""Entry point: python -m pokerion"""

import uvicorn

from pokerion.server.app import app

uvicorn.run(app, host="127.0.0.1", port=8000)
