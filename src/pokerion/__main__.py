"""Entry point: python -m pokerion

Serves the site on http://127.0.0.1:8000 for local use, with the Lab page
switched on. The deployed site is started by its container without this file,
so the Lab stays off there.
"""

import os

import uvicorn

from pokerion.server.app import app

os.environ.setdefault("POKERION_LAB", "1")

uvicorn.run(app, host="127.0.0.1", port=8000)
