# !/bin/bash
PORT=9999
python -m http.server --bind 127.0.0.1 --directory docs/build/html $PORT