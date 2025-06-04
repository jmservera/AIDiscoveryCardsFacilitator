#!/bin/bash
set -e
python3 -m pip install --upgrade pip
pip3 install -r requirements.txt
python3.12 -m chainlit run -h chainlit_app.py --port 8000 --host 0.0.0.0
