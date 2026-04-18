#!/bin/bash

cd /opt/bioclip
source venv/bin/activate

nohup uvicorn inference:app --host 0.0.0.0 --port 8000 \
  > /opt/bioclip/bioclip.log 2>&1 &

echo "Started, PID=$!"
