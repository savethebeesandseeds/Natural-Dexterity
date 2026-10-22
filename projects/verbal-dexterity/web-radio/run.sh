#!/bin/bash
# Service operations and tests; setup.sh only configures dependencies.
set -euo pipefail
cd /workspace
export PYTHONPATH=/workspace
export PYTHONDONTWRITEBYTECODE=1
PYTHON=/opt/web-radio/venv/bin/python
[ -f /opt/web-radio/ready ] || { echo 'Dependency setup has not completed; inspect container logs.' >&2; exit 1; }
case "$1" in
  start|stop|status|studio-stop|studio-start)
    exec "$PYTHON" -m web_radio.control "$1"
    ;;
  services-stop)
    exec "$PYTHON" -m web_radio.control stop
    ;;
  backup)
    exec "$PYTHON" -m web_radio.backup backup
    ;;
  logs)
    exec tail -n 60 /workspace/.runtime/control/studio.log /workspace/.runtime/control/station.log
    ;;
  test)
    exec "$PYTHON" /workspace/infra/test.py
    ;;
  check)
    exec "$PYTHON" /workspace/tests/check_services.py
    ;;
  gpu)
    exec nvidia-smi --query-gpu=name,uuid,driver_version --format=csv
    ;;
  *)
    echo 'Usage: run.sh start|stop|status|studio-stop|studio-start|services-stop|backup|logs|test|check|gpu' >&2
    exit 2
    ;;
esac

