#!/bin/bash
set -euo pipefail
cd /workspace
rm -f /opt/web-radio/ready
/bin/bash /workspace/setup.sh
printf 'ready\n' > /opt/web-radio/ready
exec sleep infinity
