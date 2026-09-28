#!/usr/bin/env bash
set -euo pipefail
apt-get update
apt-get install -y python3 python3-venv
python3 -m venv /opt/mpg-grader
/opt/mpg-grader/bin/pip install --disable-pip-version-check numpy==1.26.4
