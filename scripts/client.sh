#!/usr/bin/env bash
# This file is part of G05_A1Z, licensed under the Galaxea Dynamics A1Z License Agreement.
# See LICENSE.txt in the project root; upstream components retain their applicable licenses.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${G05_CLIENT_PYTHON:-python3}" "$ROOT/scripts/deploy.py" "$@"
