#!/bin/sh
# Runs via nginx's /docker-entrypoint.d before the server starts. Writes a tiny env.js so the SPA
# learns the backend URL at runtime (set AEGIS_API_URL in the cloud). Empty -> the app uses its
# built-in http://localhost:8080/api default (local compose, where the API is published on :8080).
set -e
if [ -n "${AEGIS_API_URL:-}" ]; then
  echo "window.AEGIS_API = \"${AEGIS_API_URL}\";" > /usr/share/nginx/html/env.js
else
  echo "/* no AEGIS_API_URL set; app uses its localhost default */" > /usr/share/nginx/html/env.js
fi
