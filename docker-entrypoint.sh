#!/bin/sh
set -eu

token_file="${STEGANO_STATE_DIR:-/state}/v2-api.key"
if [ -z "${STEGANO_API_KEY:-}" ]; then
    if [ ! -s "$token_file" ]; then
        umask 077
        python -c 'import secrets; print(secrets.token_urlsafe(32))' > "$token_file"
    fi
    STEGANO_API_KEY="$(sed -n '1p' "$token_file")"
    export STEGANO_API_KEY
fi

exec "$@"
