#!/usr/bin/env bash

set -euo pipefail

postgres_pid=""

stop_postgres() {
    if [[ -n "$postgres_pid" ]]; then
        kill -TERM "$postgres_pid" 2>/dev/null || true
        wait "$postgres_pid" 2>/dev/null || true
    fi
    exit 0
}

trap stop_postgres TERM INT

while true; do
    /usr/local/bin/docker-entrypoint.sh "$@" &
    postgres_pid=$!

    status=0
    wait "$postgres_pid" || status=$?
    postgres_pid=""

    if [[ "$status" -ne 0 ]]; then
        exit "$status"
    fi

    sleep 1
done
