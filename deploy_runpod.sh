#!/bin/bash
set -euxo pipefail

HOST=194.68.245.86
PORT=22009

DEST_PROJ_DIR=/workspace/crew-ai

function run_remote() {
    ssh -p ${PORT} root@${HOST} $@
}

run_remote "mkdir -p ${DEST_PROJ_DIR}"
rsync --exclude-from 'exclude_ssh.txt' -avz -e "ssh -T -p ${PORT}" ./ root@${HOST}:${DEST_PROJ_DIR}
