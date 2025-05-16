#!/bin/bash
set -euxo pipefail

HOST=203.57.40.101
PORT=10287

DEST_PROJ_DIR=/root/crew-ai

function run_remote() {
    ssh -p ${PORT} root@${HOST} $@
}

run_remote "mkdir -p ${DEST_PROJ_DIR}"
rsync --files-from 'rsync_include.txt' --exclude-from "rsync_exclude.txt" -rvz -e "ssh -T -p ${PORT}" ./ root@${HOST}:${DEST_PROJ_DIR}
scp -P ${PORT} ~/.tmux.conf ~/.tmux.conf.local root@${HOST}:~