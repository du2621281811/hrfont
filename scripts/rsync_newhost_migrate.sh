#!/usr/bin/env bash
# Copy migrate tarballs + protocol A + Ec cache to another host.
# Usage: scripts/rsync_newhost_migrate.sh user@NEWHOST [/root/projects/hrfont]
set -euo pipefail
ROOT=/root/projects/hrfont
HOST="${1:?usage: $0 user@NEWHOST [remote_root]}"
REMOTE_ROOT="${2:-/root/projects/hrfont}"
PACK="$ROOT/artifacts/migrate_v100"

ssh "$HOST" "mkdir -p \
  $REMOTE_ROOT/artifacts/migrate_v100 \
  $REMOTE_ROOT/artifacts/f0 \
  $REMOTE_ROOT/data \
  $REMOTE_ROOT/runs"

if [[ -d "$PACK" ]]; then
  echo "==> tarballs -> $HOST:$REMOTE_ROOT/artifacts/migrate_v100/"
  rsync -aH --info=progress2 "$PACK"/ "$HOST:$REMOTE_ROOT/artifacts/migrate_v100/"
fi

echo "==> protocol A 0.7GB / 166k files"
rsync -aH --info=progress2 \
  "$ROOT/data/fontdiffuser-p253-t295-s338-cn2west-v2/" \
  "$HOST:$REMOTE_ROOT/data/fontdiffuser-p253-t295-s338-cn2west-v2/"

echo "==> Ec cache ~94GB"
rsync -aH --info=progress2 \
  "$ROOT/artifacts/f0/ec_multiscale_f0/" \
  "$HOST:$REMOTE_ROOT/artifacts/f0/ec_multiscale_f0/"

echo "on $HOST:"
echo "  git clone git@github.com:du2621281811/hrfont.git $REMOTE_ROOT   # if empty"
echo "  bash $REMOTE_ROOT/scripts/unpack_newhost_migrate.sh $REMOTE_ROOT/artifacts/migrate_v100"
echo "  # read docs/SETUP_COLLABORATOR.md"
