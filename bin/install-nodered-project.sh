#!/usr/bin/env bash
#
# install-nodered-project.sh — install this AIP's Node-RED flow (logic/) and
# UIBuilder dashboard (dashboard/) as a Node-RED project on an EdgeAI Station.
#
# Usage:
#   bin/install-nodered-project.sh --pipeline <pipeline-id> [--project <name>] [--force]
#
#   --pipeline ID   Pipeline whose callback feeds the dashboard (required)
#   --project NAME  Node-RED project name (default: nodered-aipv1emi0001-dashboard)
#   --force         Replace an existing project of the same name
#
#   NODE_RED_PROJECTS_DIR  Node-RED projects directory (default: /data/projects,
#                          shared by App-Builder and Node-RED on the station)
#
# The project is assembled in a temporary directory and moved into place only
# when complete. An existing project is never replaced without --force. With
# --force, the previous project is moved aside, not deleted, until the new one
# is in place; it is restored on failure, or kept and reported if that fails.
# npm ci vendors the dashboard's browser libraries (Vue, hls.js), so the
# dashboard works offline; this step needs npm registry access.
#
# Not done on purpose:
#   - flows_cred.json and node_modules/ are never copied.
#   - Node-RED runtime nodes (@edgematrix/node-red-contrib-eais, UIBuilder) are
#     not installed: they belong to the station's Node-RED runtime.
#   - The project is not opened: opening a project replaces the flows running
#     for every user of the station.
#
set -euo pipefail

AIP_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PROJECTS_DIR=${NODE_RED_PROJECTS_DIR:-/data/projects}
PROJECT=nodered-aipv1emi0001-dashboard
PIPELINE_ID=
FORCE=0
CALLBACK='.[] | select(.type == "callback" and .id == "aipv1emi0001callback")'

usage() { sed -n '5,15p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }
die() { echo "ERROR: $*" >&2; exit 1; }

while [ $# -gt 0 ]; do
    case $1 in
        --pipeline) [ $# -ge 2 ] || die "--pipeline needs a value"; PIPELINE_ID=$2; shift 2 ;;
        --project) [ $# -ge 2 ] || die "--project needs a value"; PROJECT=$2; shift 2 ;;
        --force) FORCE=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) usage >&2; die "unknown option: $1" ;;
    esac
done

[ -n "$PIPELINE_ID" ] || die "--pipeline is required: run the pipeline test first, or copy the ID from the station UI"
[[ $PROJECT =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || die "invalid project name: $PROJECT"
for tool in jq npm git tar python3; do
    command -v "$tool" > /dev/null || die "missing command: $tool"
done
[ "$(jq "[$CALLBACK] | length" "$AIP_ROOT/logic/flows.json")" = 1 ] \
    || die "logic/flows.json must contain exactly one AIPV1EMI0001 callback node"

TARGET=$PROJECTS_DIR/$PROJECT
exists() { [ -e "$1" ] || [ -L "$1" ]; }
if exists "$TARGET" && [ "$FORCE" != 1 ]; then
    die "$TARGET exists: pass --force to replace it"
fi

# rename(2) itself: mv would move a directory into an existing directory.
rename_path() {
    python3 -c 'import os, sys
try:
    os.rename(sys.argv[1], sys.argv[2])
except OSError as error:
    sys.exit(f"ERROR: {error}")' "$1" "$2"
}

# Move $1 to $2 without ever replacing an existing $2: mkdir claims the name
# atomically, and rename(2) replaces only that empty claim, failing if anything
# was written into it meanwhile.
place() {
    mkdir "$2" 2> /dev/null || return 1
    rename_path "$1" "$2" && return 0
    rmdir "$2" 2> /dev/null || true
    return 1
}

STAGE=
BACKUP=
# On failure, delete the staging copy and restore the project that --force
# moved aside. If it cannot be restored, keep it and say where it is.
finish() {
    local status=$?
    if [ -n "$BACKUP" ]; then
        if place "$BACKUP/$PROJECT" "$TARGET"; then
            rmdir "$BACKUP" 2> /dev/null || true
            echo "Restored the previous project in $TARGET" >&2
        else
            echo "ERROR: could not restore $TARGET; the previous project is kept in $BACKUP/$PROJECT" >&2
        fi
        [ "$status" != 0 ] || status=1
    fi
    if [ -n "$STAGE" ]; then rm -rf "$STAGE"; fi
    exit "$status"
}
trap finish EXIT
trap 'exit 1' HUP INT TERM

mkdir -p "$PROJECTS_DIR"
STAGE=$(mktemp -d "$PROJECTS_DIR/.$PROJECT.XXXXXX")

cp "$AIP_ROOT"/logic/{package.json,package-lock.json,README.md} "$STAGE/"
jq --indent 4 --arg id "$PIPELINE_ID" "($CALLBACK).pipelineId = \$id" \
    "$AIP_ROOT/logic/flows.json" > "$STAGE/flows.json"
mkdir "$STAGE/uibuilder"
tar -C "$AIP_ROOT/dashboard" --exclude=node_modules --exclude=README.md -cf - . \
    | tar -C "$STAGE/uibuilder" -xf -
printf 'node_modules/\nflows_cred.json\n.flows*.backup\n' > "$STAGE/.gitignore"

(cd "$STAGE/uibuilder" && npm ci --omit=dev --ignore-scripts --no-audit --no-fund)
for package in vue hls.js; do
    [ -f "$STAGE/uibuilder/node_modules/$package/package.json" ] || die "npm ci did not install $package"
done

git -C "$STAGE" init -q -b main
git -C "$STAGE" add -A
git -C "$STAGE" -c user.name="EdgeAI Station" -c user.email=edgeai@localhost \
    commit -q -m "Install AIPV1EMI0001 dashboard"

if exists "$TARGET"; then
    [ "$FORCE" = 1 ] || die "$TARGET was created during installation: pass --force to replace it"
    BACKUP=$(mktemp -d "$PROJECTS_DIR/.$PROJECT.previous.XXXXXX")
    if ! rename_path "$TARGET" "$BACKUP/$PROJECT"; then
        rmdir "$BACKUP"
        BACKUP=
        die "could not move $TARGET aside"
    fi
fi
if ! place "$STAGE" "$TARGET"; then
    if [ -z "$BACKUP" ] && exists "$TARGET"; then
        die "$TARGET was created during installation: pass --force to replace it"
    fi
    die "could not move the new project into $TARGET"
fi
STAGE=
PREVIOUS=$BACKUP
BACKUP=
trap - EXIT
if [ -n "$PREVIOUS" ]; then
    rm -rf "$PREVIOUS" || echo "WARNING: could not remove the previous project in $PREVIOUS" >&2
fi

cat <<EOF
Installed Node-RED project $PROJECT in $TARGET (callback pipeline $PIPELINE_ID).
Next, in the Node-RED editor: Menu > Projects > Open > $PROJECT, then deploy.
Dashboard: https://<station>/ms5/aipv1emi0001_dash/
EOF
