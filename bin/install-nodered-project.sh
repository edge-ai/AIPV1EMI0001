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
# when complete. npm ci vendors the dashboard's browser libraries (Vue, hls.js),
# so the dashboard works offline; this step needs npm registry access.
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
for tool in jq npm git tar; do
    command -v "$tool" > /dev/null || die "missing command: $tool"
done
[ "$(jq "[$CALLBACK] | length" "$AIP_ROOT/logic/flows.json")" = 1 ] \
    || die "logic/flows.json must contain exactly one AIPV1EMI0001 callback node"

TARGET=$PROJECTS_DIR/$PROJECT
if [ -e "$TARGET" ] && [ "$FORCE" != 1 ]; then
    die "$TARGET exists: pass --force to replace it"
fi

mkdir -p "$PROJECTS_DIR"
STAGE=$(mktemp -d "$PROJECTS_DIR/.$PROJECT.XXXXXX")
trap 'rm -rf "$STAGE"' EXIT

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

rm -rf "$TARGET"
mv "$STAGE" "$TARGET"
trap - EXIT

cat <<EOF
Installed Node-RED project $PROJECT in $TARGET (callback pipeline $PIPELINE_ID).
Next, in the Node-RED editor: Menu > Projects > Open > $PROJECT, then deploy.
Dashboard: https://<station>/ms5/aipv1emi0001_dash/
EOF
