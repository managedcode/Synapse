#!/usr/bin/env bash
# REQ-WEB-002: collect data from authenticated, completed main workflow runs.
set -euo pipefail
destination="${1:?Expected a new output directory}"
repository="${GITHUB_REPOSITORY:?Expected the source repository}"
[[ "$repository" == managedcode/Synapse ]]
[[ ! -e "$destination" ]]
mkdir -p "$destination"
printf '{"verification":null,"performance":null}\n' > "$destination/runs.json"

collect_artifact() {
  local stream="$1" name="$2" filename="$3" metadata="$4" started="$5"
  local selected artifact_id digest archive target
  selected="$(jq -c --arg name "$name" --arg started "$started" \
    '[.artifacts[] | select(.name == $name and .expired == false and .created_at >= $started)]
     | sort_by(.created_at, .id) | last // empty' "$metadata")"
  [[ -n "$selected" ]] || return 0
  artifact_id="$(jq -r '.id' <<< "$selected")"
  digest="$(jq -r '.digest // empty' <<< "$selected")"
  [[ "$artifact_id" =~ ^[0-9]+$ ]]
  # Refuse legacy unhashed, oversized, or otherwise unverifiable artifacts.
  [[ "$digest" =~ ^sha256:[0-9a-f]{64}$ ]] || return 0
  [[ "$(jq -r '.size_in_bytes <= 16777216' <<< "$selected")" == true ]] || return 0
  archive="$destination/$artifact_id.zip"
  if ! gh api "repos/$repository/actions/artifacts/${artifact_id}/zip" > "$archive"; then
    rm -f "$archive"
    return 0
  fi
  [[ "$(shasum -a 256 "$archive" | awk '{ print $1 }')" == "${digest#sha256:}" ]]
  # Read one expected root member; never extract archive paths or executable files.
  if [[ "$(unzip -Z -1 "$archive" | awk -v file="$filename" '$0 == file { count++ } END { print count+0 }')" != 1 ]]; then
    rm "$archive"
    return 0
  fi
  target="$destination/$stream/$name"
  mkdir -p "$target"
  unzip -p "$archive" "$filename" | head -c 16777217 > "$target/$filename"
  [[ "$(wc -c < "$target/$filename")" -le 16777216 ]]
  rm "$archive"
  jq --arg stream "$stream" --argjson artifact "$selected" \
    '.[$stream].artifact_manifest += [$artifact | {id,name,digest,created_at}]' \
    "$destination/runs.json" > "$destination/runs.next.json"
  mv "$destination/runs.next.json" "$destination/runs.json"
}

for stream in verification performance; do
  workflow=verify
  [[ "$stream" != performance ]] || workflow=performance
  gh api "repos/$repository/actions/workflows/$workflow.yml/runs?branch=main&status=completed&per_page=100" \
    > "$destination/$stream-runs.json"
  run="$(jq -c --arg repository "$repository" \
    '[.workflow_runs[] | select(.head_branch == "main" and .head_repository.full_name == $repository
       and (.event == "push" or .event == "workflow_dispatch"))]
     | sort_by(.run_number, .run_attempt) | last // empty' "$destination/$stream-runs.json")"
  [[ -n "$run" ]] || continue
  run_id="$(jq -r '.id' <<< "$run")"
  [[ "$run_id" =~ ^[0-9]+$ ]]
  jq --arg stream "$stream" --argjson run "$run" \
    '.[$stream] = ($run | {id,run_number,run_attempt,head_sha,html_url,conclusion,updated_at,
      run_started_at,head_branch,event,repository:.head_repository.full_name,artifact_manifest:[]})' \
    "$destination/runs.json" > "$destination/runs.next.json"
  mv "$destination/runs.next.json" "$destination/runs.json"
  metadata="$destination/$stream-artifacts.json"
  gh api --paginate --slurp "repos/$repository/actions/runs/$run_id/artifacts?per_page=100" \
    | jq '{artifacts:[.[] | .artifacts[]]}' > "$metadata"
  started="$(jq -r '.run_started_at' <<< "$run")"
  if [[ "$stream" == verification ]]; then
    for runner in osx-arm64 linux-x64 win-x64; do
      collect_artifact "$stream" "test-results-$runner" test-results.json "$metadata" "$started"
    done
  else
    collect_artifact "$stream" performance-summary performance-results.json "$metadata" "$started"
  fi
  # A rerun may begin during collection; never attribute its artifacts to an old attempt.
  [[ "$(gh api "repos/$repository/actions/runs/$run_id" | jq -r \
    --argjson attempt "$(jq -r '.run_attempt' <<< "$run")" --arg started "$started" \
    --arg sha "$(jq -r '.head_sha' <<< "$run")" \
    '.status == "completed" and .run_attempt == $attempt and .run_started_at == $started and .head_sha == $sha')" == true ]]
done
