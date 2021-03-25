#!/bin/bash
#
# Run the "release" version of the model. Make sure everything is checked in and all the necessary files are in place.

BASE_DIR=$(pwd)
DATA_DIR="${BASE_DIR}/data"
LOG_DIR="${BASE_DIR}/logs/release"
SRC_DIR="${BASE_DIR}/src"
DRIVERS_TSV="${DATA_DIR}/drivers.tsv"
EVENTS_TSV="${DATA_DIR}/events.tsv"
RESULTS_TSV="${DATA_DIR}/results.tsv"
TEAM_HISTORY_TSV="${DATA_DIR}/team-history.tsv"

MAIN_PY="${SRC_DIR}/main.py"
if [[ ! -f "${MAIN_PY}" ]]
then
  echo "Could not find main program."
  exit 1
fi

for datafile in "${DRIVERS_TSV}" "${EVENTS_TSV}" "${RESULTS_TSV}" "${TEAM_HISTORY_TSV}"
do
  if [[ ! -f "${datafile}" ]]
  then
    echo "Could not find datafile \"${datafile}\""
    exit 1
  fi
done

# Check to see if any data files or code has changed and not been committed.
num_to_commit=$(git diff --name-only | grep -c -F "tsv
py")
if [[ "${num_to_commit}" -ne 0 ]]
then
  echo "Found ${num_to_commit} changed files which have not been checked in:"
  git diff --name-only | grep -F "tsv
py"
  echo "Please commit and re-run."
  exit 1
fi

current_time=$(date +"%Y%m%dT%H%M%S")
if [[ -z "${current_time}" ]]
then
  echo "Could not get current timestamp."
  exit 1
fi

current_git_tag=$(git rev-parse --short HEAD 2> /dev/null)
if [[ -z "${current_git_tag}" ]]
then
  echo "Could not get current git tag"
  exit 1
fi

python "${MAIN_PY}" "${DRIVERS_TSV}" "${EVENTS_TSV}" "${RESULTS_TSV}" "${TEAM_HISTORY_TSV}" \
  "${LOG_DIR}/${current_time}-${current_git_tag}" --logfile_uses_parameters