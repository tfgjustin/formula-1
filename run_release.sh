#!/bin/bash
#
# Run the "release" version of the model. Make sure everything is checked in and all the necessary files are in place.

BASE_DIR=$(pwd)
DATA_DIR="${BASE_DIR}/data"
MAIN_LOG_DIR="${BASE_DIR}/logs"
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

AGGREGATES_DRIVERS_PY="${SRC_DIR}/aggregates_drivers.py"
AGGREGATES_TEAMS_PY="${SRC_DIR}/aggregates_teams.py"
if [[ ! -f "${AGGREGATES_DRIVERS_PY}" || ! -f "${AGGREGATES_TEAMS_PY}" ]]
then
  echo "Could not find driver or team aggregate programs."
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

current_time=$(date +"%Y%m%dT%H%M%S")
if [[ -z "${current_time}" ]]
then
  echo "Could not get current timestamp."
  exit 1
fi

# Check to see if any data files or code has changed and not been committed.
log_dir=""
num_to_commit=$(git diff --name-only | grep -c -F "tsv
py")
if [[ "${num_to_commit}" -ne 0 ]]
then
  echo "Found ${num_to_commit} changed files which have not been checked in:"
  git diff --name-only | grep -F "tsv
py"
  log_dir="${MAIN_LOG_DIR}/test/${current_time}"
else
  log_dir="${MAIN_LOG_DIR}/release/${current_time}"
fi

current_git_tag=$(git rev-parse --short HEAD 2> /dev/null)
if [[ -z "${current_git_tag}" ]]
then
  echo "Could not get current git tag"
  exit 1
fi

mkdir -p "${log_dir}"
python "${MAIN_PY}" "${log_dir}/${current_git_tag}" "${DRIVERS_TSV}" "${EVENTS_TSV}" "${RESULTS_TSV}" \
	"${TEAM_HISTORY_TSV}" --logfile_uses_parameters --print_args
if [[ $? -ne 0 ]]
then
  echo "Run did not complete successfully."
  exit 1
fi

driver_ratings=$( find ${log_dir} -name '*.driver_ratings')
team_ratings=$( find ${log_dir} -name '*.team_ratings')
if [[ -z "${driver_ratings}" || -z "${team_ratings}" ]]
then
  echo "Cannot find driver or team ratings"
  exit 1
fi
if [[ ! -s "${driver_ratings}" || ! -s "${team_ratings}" ]]
then
  echo "Driver or team ratings files are empty"
  exit 1
fi

driver_aggregates="${driver_ratings%ratings}aggregates"
team_aggregates="${team_ratings%ratings}aggregates"
python "${AGGREGATES_DRIVERS_PY}" "${DRIVERS_TSV}" "${EVENTS_TSV}" "${driver_ratings}" "${driver_aggregates}"
if [[ $? -ne 0 ]]
then
  echo "Aggregating driver ratings did not complete successfully."
  exit 1
fi

python "${AGGREGATES_TEAMS_PY}" "${EVENTS_TSV}" "${team_ratings}" "${team_aggregates}"
if [[ $? -ne 0 ]]
then
  echo "Aggregating team ratings did not complete successfully."
  exit 1
fi
