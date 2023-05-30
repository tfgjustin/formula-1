#!/bin/bash

BASE_DIR=$(pwd)
DATA_DIR="${BASE_DIR}/data"
SRC_DIR="${BASE_DIR}/src"
PROFILES_DIR="${BASE_DIR}/profiles"
DRIVERS_TSV="${DATA_DIR}/drivers.tsv"
EVENTS_TSV="${DATA_DIR}/events.tsv"
RESULTS_TSV="${DATA_DIR}/results.tsv"
GRID_PENALTIES_TSV="${DATA_DIR}/grid-penalties.tsv"
TEAM_ADJUST_TSV="${DATA_DIR}/team-adjust.tsv"
TEAM_HISTORY_TSV="${DATA_DIR}/team-history.tsv"
FUTURE_EVENTS_TSV="${DATA_DIR}/future-events.tsv"
FUTURE_LINEUP_TSV="${DATA_DIR}/future-lineup.tsv"

MAIN_PY="${SRC_DIR}/simulator.py"
if [[ ! -f "${MAIN_PY}" ]]
then
  echo "Could not find main program: ${MAIN_PY}"
  exit 1
fi

current_time=$(date +"%Y%m%dT%H%M%S")
if [[ -z "${current_time}" ]]
then
  echo "Could not get current timestamp."
  exit 1
fi

PROFILE_SIMULATION=${PROFILE_SIMULATION:-0}
PROFILE_OPTS=""
if [[ ${PROFILE_SIMULATION} -ne 0 ]]
then
  if [[ ! -d "${PROFILES_DIR}" ]]
  then
    mkdir -p "${PROFILES_DIR}"
    if [[ ! -d "${PROFILES_DIR}" ]]
    then
      echo "Profiling enabled but could not find or make output directory ${PROFILES_DIR}. Exiting."
      exit 1
    fi
  fi
  PROFILE_OPTS="-m cProfile -o \"${PROFILES_DIR}/simulate.profile.${current_time}.dat\""
fi

for datafile in "${DRIVERS_TSV}" "${EVENTS_TSV}" "${RESULTS_TSV}" "${TEAM_ADJUST_TSV}" "${TEAM_HISTORY_TSV}"
do
  if [[ ! -f "${datafile}" ]]
  then
    echo "Could not find datafile \"${datafile}\""
    exit 1
  fi
done

ratings_dir=$(find logs -type d -name '????????T??????' | sort -t / -k 3,3 | tail -1)
if [[ $# -ge 1 ]]
then
  ratings_dir=$(find logs -type d -name "$1" | sort -t / -k 3,3 | tail -1)
elif [[ $# -ne 0 ]]
then
  echo "Invalid usage" > /dev/stderr
  echo "Usage: $0 [dir_timestamp]" > /dev/stderr
  exit 1
fi


if [[ ! -d "${ratings_dir}" ]]
then
  echo "Could not find ratings directory ${ratings_dir}" > /dev/stderr
  exit 1
fi

args_file=$( find "${ratings_dir}" -name '*.args')
if [[ -z "${args_file}" ]]
then
  echo "Cannot find args file"
  exit 1
fi
if [[ ! -s "${args_file}" ]]
then
  echo "Args file is empty: ${args_file}"
  exit 1
fi
driver_ratings=$( find "${ratings_dir}" -name '*.driver_ratings')
team_ratings=$( find "${ratings_dir}" -name '*.team_ratings')
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

tag="simulate"
if [[ $# -eq 2 ]]
then
  tag=$2
fi

# positional arguments:
#   logfile               Write results to this logfile.
#   drivers_tsv           TSV file with the list of drivers.
#   events_tsv            TSV file with the list of events.
#   results_tsv           TSV file with the list of results.
#   teams_tsv             TSV file with the history of F1 teams.
#   team_adjustments_tsv  TSV file with adjustments for team ratings (current and EOY)
#   future_events_tsv     TSV file containing list of future events.
#   driver_ratings_tsv    TSV file with the log of driver ratings.
#   team_ratings_tsv      TSV file with the log of team ratings.
#   grid_penalties_tsv    TSV file with the log of grid penalties.
#
# optional arguments:
#   --future_lineup_tsv FUTURE_LINEUP_TSV
#                         TSV file containing driver and team lineup for future events.
#   --simulate_season SIMULATE_SEASON
#                         Season to simulate.
#   --simulate_start_round SIMULATE_START_ROUND
#                         The first round in the season we will simulate; if "XX" start with the final real event.
python3 ${PROFILE_OPTS} "${MAIN_PY}" "${ratings_dir}/${tag}" \
  "${DRIVERS_TSV}" "${EVENTS_TSV}" "${RESULTS_TSV}" "${TEAM_HISTORY_TSV}" "${FUTURE_EVENTS_TSV}" \
  "${TEAM_ADJUST_TSV}" "${driver_ratings}" "${team_ratings}" "${GRID_PENALTIES_TSV}" "@${args_file}" \
  --logfile_uses_parameters --future_lineup_tsv "${FUTURE_LINEUP_TSV}"

if [[ $? -ne 0 ]]
then
  echo "Run did not complete successfully."
  exit 1
fi
