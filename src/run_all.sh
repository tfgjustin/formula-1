#!/bin/bash

set -e

DRIVERS_TSV="drivers.tsv"
RACES_TSV="races.tsv"
RESULTS_TSV="results.tsv"
ELO_RATINGS_TSV="elo_ratings.tsv"
FLATTENED_RATINGS_TSV="flat_ratings.tsv"
METRICS_TSV="metrics.tsv"

if [[ $# -ne 1 ]]
then
  echo "Usage: $0 <input_file>"
  exit 1
fi

if [[ ! -f $1 ]]
then
  echo "No input file $1"
  exit 1
fi

# Generate the raw input files
echo -n "Transforming data from $1 ... "
python3 sqlite_to_tsv.py "$1" "${DRIVERS_TSV}" "${RACES_TSV}" "${RESULTS_TSV}"
echo "done."

echo -n "Calculating driver ratings ... "
# Now generate the per-event Elo rating TSV
python3 elo_ratings.py "${RESULTS_TSV}" "${ELO_RATINGS_TSV}"
echo "done."

echo -n "Flattening ratings ... "
python3 flatten_ratings.py "${ELO_RATINGS_TSV}" "${FLATTENED_RATINGS_TSV}"
echo "done."

echo -n "Calculating aggregated metrics ... "
python3 calculate_aggregates.py "${DRIVERS_TSV}" "${RACES_TSV}" \
  "${FLATTENED_RATINGS_TSV}" "${METRICS_TSV}"
echo "done."
echo ; echo

echo "Top 30 drivers over a 5-year span:"
python3 bests.py "${METRICS_TSV}" "Range5y" "Elo" /dev/stdout | cut -d'.' -f1
