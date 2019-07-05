## Usage
1. Make sure you have the following dependencies installed: csv, scipy, sklearn, sqlite3
2. Pull the latest data from the emkael/elof1 repo and build the elo.db
3. Run `python3 src/sqlite_to_tsv.py <elo.db> <drivers.tsv> <races.tsv> <results.tsv>`
4. Run `python3 src/elo_ranking.py <results.tsv> <ratings.tsv>`
  where `<results.tsv>` is the file generated in the previous step.
5. Run `python3 src/flatten_ratings.py <ratings.tsv> <flat_ratings.tsv>` where `<ratings.tsv>` is
  the file generated in the previous step.
6. Run `calculate_aggregates.py <drivers.tsv> <races.tsv> <flat_ratings.tsv> <metrics.tsv>`

From there you can use the `bests.py` script to print out various metrics of
interest. E.g.,
```
python3 bests.py <metrics.tsv> "Range5y" "Elo" /dev/stdout | cut -d'.' -f1
```
