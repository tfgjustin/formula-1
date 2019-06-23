## Usage
1. Make sure you have the following dependencies installed: csv, scipy, sqlite3
2. Pull the latest data from the emkael/elof1 repo and build the elo.db
3. Run `python3 src/sqlite_to_tsv.py <elo.db> <drivers.tsv> <races.tsv> <results.tsv>`
4. Run `python3 src/elo_ranking.py <results.tsv> <rankings.tsv>`
  where `<results.tsv>` is the file generated in the previous step.
5. Run `python3 src/flatten_ratings.py <rankings.tsv> <final.tsv>` where `<rankings.tsv>` is
  the file generated in the previous step.
