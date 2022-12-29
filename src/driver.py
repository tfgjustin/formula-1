from collections import defaultdict


class Driver(object):
    _MIN_RACES_FOR_SEASON = 3

    def __init__(self, driver_id, driver_name, elo_rating, birthday=None, birth_year=None):
        self._id = driver_id
        self._name = driver_name
        self._rating = elo_rating
        self._birthday = birthday
        self._birth_year = int(birth_year)
        self._race_counts_by_year = defaultdict(int)

    def id(self):
        return self._id

    def name(self):
        return self._name

    def birthday(self):
        return self._birthday

    def birth_year(self):
        return self._birth_year

    def seasons(self):
        return sorted(
            [year for year, count in self._race_counts_by_year.items() if count >= Driver._MIN_RACES_FOR_SEASON],
            reverse=True
        )

    def start_update(self, event_id, base_driver_reliability):
        if self._rating is None:
            return
        self._rating.start_update(event_id, self._id, base_reliability=base_driver_reliability)

    def maybe_regress(self):
        self._rating.update(0)

    def commit_update(self):
        self._rating.commit_update()

    def rating(self):
        return self._rating

    def set_rating(self, rating):
        self._rating = rating

    def add_year_participation(self, event_id):
        if not event_id.endswith('R'):
            return
        year = int(event_id[0:4])
        self._race_counts_by_year[year] += 1
