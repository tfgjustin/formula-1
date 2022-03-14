class Driver(object):
    def __init__(self, driver_id, driver_name, elo_rating, birthday=None, birth_year=None):
        self._id = driver_id
        self._name = driver_name
        self._rating = elo_rating
        self._birthday = birthday
        self._birth_year = birth_year

    def id(self):
        return self._id

    def name(self):
        return self._name

    def birthday(self):
        return self._birthday

    def birth_year(self):
        return self._birth_year

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
