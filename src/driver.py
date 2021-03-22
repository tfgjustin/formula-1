from ratings import EloRating


class Driver(object):
    def __init__(self, driver_id, driver_name, elo_rating):
        self._id = driver_id
        self._name = driver_name
        self._rating = elo_rating

    def id(self):
        return self._id

    def name(self):
        return self._name

    def start_update(self, event_id):
        if self._rating is None:
            return
        self._rating.start_update(event_id, self._id)

    def commit_update(self):
        self._rating.commit_update()

    def rating(self):
        return self._rating

