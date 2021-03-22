class Result(object):
    def __init__(self, event, driver, end_position, team=None, start_position=0, num_racers=0, dnf_category='', laps=0):
        self._event = event
        self._driver = driver
        self._team = team
        self._start_position = int(start_position)
        self._end_position = int(end_position)
        self._num_racers = int(num_racers)
        self._laps = int(laps)
        self._dnf_category = dnf_category

    def event(self):
        return self._event

    def driver(self):
        return self._driver

    def team(self):
        return self._team

    def start_position(self):
        return self._start_position

    def end_position(self):
        return self._end_position

    def num_racers(self):
        return self._num_racers

    def position_from_back(self):
        if self._num_racers and self._start_position:
            return self._num_racers - self._start_position
        return 0

    def dnf_category(self):
        return self._dnf_category

    def laps(self):
        return self._laps
