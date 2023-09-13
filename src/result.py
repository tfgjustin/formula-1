class Result(object):
    def __init__(self, event, entrant, opening_position, partial_position, end_position, dnf_category='',
                 laps_completed=0):
        self._event = event
        self._entrant = entrant
        self._end_position = int(end_position)
        if opening_position is None or opening_position == '-':
            self._opening_position = None
        else:
            self._opening_position = int(opening_position)
        if partial_position is None or partial_position == '-':
            self._partial_position = None
        else:
            self._partial_position = int(partial_position)
        self._num_racers = event.num_entrants()
        self._laps_completed = int(laps_completed)
        self._dnf_category = dnf_category
        self._probability_fail_at_n = None
        self._probability_fail_after_n = None
        self._probability_succeed_through_n = None

    def event(self):
        return self._event

    def entrant(self):
        return self._entrant

    def driver(self):
        return self._entrant.driver()

    def team(self):
        return self._entrant.team()

    def start_position(self):
        return self._entrant.start_position()

    def opening_position(self):
        return self._opening_position

    def partial_position(self):
        return self._partial_position

    def end_position(self):
        return self._end_position

    def num_racers(self):
        return self._num_racers

    def position_from_back(self):
        if self._num_racers and self._entrant.start_position():
            return self._num_racers - self._entrant.start_position()
        return 0

    def dnf_category(self):
        return self._dnf_category

    def laps(self):
        return self._laps_completed
