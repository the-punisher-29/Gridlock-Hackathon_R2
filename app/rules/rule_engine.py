from collections import defaultdict


def line_side(p, a, b):
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


class RuleEngine:
    def __init__(self, zone_config):
        self.zone = zone_config.data
        self.track_history = defaultdict(list)

    def update_track(self, tracker_id, centroid):
        self.track_history[tracker_id].append(centroid)
        if len(self.track_history[tracker_id]) > 10:
            self.track_history[tracker_id].pop(0)

    def check_stop_line_violation(self, tracker_id):
        if self.zone["stop_line"] is None or self.zone["traffic_light_state"] != "red":
            return False
        history = self.track_history[tracker_id]
        if len(history) < 2:
            return False
        a, b = self.zone["stop_line"]
        prev_side = line_side(history[-2], a, b)
        curr_side = line_side(history[-1], a, b)
        return (prev_side > 0 and curr_side < 0) or (prev_side < 0 and curr_side > 0)

    def check_red_light_violation(self, tracker_id):
        return self.check_stop_line_violation(tracker_id)