import functools

import pandas as pd


class Watch:
    def __init__(self, fctn=pd.Timestamp.now):
        self.fctn = fctn
        self.tick = self.fctn()
        self.records = []

    def see_timedelta(self):
        tick = self.fctn()
        res = tick - self.tick
        self.records.append(res)
        self.tick = tick
        return res

    def see_seconds(self):
        return round(self.see_timedelta().total_seconds(), 6)

    def total_timedelta(self):
        totalTime = sum(self.records, pd.Timedelta(seconds=0))
        return totalTime

    def total_seconds(self):
        return round(self.total_timedelta().total_seconds(), 6)


def watch_time(func):
    @functools.wraps(func)
    def wrap(*args, **kwargs):
        w = Watch()
        res = func(*args, **kwargs)

        print(f"Time Cost : {func.__name__} {w.see_seconds()}")
        return res

    return wrap


if __name__ == '__main__':
    pass
    w = Watch()
