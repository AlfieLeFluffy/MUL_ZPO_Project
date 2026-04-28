import io
import cProfile
import pstats
from pstats import SortKey

def start_profiling():
    profile = cProfile.Profile()
    profile.enable()
    return profile

def end_profiling(profile: cProfile.Profile, filename: str):
    profile.disable()
    s = io.StringIO()
    sortby = SortKey.CUMULATIVE
    ps = pstats.Stats(profile, stream=s).sort_stats(sortby)
    ps.print_stats()
    with open(filename, "w") as f:
        f.write(s.getvalue())