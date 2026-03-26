
import os
import sys
from time import time
from typing import Tuple

def get_duration(start_time: float, end_time: float) -> Tuple[float, float, float]:
    """
    Compute the elapsed time between two timestamps.

    The timestamps are expected to be values returned by time.time(),
    representing the number of seconds elapsed since January 1, 1970,
    00:00:00 UTC (the Unix epoch).

    The function converts the elapsed time into hours, minutes, and seconds.

    :param start_time: Start timestamp in seconds since the Unix epoch.
    :param end_time: End timestamp in seconds since the Unix epoch.
    :return: A tuple (hours, minutes, seconds) representing the elapsed time.
             - hours (float): Number of hours in the elapsed duration
             - minutes (float): Remaining minutes after extracting hours
             - seconds (float): Remaining seconds after extracting minutes
    """
    elapsed = end_time - start_time
    
    hours, remainder = divmod(elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)

    return hours, minutes, seconds

    