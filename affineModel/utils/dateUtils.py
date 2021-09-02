from utils import Holiday
from datetime import timedelta, date, datetime
from typing import AnyStr

trading_holiday = {date(year=2020, month=1, day=1): "元旦",
                   date(year=2020, month=1, day=27): "春节",
                   date(year=2020, month=1, day=28): "春节",
                   date(year=2020, month=1, day=29): "春节",
                   date(year=2020, month=1, day=30): "春节",
                   date(year=2020, month=4, day=4): "清明节",
                   date(year=2020, month=5, day=1): "劳动节",
                   date(year=2020, month=6, day=25): "端午节",
                   date(year=2020, month=6, day=26): "端午节",
                   date(year=2020, month=10, day=1): "国庆节",
                   date(year=2020, month=10, day=2): "国庆节",
                   date(year=2020, month=10, day=5): "国庆节",
                   date(year=2020, month=10, day=6): "国庆节",
                   date(year=2020, month=10, day=7): "国庆节",
                   date(year=2020, month=10, day=8): "国庆节",
                   date(year=2021, month=1, day=1): "元旦",
                   date(year=2021, month=2, day=11): "春节",
                   date(year=2021, month=2, day=12): "春节",
                   date(year=2021, month=2, day=15): "春节",
                   date(year=2021, month=2, day=16): "春节",
                   date(year=2021, month=2, day=17): "春节",
                   date(year=2021, month=4, day=5): "清明节",
                   date(year=2021, month=5, day=3): "劳动节",
                   date(year=2021, month=5, day=4): "劳动节",
                   date(year=2021, month=5, day=5): "劳动节",
                   date(year=2021, month=6, day=14): "端午节",
                   date(year=2021, month=9, day=20): "中秋节",
                   date(year=2021, month=9, day=21): "中秋节",
                   date(year=2021, month=10, day=1): "国庆节",
                   date(year=2021, month=10, day=4): "国庆节",
                   date(year=2021, month=10, day=5): "国庆节",
                   date(year=2021, month=10, day=6): "国庆节",
                   date(year=2021, month=10, day=7): "国庆节",
                   date(year=2022, month=1, day=3): "元旦",
                   date(year=2022, month=1, day=31): "春节",
                   date(year=2022, month=2, day=1): "春节",
                   date(year=2022, month=2, day=2): "春节",
                   date(year=2022, month=2, day=3): "春节",
                   date(year=2022, month=2, day=4): "春节"}


def is_holiday(date: datetime, holiday: dict) -> bool:
    """
    judge if a given date is holiday or not
    """
    if date in holiday:
        return True
    else:
        return False


def is_weekend(date: datetime) -> bool:
    """
    judge if a given date is weekend or not
    """
    weekend = [5, 6]
    if date.weekday() in weekend:
        return True
    else:
        return False


def date_range(start_date: datetime, end_date: datetime):
    """
    generate a range of dates
    """
    end_date = end_date + timedelta(1)
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)


def schedule_generator(start: AnyStr, end: AnyStr, holiday: dict, include_start=False, include_end=True):
    """
    generate trading days between start date and end date, excluding holidays and workdays
    """
    start_date = datetime.strptime(start, "%Y-%m-%d").date()
    end_date = datetime.strptime(end, "%Y-%m-%d").date()
    schedule = []
    if not include_start:
        start_date = start + timedelta(1)
    if not include_end:
        end_date = end - timedelta(1)

    for single_date in date_range(start_date, end_date):
        if is_holiday(single_date, holiday) or is_weekend(single_date):
            continue
        else:
            schedule.append(single_date)

    return schedule


def business_day_count(start: AnyStr, end: AnyStr) -> int:
    """
    count businss days between 2 dates
    """
    start = datetime.strptime(start, "%Y-%m-%d").date()
    end = datetime.strptime(end, "%Y-%m-%d").date()
    count = 0
    for single_date in date_range(start, end):
        if is_holiday(single_date, trading_holiday) or is_weekend(single_date):
            continue
        else:
            count += 1
    return count


def calendar_day_count(start: AnyStr, end: AnyStr) -> int:
    """
    count calendar days between start and end dates, depending on whether including start or end date
    """
    start = datetime.strptime(start, "%Y-%m-%d").date()
    end = datetime.strptime(end, "%Y-%m-%d").date()
    return (end - start).days


def is_after(date1: AnyStr, date2: AnyStr) -> bool:
    """
    check if date1 is before date2
    """
    date1 = datetime.strptime(date1, "%Y-%m-%d").date()
    date2 = datetime.strptime(date2, "%Y-%m-%d").date()
    return date1 > date2


if __name__ == '__main__':
    print(business_day_count("2021-07-29", "2022-07-30"))
    print(calendar_day_count("2021-07-29", "2022-07-30"))
