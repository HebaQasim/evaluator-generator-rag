"""Pricing rules of Orbit Bikes (a fictional company, made for testing a question-answering system)."""

HOURLY_PRICE = {"city": 3, "ebike": 6}   # JOD per hour
LATE_FEE = 2                             # JOD for every 15 minutes after the planned return time
WEEKEND_SURCHARGE = 0.10                 # 10% more on Friday and Saturday
PLUS_EBIKE_DISCOUNT = 0.10               # members of the Plus plan pay 10% less for e-bikes


def calculate_price(hours, bike_type, plan=None, weekend=False, late_minutes=0):
    """Return the price of a rental in JOD.

    The weekend surcharge is added first, then the Plus discount is applied to e-bikes.
    The late fee is added at the end and is never discounted.
    """
    price = hours * HOURLY_PRICE[bike_type]
    if weekend:
        price = price * (1 + WEEKEND_SURCHARGE)
    if plan == "plus" and bike_type == "ebike":
        price = price * (1 - PLUS_EBIKE_DISCOUNT)
    late_blocks = -(-late_minutes // 15)   # every started block of 15 minutes counts
    return round(price + late_blocks * LATE_FEE, 2)
