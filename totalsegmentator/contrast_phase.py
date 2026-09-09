def pi_time_to_phase(pi_time: float) -> tuple[str, float]:
    """
    Convert the pi time to a phase and get a probability for the value.

    native: 0-10
    arterial_early: 10-30
    arterial_late:  30-60
    portal_venous:  60-100
    delayed: 100+

    returns: phase, probability
    """
    if pi_time < 5:
        return "native", 1.0
    elif pi_time < 10:
        return "native", 0.7
    elif pi_time < 20:
        return "arterial_early", 0.7
    elif pi_time < 30:
        return "arterial_early", 1.0
    elif pi_time < 50:
        return "arterial_late", 1.0
    elif pi_time < 60:
        return "arterial_late", 0.7  # in previous version: "portal_venous"
    elif pi_time < 70:
        return "portal_venous", 1.0
    elif pi_time < 90:
        return "portal_venous", 1.0
    elif pi_time < 100:
        return "portal_venous", 0.7
    else:
        return "portal_venous", 0.3
        # return "delayed", 0.7  # not enough good training data for this
