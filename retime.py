def time_2(t,type):
    if type == '18':
        if t > 0.2:
            t /= 6.5
        elif t > 0.15:
            t /= 4
        elif t > 0.1:
            t /= 2.6
        elif t > 0.06:
            t /= 1.8
    elif type == '50':
        print(t)

        if t > 0.25:
            t /= 4.5
        elif t > 0.2:
            t /= 3
        elif t > 0.12:
            t /= 2
        elif t > 0.09:
            t -= 0.02
        # elif t > 1
    else:
        if t > 0.13:
            t /= 3
        elif t > 0.08:
            t /= 2
        else:
            t /= 1.5
    return round(t,3)