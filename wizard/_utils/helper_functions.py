def find_nex_greater_wave(waves, wave_1: int, maximum_deviation: int = 5):

    wave_next = -1

    for n in range(maximum_deviation):
        wave_n = wave_1 + n

        if wave_n in waves:
            wave_next = wave_n
            break

    return wave_next


def find_nex_smaller_wave(waves, wave_1: int, maximum_deviation: int = 5):
    wave_next = -1

    for n in range(maximum_deviation):
        wave_n = wave_1 - n

        if wave_n in waves:
            wave_next = wave_n
            break

    return wave_next