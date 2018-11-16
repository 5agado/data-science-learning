import numpy as np
import scipy.signal as signal

# adapted from http://stackoverflow.com/questions/4387878/simulator-of-realistic-ecg-signal-from-rr-data-for-matlab-or-python/33737898#33737898


def get_single_heartbeat(nb_rest_samples: int):
    """
    :param nb_rest_samples: number of zeroed samples to add after the "pqrst" signal
    :return:
    """

    # The "Daubechies" wavelet is a rough approximation to a real, single, heart beat ("pqrst") signal
    pqrst = signal.wavelets.daub(10)

    # Add the gap after the pqrst when the heart is resting.
    zero_array = np.zeros(nb_rest_samples, dtype=float)
    pqrst_full = np.concatenate([pqrst, zero_array])

    return pqrst_full


def get_simulated_heartbeat(bpm: int, capture_length: int, rest_factor: int = 1000):
    """
    :param bpm: beats per minute rate
    :param capture_length: simulated period of time in seconds that the ecg is captured in
    :param rest_factor
    :return:
    """

    # Calculate the number of beats in capture time period
    nb_heart_beats = int(capture_length * (bpm/60))

    hb = get_single_heartbeat(int(rest_factor/bpm))

    # Concatenate together the number of heart beats needed
    ecg_template = np.tile(hb, nb_heart_beats)

    # Add gaussian noise
    noise = np.random.normal(0, 0.01, len(ecg_template))
    ecg_template_noisy = noise + ecg_template

    return ecg_template_noisy


def simulate_heartbeat(bpm_series: list, capture_length: int = 10):
    simulations = [get_simulated_heartbeat(bpm, capture_length) for bpm in bpm_series]

    return np.concatenate(simulations)
