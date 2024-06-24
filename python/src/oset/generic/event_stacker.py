import numpy as np

def event_stacker(signal, event_indexes, event_bounds, method="unnormalized"):
    """
    Synchronously stacks events from input signal vectors based on specified indices and boundaries.

    Args:
        signal (ndarray): The input signal in vector form.
        event_indexes (array-like): An array of event indices.
        event_bounds :
            - If integer scalar: event_width, the time width of the stacked events
              (must be odd valued)
            - If a 2-element list or tuple: [left_wing_len, right_wing_len], a two
              element vector containing the number of samples from the left and
              right sides of each event (used for asymmetric events)
        method (str), optional: Stacking method (default: 'unnormalized').

    Returns:
        stacked_events (ndarray):
            A matrix of the form N x event_bounds, where N = length(event_indexes).
        num_non_zeros : numpy.ndarray
            Number of stacked samples per event (equal to event_bounds except for the boundary events).
            Used to find the number of zero-padded samples per event.

    Note:
        - If event_bounds is provided as an even integer, the function adjusts it
          to the next odd value and issues a notification.
        - The function handles boundary cases by applying zero-padding where necessary to maintain consistent event widths.

    Revision History:
        2024: Translated to Python from Matlab.

    Jasper Hsu, 2024
        The Open-Source Electrophysiological Toolbox
        https://github.com/alphanumericslab/OSET
    """

    if not isinstance(signal, np.ndarray):
        raise ValueError("First input should be a numpy array")

    signal_len = len(signal)
    num_events = len(event_indexes)
    num_non_zeros = np.zeros(num_events, dtype=int)

    if isinstance(event_bounds, int):
        # Ensure event_bounds is odd
        if event_bounds % 2 == 0:
            event_bounds += 1
            print(
                "event_bounds must be odd valued (in scalar mode); automatically modified to the closest greater odd value."
            )

        half_len = event_bounds // 2
        center_index = half_len
        stacked_events = np.zeros((num_events, event_bounds))

        for i in range(num_events):
            start = max(event_indexes[i] - half_len, 0)
            stop = min(event_indexes[i] + half_len, signal_len)

            left_wing_len = event_indexes[i] - start
            right_wing_len = stop - event_indexes[i]

            stacked_events[i, center_index - left_wing_len : center_index + right_wing_len + 1] = signal[start:stop]
            num_non_zeros[i] = stop - start

    elif isinstance(event_bounds, list) and len(event_bounds) == 2:
        left_wing_len0, right_wing_len0 = event_bounds
        stacked_events = np.zeros((num_events, left_wing_len0 + right_wing_len0 + 1))
        intermediate_index = left_wing_len0

        for i in range(num_events):
            start = max(event_indexes[i] - left_wing_len0, 0)
            stop = min(event_indexes[i] + right_wing_len0, signal_len)

            left_wing_len = event_indexes[i] - start
            right_wing_len = stop - event_indexes[i]

            stacked_events[i, intermediate_index - left_wing_len: intermediate_index + right_wing_len + 1] = signal[start:stop]
            num_non_zeros[i] = stop - start

    else:
        raise ValueError(
            "event_bounds can be either a scalar or a two-element vector; see function help for details."
        )

    return stacked_events, num_non_zeros
