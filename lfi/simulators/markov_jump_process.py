import numpy as np


class SimTooLongException(Exception):
    """
    Exception to be thrown when a simulation runs for too long.
    """

    def __init__(self, max_num_steps):
        self._max_num_steps = max_num_steps

    def __str__(self):
        return "Simulation exceeded the maximum of {} steps.".format(
            self._max_num_steps
        )


class MarkovJumpProcess:
    """
    Implements a generic Markov Jump Process.
    It's an abstract class and must be implemented by a subclass.
    """

    def __init__(self, initial_state, parameters):
        """
        :param init: initial state
        :param params: parameters
        """

        self._state = None
        self._parameters = None
        self._time = None
        self.reset(initial_state, parameters)

    def reset(self, initial_state, parameters):
        """
        Resets the simulators.
        :param init: initial state
        :param params: parameters
        """

        self._state = np.asarray(initial_state, dtype=float)
        self._parameters = np.asarray(parameters, dtype=float)
        self._time = 0.0

    def _compute_propensities(self):
        raise NotImplementedError(
            "This is an abstract method and should be implemented in a subclass."
        )

    def _do_reaction(self, reaction):
        raise NotImplementedError(
            "This is an abstract method and should be implemented in a subclass."
        )

    def simulate_for_num_steps(self, num_steps, include_initial_state=True):
        """
        Runs the simulators for a given number of steps.

        :param num_steps: number of steps
        :param include_init_state: if True, include the initial state in the output
        :param rng: random number generator to use
        :return: times, states
        """

        times = [self._time]
        states = [self._state.copy()]

        for _ in range(num_steps):

            rates = self._parameters * self._compute_propensities()
            total_rate = rates.sum()

            if total_rate == 0:
                self._time = float("inf")
                break

            self._time += np.random.exponential(scale=1.0 / total_rate)

            reaction = np.random.choice(range(4), p=rates / total_rate)
            self._do_reaction(reaction)

            times.append(self._time)
            states.append(self._state.copy())

        if not include_initial_state:
            times, states = times[1:], states[1:]

        return np.array(times), np.array(states)

    def simulate_for_time(
        self, dt, duration, include_init_state=True, max_n_steps=float("inf")
    ):
        """
        Runs the simulators for a given amount of time.

        :param dt: time step
        :param duration: total amount of time
        :param include_init_state: if True, include the initial state in the output
        :param max_n_steps: maximum number of simulators steps allowed. If exceeded, an exception is thrown.
        :param rng: random number generator to use
        :return: states
        """

        num_rec = int(duration / dt) + 1
        states = np.empty([num_rec, self._state.size], float)
        cur_time = self._time
        n_steps = 0

        for i in range(num_rec):

            while cur_time > self._time:

                rates = self._parameters * self._compute_propensities()
                total_rate = rates.sum()

                if total_rate == 0:
                    self._time = float("inf")
                    break

                try:
                    self._time += np.random.exponential(scale=1.0 / total_rate)
                except:
                    print(self._parameters, self._compute_propensities())

                reaction = np.random.choice(range(4), p=rates / total_rate)
                self._do_reaction(reaction)

                n_steps += 1
                if n_steps > max_n_steps:
                    raise SimTooLongException(max_n_steps)

            states[i] = self._state.copy()
            cur_time += dt

        return states if include_init_state else states[1:]
