def decode(self, sequence):

    sequence_length = len(sequence)
    if sequence_length == 0:
        return []

    delta = {}

    for state in self._states:
        delta[state] = self.start_prob(state) * self.emit_prob(state, sequence[0])

    pre = []


    for index in range(1, sequence_length):

        delta_bar = {}

        pre_state = {}
        for state_to in self._states:
            max_prob = 0
            max_state = None  # backtrace 변수
            for state_from in self._states:

                prob = delta[state_from] * self.trans_prob(state_from, state_to)

                if prob > max_prob:

                    max_prob = prob

                    max_state = state_from
            delta_bar[state_to] = max_prob * self.emit_prob(state_to, sequence[index])
            pre_state[state_to] = max_state

        delta = delta_bar

        pre.append(pre_state)


    max_state = None
    max_prob = 0
    for state in self._states:
        if delta[state] > max_prob:
            max_prob = delta[state]
            max_state = state

    if max_state is None:
        return []

    result = [max_state]

    for index in range(sequence_length - 1, 0, -1):

        max_state = pre[index - 1][max_state]

        result.insert(0, max_state)

    return result