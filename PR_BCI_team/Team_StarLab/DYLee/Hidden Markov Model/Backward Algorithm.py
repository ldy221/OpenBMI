def _backward(self, sequence):
    sequence_length = len(sequence)
    if sequence_length == 0:
        return []

    beta = [{}]
    for state in self._states:
        beta[0][state] = 1

    for index in range(sequence_length - 1, 0, -1):
        beta.insert(0, {})
        for state_from in self._states:
            prob = 0
            for state_to in self._states:
                prob += beta[1][state_to] * \
                        self.trans_prob(state_from, state_to) * \
                        self.emit_prob(state_to, sequence[index])
            beta[0][state_from] = prob

    return beta