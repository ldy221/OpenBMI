def _forward(self, sequence):
    sequence_length = len(sequence)
    if sequence_length == 0:
        return []

    alpha = [{}]

    for state in self._states:
        alpha[0][state] = self.start_prob(state) * self.emit_prob(state, sequence[0])

    for index in range(1, sequence_length):
        alpha.append({})
        for state_to in self._states:
            prob = 0
            for state_from in self._states:

                prob += alpha[index - 1][state_from] * \
                        self.trans_prob(state_from, state_to)

            alpha[index][state_to] = prob * self.emit_prob(state_to, sequence[index])

    return alpha