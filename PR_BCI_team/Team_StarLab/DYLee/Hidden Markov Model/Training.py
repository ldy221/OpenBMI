def learn(self, sequence, smoothing=0):
    length = len(sequence)
    alpha = self._forward(sequence)
    beta = self._backward(sequence)

    gamma = []
    for index in range(length):
        prob_sum = 0
        gamma.append({})
        for state in self._states:
            prob = alpha[index][state] * beta[index][state]
            gamma[index][state] = prob
            prob_sum += prob

        if prob_sum == 0:
            continue

        for state in self._states:
            gamma[index][state] /= prob_sum

    xi = []
    for index in range(length - 1):
        prob_sum = 0
        xi.append({})
        for state_from in self._states:
            xi[index][state_from] = {}
            for state_to in self._states:
                prob = alpha[index][state_from] * beta[index + 1][state_to] * \
                       self.trans_prob(state_from, state_to) * \
                       self.emit_prob(state_to, sequence[index + 1])
                xi[index][state_from][state_to] = prob
                prob_sum += prob

        if prob_sum == 0:
            continue

        for state_from in self._states:
            for state_to in self._states:
                xi[index][state_from][state_to] /= prob_sum

    states_number = len(self._states)
    symbols_number = len(self._symbols)
    for state in self._states:
        # update start probability
        self._start_prob[state] = \
            (smoothing + gamma[0][state]) / (1 + states_number * smoothing)

        # update transition probability
        gamma_sum = 0
        for index in range(length - 1):
            gamma_sum += gamma[index][state]

        if gamma_sum > 0:
            denominator = gamma_sum + states_number * smoothing
            for state_to in self._states:
                xi_sum = 0
                for index in range(length - 1):
                    xi_sum += xi[index][state][state_to]
                self._trans_prob[state][state_to] = (smoothing + xi_sum) / denominator
        else:
            for state_to in self._states:
                self._trans_prob[state][state_to] = 0

        # update emission probability
        gamma_sum += gamma[length - 1][state]
        emit_gamma_sum = {}
        for symbol in self._symbols:
            emit_gamma_sum[symbol] = 0

        for index in range(length):
            emit_gamma_sum[sequence[index]] += gamma[index][state]

        if gamma_sum > 0:
            denominator = gamma_sum + symbols_number * smoothing
            for symbol in self._symbols:
                self._emit_prob[state][symbol] = \
                    (smoothing + emit_gamma_sum[symbol]) / denominator
        else:
            for symbol in self._symbols:
                self._emit_prob[state][symbol] = 0        