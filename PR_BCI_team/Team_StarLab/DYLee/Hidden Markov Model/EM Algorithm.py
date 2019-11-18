import hmm
sequences = [
    (state_list1, symbol_list1),
    (state_list2, symbol_list2),
    ...
    (state_listN, symbol_listN)]
model = hmm.train(sequences)