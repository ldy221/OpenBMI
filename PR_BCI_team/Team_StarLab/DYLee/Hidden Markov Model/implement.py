import hmm
states = ('hot', 'cold')
symbols = ('1', '2', '3')

start_prob = {
    'hot' : 0.8,
    'cold' : 0.2
}

trans_prob = {
    'hot': { 'hot' : 0.6, 'cold' : 0.4 },
    'cold': { 'hot' : 0.4, 'cold' : 0.6 }
}

emit_prob = {
    'hot': { '1' : 0.2, '2' : 0.4, '3' : 0.4 },
    'cold': { '1' : 0.5, '2' : 0.4, '3' : 0.1 }
}

model = hmm.Model(states, symbols, start_prob, trans_prob, emit_prob)
sequence = ['3', '1', '3']
print(model.evaluate(sequence))
print(model.decode(sequence)) 