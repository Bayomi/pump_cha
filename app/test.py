import itertools

def getPermutations(gen): 
	permutes = []
	for array in itertools.product(*gen):
		permutes.append(list(array))

	return permutes

papel = ['papel 1', 'papel 2']
apoio = ['apoio 1', 'apoio 2']
mesa = ['mesa 1', 'mesa 2']

arrGer = []
arrGer.append(papel)
arrGer.append(apoio)


print getPermutations(arrGer)