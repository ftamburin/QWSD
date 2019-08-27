# -*- coding: utf-8 -*-

import sys
import numpy as np
import cmath
import lxml.etree as et
from nltk.corpus import wordnet as wn
from scipy.linalg import orth

pos_dic = { 'ADJ': u'a', 'ADV': u'r', 'NOUN': u'n', 'VERB': u'v', }
POS_LIST = pos_dic.values()  # ['a', 'r', 'n', 'v']


def load_all_words_data(data_path):
	# Partially taken from
	# https://github.com/luofuli/word-sense-disambiguation/blob/master/utils/data.py
	print('LOADING:',data_path,file=sys.stderr)

	context = et.iterparse(data_path, tag='sentence')

	data = []
	poss = set()
	for event, elem in context:
		sent_list = []
		pos_list = []
		for child in elem:
			word = child.get('lemma').lower()
			sent_list.append(word)
			pos = child.get('pos')
			pos_list.append(pos)
			poss.add(pos)

		i = -1
		for child in elem:
			if child.tag == 'wf':
				i += 1
			elif child.tag == 'instance':
				i += 1
				id = child.get('id')
				lemma = child.get('lemma').lower()
				if '(' in lemma:
					print(id)
				pos = child.get('pos')
				word = lemma + '#' + pos_dic[pos]

				context = sent_list[:]
				if context[i] != lemma:
					print('/'.join(context))
					print(i)
					print(lemma)
				context[i] = '<target>'

				x = {
					'id': id,
					'context': context,
					'target_word': word,
					'poss': pos_list,
				}
				
				data.append(x)

	return data


def load_embeddings(fname):
	print('LOADING BINARY EMBEDDING from',fname,file=sys.stderr)
	word_vecs = {}
	with open(fname, "rb") as f:
		header = f.readline()
		vocab_size, layer1_size = map(int, header.split())
		print(vocab_size, int(layer1_size/2), file=sys.stderr)
		binary_len = np.dtype('float32').itemsize * layer1_size
		for line in range(vocab_size):
			if (line%10000 == 0):
				print('.',end='',file=sys.stderr)
				sys.stderr.flush()
			word = []
			while True:
				ch = f.read(1)
				if ch == b' ':
					word = b''.join(word)
					break
				if ch != b'\n':
					word.append(ch)  
			word = word.decode("utf-8")
			temp = np.fromstring(f.read(binary_len), dtype='float32')  
			word_vecs[word] = temp.view(np.complex64)
			# NORMALIZE EMBEDDING
			word_vecs[word] = (word_vecs[word]/np.linalg.norm(word_vecs[word])).astype(np.complex128)
		print('',file=sys.stderr)
	return word_vecs, int(layer1_size/2)
		

def	GetWNSenses(word, p, min_sense_freq):
	n_hyper = 0
	n_hypo  = 0
	posmap = {'n': wn.NOUN, 'v': wn.VERB, 'a':wn.ADJ, 'r': wn.ADV}
	s=wn.synsets(word, pos=posmap[p])
	mword = wn.morphy(word, posmap[p])

	senses = {}
	for lemma in wn.lemmas(word, posmap[p]):
		if (lemma.count() > min_sense_freq):
			senses[lemma.key()] = lemma.count()
	return senses


def BuildSubspacePrj(s,cembs):
	# FIND ALL VECTORS CONNECTED WITH A SPECIFIC SENSE
	vects = []
	for k, cemb in cembs.items():
		if (k.find('#')):
			sk = k[0:k.find('#')]
			if (sk == s):
				vects.append(cemb)
	if (len(vects) == 0):
		print('ERROR 3: id not found',s)
		sys.exit(1)

	# COMPUTE AN ORTHONORMAL BASIS OF THE SPANNED SPACE
	esize = vects[0].size
	usedV = min(len(vects),esize-1);
	m = np.zeros((esize,usedV), dtype=np.complex64)
	for j in range(usedV):
		m[:,j] = vects[j]
	basis = orth(m)

	# COMPUTE THE PROJECTOR ONTO THE SENSE SPACE
	mrank = basis.shape[1]
	prj = np.zeros((esize,esize), dtype=np.complex64)
	basis = np.matrix(basis)
	for j in range(mrank):
		prj = prj + basis[:,j] * basis[:,j].getH()

	return prj, usedV




if __name__ == "__main__":
	cembs, edim = load_embeddings(sys.argv[1])
	insts = load_all_words_data(sys.argv[2])

	print('\n-------------------------------------------------------------',file=sys.stderr)
	for i in range(len(insts)):
		print('TEST_INSTANCE_ID:',insts[i]['id'],file=sys.stderr)
		print('TEST_INSTANCE_TARGET_WORD:',insts[i]['target_word'],file=sys.stderr)

		# RETRIEVE ALL POSSIBLE SENSES FOT THE TARGET
		tw, pos = insts[i]['target_word'].split('#')
		msf = -1
		senses = GetWNSenses(tw,pos,msf)
		while ((len(senses) == 0) and (msf > -2)):
			msf -= 1
			senses = GetWNSenses(tw,pos,msf)
		if (len(senses) == 0):
			print('ERROR 0: No senses into wordnet for instance',insts[i]['id'],file=sys.stderr)
			sys.exit(1)
		print('Senses:',senses, file=sys.stderr)

		if (len(senses) == 1):
			print('Only one sense available.', file=sys.stderr)
			print(insts[i]['id'], list(senses.keys())[0])
		else:
			# GET COMPLEX EMBEDDING VECTOR FOR THE TARGET
			if (tw in cembs):
				wV = np.matrix(cembs.get(tw)).T
			else:
				print('Warning: Word vector not found. Defaulting...', file=sys.stderr)
				wV = np.ones(edim, dtype=np.complex128)
				wV = np.matrix(wV / np.linalg.norm(wV)).T

			# GET THE PROJECTOR FOR THE CONTEXT
			#cntxV = np.matrix(cembs.get(insts[i]['id'])).T
			prjC, usedV = BuildSubspacePrj(insts[i]['id'],cembs)

			# PROJECTION OVER THE CONTEXT SUBSPACE
			#prjC = cntxV * cntxV.getH()
			wC = prjC * wV / np.sqrt(wV.getH() * prjC.getH() * prjC * wV)

			# COMPUTE SENSE WITH THE BEST PROBABILITY
			maxprob = -1.0
			for s in senses:
				prjS, usedV = BuildSubspacePrj(s,cembs)
				prob = np.asscalar(np.real(wC.getH() * prjS.getH() * prjS * wC))
				print(s,'->',prob, file=sys.stderr)
				if (prob > maxprob):
					maxprob = prob
					imaxprob = s
			if (maxprob == -1.0):
				# BACKOFF ON MFS
				maxc = -1
				for s, c in senses:
					if (c > maxc):
						maxc = c
						imaxprob = s
				print('CASE2: Recovering on MFS',imaxprob, file=sys.stderr)
			print(insts[i]['id'], imaxprob, file=sys.stderr)
			print(insts[i]['id'], imaxprob)

		print('-------------------------------------------------------------',file=sys.stderr)
		sys.stdout.flush()
		sys.stderr.flush()

