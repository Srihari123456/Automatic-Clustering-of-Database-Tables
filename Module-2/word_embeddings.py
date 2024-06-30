import pickle
from gensim.downloader import load
import numpy as np
from bpemb import BPEmb

print('loading BPEmb')
bpemb_en = BPEmb(lang="en", vs=200000, dim=100)

with open('../Module-1/train_token_list.pkl', 'rb') as input_file:
	token_list = pickle.load(input_file)

print('loading pretrained embeddings...') # see available pretrained models here https://radimrehurek.com/gensim_3.8.3/auto_examples/howtos/run_downloader_api.html
word_vectors = load('glove-wiki-gigaword-100')
embedding_dim = word_vectors['the'].shape[0]

doc_vectors = np.zeros((len(token_list), embedding_dim))

tokens_not_found = 0
for doc_idx, doc in enumerate(token_list):
	print('doc {}/{}'.format(doc_idx+1, len(token_list)))
	doc_vector = np.zeros(embedding_dim)
	for word in doc:
		try:
			doc_vector = doc_vector + word_vectors[word]
		except KeyError:
			print('key \'{}\' not found'.format(word))
			tokens_not_found = tokens_not_found + 1

			# compute bpemb embedding
			bpemb_embedding = bpemb_en.embed(word)
			bpemb_segmentation = bpemb_en.encode(word)
			print("word = {}, bpemb_segmentation = {}".format(word, bpemb_segmentation))
			doc_vector = doc_vector + bpemb_embedding.sum(axis=0).T

			pass

	doc_vector = doc_vector/embedding_dim
	doc_vectors[doc_idx, :] = doc_vector

print('OOV tokens = {}'.format(tokens_not_found))
with open('train_doc_vectors_glove_wiki_gigaword_100_bpemb_vs_200000.npy', 'wb') as f:
	np.save(f, doc_vectors)


print('Done.')


