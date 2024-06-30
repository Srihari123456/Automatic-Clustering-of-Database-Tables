from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
import numpy as np
import pickle

with open('../Module-1/test_token_list.pkl', 'rb') as input_file:
	token_list = pickle.load(input_file)

docs_sentences = []
for doc in token_list:
	doc_single_string = ' '.join(doc)
	docs_sentences.append(doc_single_string)

embeddings = model.encode(docs_sentences)

with open('test_sentence_transformer_embeddings.npy', 'wb') as f:
	np.save(f, embeddings)
