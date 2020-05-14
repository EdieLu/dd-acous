# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import torch
import collections
import codecs
import numpy as np
import random
from bpemb import BPEmb
from os.path import join

from utils.config import PAD, UNK, BOS, EOS, SPC


class IterDataset(torch.utils.data.Dataset):

	"""
		load features from 

		'src_word_ids':train_src_word_ids[i_start:i_end],
		'src_sentence_lengths': train_src_sentence_lengths[i_start:i_end],
		'acous_flis':train_acous_flis[i_start:i_end],
		'acous_spkids':train_acous_spkids[i_start:i_end],
		'acous_lengths':train_acous_lengths[i_start:i_end]
	"""
	
	def __init__(self, batches, acous_norm, add_acous=True, add_timestamp=False):
		
		super(Dataset).__init__()

		self.batches = batches
		self.acous_norm = acous_norm
		self.add_acous = add_acous
		self.add_timestamp = add_timestamp
		
	def __len__(self):

		return len(self.batches)

	def __getitem__(self, index):

		srcid = self.batches[index]['src_word_ids'] # lis
		srcid = torch.nn.utils.rnn.pad_sequence([torch.LongTensor(elem) for elem in srcid], batch_first=True) # tensor
		tag = self.batches[index]['tags']
		tag = torch.nn.utils.rnn.pad_sequence([torch.LongTensor(elem) for elem in tag], batch_first=True) # tensor
		srclen = self.batches[index]['src_sentence_lengths'] # lis

		if self.add_acous:
			acous_feat = self.load_file(index) # tensor
			acouslen = self.batches[index]['acous_lengths'] # lis
			if self.add_timestamp:
				timestamp = self.batches[index]['acous_times'] # lis b x [[st0,ed0],[st1,ed1], ... ] 	(lx2)
			else:
				b = srcid.size(0)
				timestamp = torch.zeros(b,1,2)

			# debug - set to dummy input
			# acous_feat = torch.zeros(acous_feat.size())

		else:
			# dummy output
			b = srcid.size(0)
			seqlen = srcid.size(1)
			acous_feat = torch.zeros(b,1,40)
			acouslen = 0
			timestamp = torch.zeros(b,1,2)

		return srcid, srclen, acous_feat, acouslen, tag, timestamp

	def load_file(self, index):

		# import pdb; pdb.set_trace()
		if self.acous_norm:
			norm_param = self.load_mu_std(index)
		else:
			norm_param = None
		acous_feat = self.load_acous_from_flis(index, norm_param=norm_param)

		return acous_feat

	def load_mu_std(self, index):

		spkids = self.batches[index]['acous_spkids']
		norm_param = []
		mydict = {}

		# base = '/home/alta/BLTSpeaking/exp-ytl28/encdec/lib-bpe/swbd-asr/prep/fbk_mu_std' #for swbd
		base = '/home/alta/BLTSpeaking/exp-ytl28/encdec/lib/add-acoustics/eval3/fbk_mu_std' #for eval3 

		for idx in range(len(spkids)):
			spkid = spkids[idx]
			if spkid in mydict:
				pass
			else:
				f_mu = join(base, spkid+'.mu.npy')
				f_std = join(base, spkid+'.std.npy')
				mu = np.load(f_mu)
				std = np.load(f_std)
				mydict[spkid] = [mu, std]
				
			norm_param.append(mydict[spkid])

		return norm_param

	def load_acous_from_flis(self, index, norm_param=None):

		flis = self.batches[index]['acous_flis']
		max_len = 0
		feat_lis = []
		for idx in range(len(flis)):
			f = flis[idx]
			featarr = np.load(f)		
			if type(norm_param) != type(None):
				mu, std = norm_param[idx] # dim=40
				featarr = 1. * (featarr - mu) / std
			feat = torch.FloatTensor(featarr) # np array (len x 40)
			max_len = max(max_len, feat.size(0))
			feat_lis.append(feat)

		# import pdb; pdb.set_trace()
		divisible_eight = max_len + 8 - max_len % 8
		dummy = torch.ones(divisible_eight , 40)
		feat_lis.append(dummy)
		feat_lis = torch.nn.utils.rnn.pad_sequence(feat_lis, batch_first=True)[:-1]

		return feat_lis


class Dataset(object):

	""" load src-tgt from file """

	def __init__(self,
		# add params 
		path_src,
		path_tgt,
		path_vocab_src,
		path_vocab_tgt,
		use_type=True,
		seqrev=False,
		# 
		acous_path=None,
		add_acous=True,
		acous_norm=False,
		# 
		tsv_path=None,
		tag_rev=False,
		keep_filler=True,
		# 
		timestamp_path=None,
		add_timestamp=False,
		# 
		max_seq_len=32,
		batch_size=64,
		use_gpu=True
		):

		super(Dataset, self).__init__()

		self.path_src = path_src
		self.path_tgt = path_tgt

		self.path_vocab_src = path_vocab_src
		self.path_vocab_tgt = path_vocab_tgt
		self.use_type = use_type
		self.seqrev = seqrev

		self.max_seq_len = max_seq_len
		self.batch_size = batch_size
		self.use_gpu = use_gpu

		self.tsv_path = tsv_path
		self.tag_rev = tag_rev
		self.keep_filler = keep_filler

		self.acous_path = acous_path
		self.add_acous = add_acous
		self.acous_norm = acous_norm

		self.timestamp_path = timestamp_path
		self.add_timestamp = add_timestamp

		self.acous_max_len = 6200 # eval3 - 6200; swbd 4200 # should include all acous 
		print('acous path: {}'.format(self.acous_path))
		print('max acous length: {}'.format(self.acous_max_len))

		self.load_vocab()
		self.load_sentences()
		self.load_acous_flis()
		self.load_timestamp()
		self.load_tsv()
		self.preprocess()


	def load_vocab(self):
		
		# import pdb; pdb.set_trace()
		self.vocab_src = []
		with codecs.open(self.path_vocab_src, encoding='UTF-8') as f:
			vocab_src_lines	= f.readlines()

		self.src_word2id = collections.OrderedDict()
		self.src_id2word = collections.OrderedDict()

		for i, word in enumerate(vocab_src_lines):
			if word == '\n':
				continue
			word = word.strip().split()[0] # remove \n
			self.vocab_src.append(word)
			self.src_word2id[word] = i
			self.src_id2word[i] = word

		if type(self.path_vocab_tgt) != type(None):
			self.vocab_tgt = []
			with codecs.open(self.path_vocab_tgt, encoding='UTF-8') as f:
				vocab_tgt_lines = f.readlines()

			self.tgt_word2id = collections.OrderedDict()
			self.tgt_id2word = collections.OrderedDict()
			for i, word in enumerate(vocab_tgt_lines):
				word = word.strip().split()[0] # remove \n
				self.vocab_tgt.append(word)
				self.tgt_word2id[word] = i
				self.tgt_id2word[i] = word


	def load_sentences(self):

		# import pdb; pdb.set_trace()
		with codecs.open(self.path_src, encoding='UTF-8') as f:
			self.src_sentences = f.readlines()

		if type(self.path_vocab_tgt) != type(None):
			with codecs.open(self.path_tgt, encoding='UTF-8') as f:
				self.tgt_sentences = f.readlines()

			assert len(self.src_sentences) == len(self.tgt_sentences), 'Mismatch src:tgt - {}:{}' \
						.format(len(self.src_sentences),len(self.tgt_sentences))
		
		if self.seqrev:
			for idx in range(len(self.src_sentences)):
				src_sent_rev = self.src_sentences[idx].strip().split()[::-1]
				tgt_sent_rev = self.tgt_sentences[idx].strip().split()[::-1]
				self.src_sentences[idx] = ' '.join(src_sent_rev)
				self.tgt_sentences[idx] = ' '.join(tgt_sent_rev)
	

	def load_tsv(self):

		""" laod the dd tag of each src word """

		if self.tsv_path == None:
			self.ddfd_seq_labs = None
		else:
			with codecs.open(self.tsv_path, encoding='UTF-8') as f:
				lines = f.readlines()

				lab_seq = []
				self.ddfd_seq_labs = []
				for line in lines:
					if line == '\n':
						if len(lab_seq):
							if self.seqrev:
								# reverse sequence for reverse decoding
								lab_seq = lab_seq[::-1]
							self.ddfd_seq_labs.append(lab_seq)
							lab_seq = []
						else:
							import pdb; pdb.set_trace()
					else:
						elems = line.strip().split('\t')
						tok = elems[0]
						if len(elems) == 3:
							lab = elems[-1][-1]
							if self.keep_filler:
								pass
							else:
								fl = elems[-2]
								if fl != '-':
									lab = 'E'
						elif len(elems) == 2:
							lab = elems[-1]
						elif len(elems) == 4:							
							lab = elems[-1][-1]
							if self.keep_filler:
								pass
							else:
								fl = elems[1]
								if fl != '-':
									lab = 'E'
						else:
							assert False, 'check tsv file, requires either 2 or 3 or 4 elems per line'

						lab_seq.append(lab)

			assert len(self.src_sentences)==len(self.ddfd_seq_labs), 'Mismatch src:ddfd_lab - {}:{}'.format(len(self.src_sentences),len(self.ddfd_seq_labs))


	def load_acous_flis(self):

		""" load acoustic npy file list """

		self.acous_flis = []
		self.acous_length_lis = []
		self.acous_spkids = []
		if type(self.acous_path) == type(None):
			pass
		else: 
			f = open(self.acous_path, 'r')
			lines = f.readlines()
			for line in lines:
				elems = line.strip().split()
				fname = elems[0]
				length = int(elems[1])
				spkid = elems[2].split('.')[0] # 'sw04004A' or 'CBL304-00069_sc.fbk'
				self.acous_flis.append(fname)
				self.acous_length_lis.append(length)
				self.acous_spkids.append(spkid)
			assert len(self.acous_flis) == len(self.src_sentences), 'mismatch acoustics and sentences {} : {}'.format(len(self.acous_flis), len(self.src_sentences))


	def load_timestamp(self):

		""" load timestamp per word per sentence: only loaded if acoustics is enabled """

		self.acous_times = []
		if type(self.timestamp_path) == type(None) or type(self.acous_path) == type(None):
			pass
		else:
			f = open(self.timestamp_path, 'r')
			lines = f.readlines()
			for line in lines:
				elems = line.strip().split()
				timelis = []
				for elem in elems:
					st,ed = elem.strip('[]').split(',')
					st = int(round(float(st) * 100))
					ed = int(round(float(ed) * 100))
					if ed <= st:
						import pdb; pdb.set_trace()
					#assert ed > st
					timelis.append([st,ed])
				self.acous_times.append(timelis) # [[]*l] * n_sent (type = int)
			assert len(self.acous_times) == len(self.src_sentences), 'mismatch acoustics timestamps and sentences {} : {}'.format(len(self.acous_times), len(self.src_sentences))
		# import pdb; pdb.set_trace()


	def preprocess(self):

		"""
			used for LAS + DD
			assume tsv; acous; timestamp paths all are available 
			Use:
				map word2id once for all epoches (improved data loading efficiency)
				shuffling is done in dataloader
		"""

		self.vocab_size = {'src': len(self.src_word2id)}
		print("num_vocab: ", self.vocab_size['src'])

		# declare temporary vars
		train_src_word_ids = []
		train_src_sentence_lengths = []
		train_acous_flis = []
		train_acous_spkids = []
		train_acous_lengths = []
		train_acous_times = []
		# dd sequence
		train_tags = []

		for idx in range(len(self.src_sentences)):
			# import pdb; pdb.set_trace()
			src_sentence = self.src_sentences[idx]
			if self.use_type == 'char':
				src_words = src_sentence.strip()
			else:	
				src_words = src_sentence.strip().split()

			# ignore long seq of words
			if len(src_words) > self.max_seq_len - 1:
				# src + EOS
				continue

			# emtry seq - caused by [vocalised-noise]
			if len(src_words) == 0:
				continue

			# ignore long seq of acoustic features
			if self.acous_length_lis[idx] > self.acous_max_len:
				continue
			else:
				train_acous_flis.append(self.acous_flis[idx])
				train_acous_spkids.append(self.acous_spkids[idx])
				train_acous_lengths.append(self.acous_length_lis[idx]) #lengths is accurate

				# if self.acous_times[idx][-1][1] > self.acous_length_lis[idx]:
				# 	print(self.acous_spkids[idx], self.acous_flis[idx])
				# 	print(self.acous_times[idx][-1][1], self.acous_length_lis[idx])
				# 	count_err += 1

				self.acous_times[idx][-1][1] = min(self.acous_times[idx][-1][1], self.acous_length_lis[idx]-1)
				train_acous_times.append(self.acous_times[idx])

			# source & tags
			src_ids = []
			tags = []

			if self.use_type == 'char':
				tag_idx = 0
				if self.tag_rev:
					tagcurr = int(self.ddfd_seq_labs[idx][tag_idx] == 'E') # E=1, O=0 (asr Ierror as lab = 'O'?)
				else:
					tagcurr = int(self.ddfd_seq_labs[idx][tag_idx] == 'O') # E=0, O=1 (asr Ierror as lab = 'E'?)
				for i in range(len(src_words)):
					word = src_words[i]
					if word == ' ':
						src_ids.append(SPC)
						tags.append(tagcurr)
						tag_idx += 1
						if self.tag_rev:
							tagcurr = int(self.ddfd_seq_labs[idx][tag_idx] == 'E') # E=1, O=0 (asr Ierror as lab = 'O'?)
						else:
							tagcurr = int(self.ddfd_seq_labs[idx][tag_idx] == 'O') # E=0, O=1 (asr Ierror as lab = 'E'?)
					elif word in self.src_word2id:
						src_ids.append(self.src_word2id[word])
						tags.append(tagcurr)
					else:
						src_ids.append(UNK)
						tags.append(tagcurr)
				src_ids.append(EOS)
				if self.tag_rev:
					tags.append(0)
				else:
					tags.append(1)
				assert src_ids[0] != PAD
				assert len(tags) == len(src_ids)

			elif self.use_type == 'word' or self.use_type == 'bpe':
				for i in range(len(src_words)):
					word = src_words[i]
					if self.tag_rev:
						tag = int(self.ddfd_seq_labs[idx][i] == 'E') # E=1, O=0 (asr Ierror as lab = 'O'?)
					else:
						tag = int(self.ddfd_seq_labs[idx][i] == 'O') # E=0, O=1 (asr Ierror as lab = 'E'?)
					if word in self.src_word2id:
						src_ids.append(self.src_word2id[word])
						tags.append(tag)
					else:
						src_ids.append(UNK)
						tags.append(tag)
				src_ids.append(EOS)
				if self.tag_rev:
					tags.append(0)
				else:
					tags.append(1)
				assert src_ids[0] != PAD
				assert len(tags) == len(src_ids)

			train_src_word_ids.append(src_ids)
			train_src_sentence_lengths.append(len(src_words)+1) # include one EOS
			train_tags.append(tags)

		# import pdb; pdb.set_trace()
		assert (len(train_src_word_ids) == len(train_acous_flis)), "train_src_word_ids != train_acous_flis"
		assert (len(train_src_word_ids) == len(train_tags)), "train_src_word_ids != train_tags"

		self.num_training_sentences = len(train_src_word_ids)
		print("num_sentences: ", self.num_training_sentences) # only those that are not too long

		# set class var to be used in batchify
		self.train_src_word_ids = train_src_word_ids
		self.train_src_sentence_lengths = train_src_sentence_lengths
		self.train_acous_flis = train_acous_flis # list of acous npy fnames 
		self.train_acous_spkids = train_acous_spkids
		self.train_acous_lengths = train_acous_lengths
		self.train_tags = train_tags
		self.train_acous_times = train_acous_times


	def construct_batches(self, is_train=False):

		"""
			Args:
				is_train: switch on shuffling is is_train
			Returns:
				batches of dataset
				src:            a  SPC c a t SPC s a t SPC o n SPC t h e SPC m a t EOS PAD PAD ...
		"""

		# organise by length
		_x = list(zip(self.train_src_word_ids, self.train_src_sentence_lengths, self.train_acous_flis, self.train_acous_spkids, self.train_acous_lengths, self.train_tags, self.train_acous_times))
		if is_train: 
			# _x = sorted(_x, key=lambda l:l[1])
			random.shuffle(_x)
		train_src_word_ids, train_src_sentence_lengths, train_acous_flis, train_acous_spkids, train_acous_lengths, train_tags, train_acous_times = zip(*_x)

		# manual batching to allow shuffling by pt dataloader
		n_batches = int(self.num_training_sentences/self.batch_size  + (self.num_training_sentences % self.batch_size > 0))
		batches = []
		for i in range(n_batches):
			i_start = i * self.batch_size
			i_end = min(i_start + self.batch_size, self.num_training_sentences)
			batch = {
				'src_word_ids':train_src_word_ids[i_start:i_end],
				'src_sentence_lengths': train_src_sentence_lengths[i_start:i_end],
				'acous_flis':train_acous_flis[i_start:i_end],
				'acous_spkids':train_acous_spkids[i_start:i_end],
				'acous_lengths':train_acous_lengths[i_start:i_end],
				'tags':train_tags[i_start:i_end],
				'acous_times':train_acous_times[i_start:i_end]
			}
			batches.append(batch)
		
		# pt dataloader
		params = {'batch_size': 1,
					'shuffle': is_train,
					'num_workers': 0}

		self.iter_set = IterDataset(batches, self.acous_norm, self.add_acous, self.add_timestamp)
		self.iter_loader = torch.utils.data.DataLoader(self.iter_set, **params)
		# import pdb; pdb.set_trace()


	def my_collate(self, batch):

		""" srcid, srclen, acous_feat, acouslen """

		srcid = [torch.LongTensor(item[0]) for item in batch]
		srclen = [item[1] for item in batch]
		acous_feat = [torch.Tensor(item[2]) for item in batch]
		acouslen = [item[3] for item in batch]

		srcid = torch.nn.utils.rnn.pad_sequence(srcid, batch_first=True) # b x l
		acous_feat = torch.nn.utils.rnn.pad_sequence(acous_feat, batch_first=True) # b x l x 40

		return [srcid, srclen, acous_feat, acouslen]


def load_pretrained_embedding(word2id, embedding_matrix, embedding_path):

	""" assign value to src_word_embeddings and tgt_word_embeddings """

	counter = 0
	with codecs.open(embedding_path, encoding="UTF-8") as f:
		for line in f:
			items = line.strip().split()
			if len(items) <= 2:
				continue
			word = items[0].lower()
			if word in word2id:
				id = word2id[word]
				vector = np.array(items[1:])
				embedding_matrix[id] = vector
				counter += 1	

	print('loaded pre-trained embedding:', embedding_path)
	print('embedding vectors found:', counter)

	return embedding_matrix


def load_pretrained_embedding_bpe(embedding_matrix):

	""" load bpe embedding; add <pad> as id=0 """

	bpemb = BPEmb(lang="en", vs=25000, dim=200)
	embedding_matrix[1:] = bpemb.vectors
	print('loaded bpe pre-trained embedding')
	print('embedding vectors count:', embedding_matrix.shape[0])

	return embedding_matrix

















