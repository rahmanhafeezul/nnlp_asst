import io
import numpy as np
import torch
from torch.autograd import Variable
import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.utils import shuffle
import torch.nn.functional as F
import pdb
import random
import re
import sys

class CNN(nn.Module):
    def __init__(self, batch_size, max_sent_len, word_dim, vocab_size, class_size, filters, filter_num, drop_out, embed_matrix, model_type):
        super(CNN, self).__init__()

        self.model_name = model_type
        self.max_len_sorpus = max_sent_len
        self.embed_dim = word_dim
        self.unique_count = vocab_size
        self.num_channel = 1
        self.embedding = nn.Embedding(self.unique_count + 2, self.embed_dim, padding_idx=self.unique_count + 1)
        if self.model_name=="static":
        	print("static")
        	self.embedding.weight.data.copy_(torch.from_numpy(embed_matrix))
        	self.embedding.weight.requires_grad = False
        elif self.model_name=="non-static":
        	print("non-static")
        	self.embedding.weight.data.copy_(torch.from_numpy(embed_matrix))
        elif self.model_name=="multichannel":
        	print("Multichannel")
        	self.embedding.weight.data.copy_(torch.from_numpy(embed_matrix))
        	self.embedding2 = nn.Embedding(self.unique_count + 2, self.embed_dim, padding_idx=self.unique_count + 1)
        	self.embedding2.weight.data.copy_(torch.from_numpy(embed_matrix))
        	self.embedding2.weight.requires_grad = False
        	self.num_channel = 2
        self.conv1 = nn.Conv1d(self.num_channel, 100, self.embed_dim * 3, stride=self.embed_dim)
        self.conv2 = nn.Conv1d(self.num_channel, 100, self.embed_dim * 4, stride=self.embed_dim)
        self.conv3 = nn.Conv1d(self.num_channel, 100, self.embed_dim * 5, stride=self.embed_dim)

        self.fc = nn.Linear(sum([100, 100, 100]), class_size)

    def forward(self, inp):
    	# pdb.set_trace()
        x = self.embedding(inp).view(-1, 1, self.embed_dim * self.max_len_sorpus)
        if self.model_name == "multichannel":
        	x2 = self.embedding2(inp).view(-1, 1, self.embed_dim * self.max_len_sorpus)
        	x = torch.cat((x, x2), 1)
        x1 = F.max_pool1d(F.relu(self.conv1(x)), self.max_len_sorpus - 3 + 1).view(-1, 100)
        x2 = F.max_pool1d(F.relu(self.conv2(x)), self.max_len_sorpus - 4 + 1).view(-1, 100)
        x3 = F.max_pool1d(F.relu(self.conv3(x)), self.max_len_sorpus - 5 + 1).view(-1, 100)

        x = torch.cat((x1, x2, x3), 1)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc(x)

        return x


def read_data(filename):
	lines = open(filename, "r").readlines()
	x = list()
	y = list()
	vocab = set()
	for line in lines:
		line = re.sub(r"[-()\"#/@;:<>{}`+=~.!?,]", "", line)
		line = line.split("|||")
		x.append(line[1].split())
		y.append(line[0].rstrip())
		for word in line[1].split():
			vocab.add(word)
	return x, y, vocab
def read_glove_embed(glove_location):
	words_glove = set()
	vectors = list()
	word2idx = dict()
	idx = 0;
	lines = open(glove_location, "r").readlines()
	for line in lines:
		line = line.split()
		words_glove.add(line[0])
		vectors.append(np.array(line[1:]).astype(np.float))
		word2idx[line[0]] = idx
		idx += 1
	glove = {w: vectors[word2idx[w]] for w in words_glove}
	return glove, words_glove

def form_embed_matrix(glove, vocab, words_glove, idx_to_word, embed_dim):
	embed_matrix = []
	for i in range(len(vocab)):
		word = idx_to_word[i]
		if word in words_glove:
			embed_matrix.append(glove[word])
		else:
			embed_matrix.append(np.random.uniform(-0.01, 0.01, embed_dim).astype("float32"))
	embed_matrix.append(np.random.uniform(-0.01, 0.01, embed_dim).astype("float32"))
	embed_matrix.append(np.zeros(embed_dim).astype("float32"))
	embed_matrix = np.array(embed_matrix)
	return embed_matrix

def get_current_batch_max_len(list_of_lists):
	curr_max = 0
	for curr_list in list_of_lists:
		curr_max = max(curr_max, len(curr_list))
	return curr_max

def save_checkpoint(state, is_best, filename):
	if is_best:
		print("Found better model, saving to:", filename)
		torch.save(state, filename)



def perform_train(train_x, train_y, val_x, val_y, test_x, test_y, word_to_idx, vocab, max_sent_len, labels, embed_matrix, batch_size, embed_dim, model, saved_model_path):
	
	best_valid_acc = 0
	best_loss = np.inf
	if(torch.cuda.is_available):
		model=model.cuda()
	loss_fn = nn.CrossEntropyLoss()
	if(torch.cuda.is_available):
		loss_fn=loss_fn.cuda()
	# optimizer = optim.Adam(model.parameters(), lr=0.001)
	# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, nesterov=True)
	# optimizer = optim.RMSprop(model.parameters(), lr=0.001)
	optimizer = optim.Adadelta(model.parameters(), lr=0.001)
	val_accuracy_values = []
	for epoch in range(20):
		is_best=False
		train_x, train_y = shuffle(train_x, train_y)
		total_loss = 0
		for i in range(0, len(train_x), batch_size):
			batch_range = min(batch_size, len(train_x)-i)
			# max_sent_len = get_current_batch_max_len(train_x[i:i+batch_range])
			batch_x = [[word_to_idx[w] for w in sent] + [len(vocab) + 1] * (max_sent_len - len(sent)) for sent in train_x[i:i + batch_range]]
			batch_y = [labels.index(c) for c in train_y[i:i + batch_range]]
			batch_x = Variable(torch.LongTensor(batch_x))
			batch_y = Variable(torch.LongTensor(batch_y))
			if(torch.cuda.is_available):
				batch_x=batch_x.cuda()
				batch_y=batch_y.cuda()
			if(i%10000==0):
				print("Epoch:", epoch, " batch:", i, "curr_max_len:", max_sent_len)
			optimizer.zero_grad()
			model.train()
			pred = model(batch_x)
			loss = loss_fn(pred, batch_y)
			total_loss+=loss.cpu().data.item()
			# print("training Loss:", loss)
			loss.backward()
			# nn.utils.clip_grad_norm_(model.parameters(), max_norm=3)
			optimizer.step()
		print("Total loss:", total_loss)
		val_accuracy, val_loss = perform_val(model, val_x, val_y, word_to_idx, vocab, max_sent_len, labels)
		val_accuracy_values.append(val_accuracy)
		print("Val acc:", val_accuracy_values)
		print("val loss:", val_loss)
		is_best = val_accuracy > best_valid_acc
		if val_accuracy == best_valid_acc and best_loss > val_loss:
			is_best = True
			best_loss = val_loss
		best_valid_acc = max(val_accuracy, best_valid_acc)
		save_checkpoint({'epoch': epoch + 1,'state_dict': model.state_dict(),'best_acc': best_valid_acc,'optimizer' : optimizer.state_dict()}, is_best, saved_model_path)

def perform_val(model, val_x, val_y, word_to_idx, vocab, max_sent_len, labels):
	# max_sent_len = get_current_batch_max_len(val_x)
	val_x = [[word_to_idx[w] if w in vocab else len(vocab) for w in sent] + [len(vocab) + 1] * (max_sent_len - len(sent)) for sent in val_x]
	val_y = [labels.index(c) for c in val_y]
	val_y = Variable(torch.LongTensor(val_y))
	val_x = Variable(torch.LongTensor(val_x))
	model.eval()
	if(torch.cuda.is_available):
		val_x = val_x.cuda()
		val_y = val_y.cuda()
	# pdb.set_trace()
	output = model(val_x)
	loss = F.cross_entropy(output, val_y)
	pred = np.argmax(output.cpu().data.numpy(), axis=1)
	val_y = val_y.data.long().cpu().numpy()
	is_right = []
	for i in range(len(val_y)):
		if(val_y[i]==pred[i]):
			is_right.append(1)
		else:
			is_right.append(0)
	acc = (float(sum(is_right)) / float(len(pred))) * 100.0
	return acc, loss

def perform_test(test_x, test_y, word_to_idx, vocab, max_sent_len, labels, embed_matrix, batch_size, embed_dim, model):
	model.eval()
	# max_sent_len = get_current_batch_max_len(test_x)
	test_x = [[word_to_idx[w] if w in vocab else len(vocab) for w in sent] + [len(vocab) + 1] * (max_sent_len - len(sent)) for sent in test_x]
	test_x = Variable(torch.LongTensor(test_x))
	if(torch.cuda.is_available):
		model = model.cuda()
		test_x = test_x.cuda()
	output = model(test_x)
	pred = np.argmax(output.cpu().data.numpy(), axis=1)
	final_pred = []
	for curr_pred in pred:
		final_pred.append(labels[curr_pred])
	return final_pred


def write_to_file(filename, curr_list):
	f = open(filename, "w")
	for line in curr_list:
		f.write(line)
		f.write("\n")
	f.close()

def main(train_data_filename, val_data_filename, test_data_filename, glove_location, batch_size, mode, embed_dim, saved_model_path):
	glove, words_glove = read_glove_embed(glove_location)
	train_x, train_y, vocab_train = read_data(train_data_filename)

	print(train_x[:10])
	print(train_y[:10])
	val_x, val_y, vocab_val = read_data(val_data_filename)
	test_x, test_y, vocab_test = read_data(test_data_filename)
	max_sent_len = max([len(sent) for sent in train_x + val_x + test_x])
	vocab = set()
	vocab.update(vocab_train)
	vocab.update(vocab_val)
	vocab.update(vocab_test)
	vocab = sorted(list(vocab))
	labels = sorted(list(set(train_y)))
	word_to_idx = {w: i for i, w in enumerate(vocab)}
	idx_to_word = {i: w for i, w in enumerate(vocab)}
	embed_matrix = form_embed_matrix(glove, vocab, words_glove, idx_to_word, embed_dim)
	model = CNN(batch_size, max_sent_len, embed_dim, len(vocab), len(labels), [3, 4, 5], [100, 100, 100], 0.5, embed_matrix, "non-static")

	if(mode=="train"):
		print("Inside train")
		perform_train(train_x, train_y, val_x, val_y, test_x, test_y, word_to_idx, vocab, max_sent_len, labels, embed_matrix, batch_size, embed_dim, model, saved_model_path)
		# pdb.set_trace()
	if(mode=="test"):
		print("Inside test")
		checkpoint = torch.load(saved_model_path)
		best_valid_acc = checkpoint['best_acc']
		model.load_state_dict(checkpoint['state_dict'])
		print("Loaded checkpoint with val acc:", best_valid_acc, " in epoch:", checkpoint['epoch'])

		val_results = perform_test(val_x, val_y, word_to_idx, vocab, max_sent_len, labels, embed_matrix, batch_size, embed_dim, model)
		write_to_file("val_results.txt", val_results)


		test_results = perform_test(test_x, test_y, word_to_idx, vocab, max_sent_len, labels, embed_matrix, batch_size, embed_dim, model)
		write_to_file("test_results.txt", test_results)
		

if __name__ == "__main__":
	# torch.manual_seed(12)
	# torch.cuda.manual_seed(12)
	# np.random.seed(12)
	# random.seed(12)

	# torch.backends.cudnn.deterministic=True
	mode = sys.argv[1]
	print("Mode:", mode)
	batch_size = 50
	embed_dim = 300
	print(embed_dim)
	glove_location = "glove.6B/glove.6B."+str(embed_dim)+"d.txt"
	train_data_filename = "topicclass/topicclass_train.txt"
	# train_data_filename = "sst_train.txt"
	val_data_filename = "topicclass/topicclass_valid.txt"
	# val_data_filename = "sst_test.txt"
	test_data_filename = "topicclass/topicclass_test.txt"
	# test_data_filename = "sst_test.txt"
	saved_model_path = "best_model.pt"
	main(train_data_filename, val_data_filename, test_data_filename, glove_location, batch_size, mode, embed_dim, saved_model_path)