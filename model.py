import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.functional as F
import torch.optim as optim

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTMClassifier(nn.Module):

	def __init__(self, embedding_dim, hidden_dim, output_size):

		super(LSTMClassifier, self).__init__()

		self.embedding_dim = embedding_dim
		self.hidden_dim = hidden_dim

		self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1)

		self.hidden2out = nn.Linear(hidden_dim, output_size)

		self.dropout_layer = nn.Dropout(p=0.2)


	def init_hidden(self, batch_size):
		return(autograd.Variable(torch.randn(1, batch_size, self.hidden_dim).cuda()),
						autograd.Variable(torch.randn(1, batch_size, self.hidden_dim)).cuda())


	def forward(self, batch):
		
		self.hidden = self.init_hidden(batch.size(0))
		#print('size: ',batch.shape)
		outputs, (ht, ct) = self.lstm(batch.transpose(0,1), self.hidden)

		# ht is the last hidden state of the sequences
		# ht = (1 x batch_size x hidden_dim)
		# ht[-1] = (batch_size x hidden_dim)
		output = self.dropout_layer(ht[-1])
		output = self.hidden2out(output)
		output = torch.sigmoid(output)
		total_output = torch.zeros(outputs.size(0),outputs.size(1),3).float()
		for x_idx in range(outputs.size(0)):
			pre_h = self.dropout_layer(outputs[x_idx])
			pre_h = self.hidden2out(pre_h)
			total_output[x_idx] = torch.sigmoid(pre_h)

		return output,total_output