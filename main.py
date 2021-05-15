import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from functools import partial
import pdb


PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'

corpus = [
    'గౌహతి పట్టణం బ్రహ్మపుత్రా నదీ తీరంలో నున్నది',
    'హైద్రాబాదు భారతదేశం లో ఐదవ అతిపెద్ద మహానగరం',
    'అస్సాం రాష్ట్ర రాజధాని గౌహతి',
    'పదకవితా పితామహుడిగా పేరు గాంచిన తాళ్ళపాక అన్నమాచార్యులు కడప జిల్లాలోని తాళ్లపాక అనే గ్రామంలో జన్మించారు',
    'కాకతీయుల రాజధాని ఓరుగల్లు'
    'హైద్రాబాదు కి మరో పేరు భాగ్యనగరం',
    'తిరుమల దేవస్థానం చిత్తూరు జిల్లాలో ఉంది',
]

locations = ['హైద్రాబాదు', 'గౌహతి', 'కడప', 'తిరుమల', 'చిత్తూరు', 'తాళ్ళపాక', 'అస్సాం', 'ఓరుగల్లు', 'భాగ్యనగరం']

vocab = set([word for sent in corpus for word in sent.split()])
vocab.add('<pad>')
vocab.add('<unk>')
ix_to_word = sorted(list(vocab))
word_to_ix = {word: index for index, word in enumerate(ix_to_word)}
print(f'vocab size: {len(word_to_ix)}')
print(f'pad token: {PAD_TOKEN} and pad index: {word_to_ix["<pad>"]}')
print(f'unk token: {UNK_TOKEN} and unk index: {word_to_ix["<unk>"]}')

train_sentences = [sentence.split() for sentence in corpus]
train_labels = [[1 if word in locations else 0 for word in sentence]for sentence in train_sentences]

print(train_sentences)
print(train_labels)


def custom_collate_fn(batch, window_size, word_to_ix):
    # window-pad each training example and convert the tokens to indices
    # pad the training examples so that all of them will be of same length
    # pad the training labels so that all of them will be of same length
    # get the number of words in each training example
    x, y = zip(*batch)

    def pad_window(sentence, window_size=2):
        window = [PAD_TOKEN] * window_size
        return window + sentence + window

    x = [pad_window(s, window_size=window_size) for s in x]

    def convert_tokens_to_indices(sentence, word_to_ix):
        return [word_to_ix.get(token, word_to_ix[UNK_TOKEN]) for token in sentence]

    x = [convert_tokens_to_indices(s, word_to_ix) for s in x]

    pad_token_ix = word_to_ix[PAD_TOKEN]

    # pad_sequence function expects input to be a tensor, so we turn x into a tensor
    x = [torch.LongTensor(x_i) for x_i in x]
    x_padded = nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=pad_token_ix)

    # Before padding labels, find out how many words existed in each training example
    lengths = [len(label) for label in y]
    lengths = torch.LongTensor(lengths)

    y = [torch.LongTensor(y_i) for y_i in y]
    y_padded = nn.utils.rnn.pad_sequence(y, batch_first=True, padding_value=0)

    return x_padded, y_padded, lengths


# prepare data and create a data loader instance
data = list(zip(train_sentences, train_labels))
window_size = 2
collate_fn = partial(custom_collate_fn, window_size=window_size, word_to_ix=word_to_ix)

loader = DataLoader(data, batch_size=2, shuffle=True, collate_fn=collate_fn)


# network
class Network(nn.Module):
    def __init__(self, hyperparameters, vocab_size, pad_ix=0):
        super(Network, self).__init__()

        self.window_size = hyperparameters['window_size']
        self.embed_dim = hyperparameters['embed_dim']
        self.hidden_dim = hyperparameters['hidden_dim']

        # Embedding layer
        self.embeds = nn.Embedding(vocab_size, self.embed_dim, padding_idx=pad_ix)

        # Hidden layer
        full_window_size = 2 * window_size + 1
        self.hidden_layer = nn.Sequential(
            nn.Linear(full_window_size * self.embed_dim, self.hidden_dim),
            nn.Tanh()
        )

        # Output Layer
        self.output_layer = nn.Linear(self.hidden_dim, 1)

        # Probabilities
        self.probabilities = nn.Sigmoid()

    def forward(self, inputs):
        # inputs is a tensor of size BxL i.e. a batch of token indices
        batch_size, sent_len = inputs.size()

        # Create token windows for each sentence. Each token window will have 2*N + 1 tokens (i.e. N context words
        # left side and N context words right side of the center word
        # token_windows will be of shape (batch_size, num_token_windows_per_example, num_of_tokens_in_each_window)
        token_windows = inputs.unfold(1, 2 * self.window_size + 1, 1)
        _, num_token_windows, _ = token_windows.size()

        assert token_windows.size() == (batch_size, num_token_windows, 2 * self.window_size + 1)

        # input shape: (batch_size, num_token_windows, num_tokens_in_window)
        # output shape: (batch_size, num_token_windows, num_tokens_in_window, embed_dim)
        embedded_windows = self.embeds(token_windows)

        # Reshaping the embedded windows so it becomes (batch_size, num_token_windows, num_tokens_in_window * embed_dim)
        embedded_windows = embedded_windows.view(batch_size, num_token_windows, -1)

        # output shape: (batch_size, number_of_token_windows, num_hidden_units)
        layer_1 = self.hidden_layer(embedded_windows)

        # output shape: (batch_size, number_of_token_windows, 1)
        output = self.output_layer(layer_1)

        # no change in shape
        output = self.probabilities(output)

        # output shape: (batch_size, number_ok_token_windows*1)
        output = output.view(batch_size, -1)

        return output


# Training
hypers = {
    "window_size": window_size,
    "embed_dim": 25,
    "hidden_dim": 25
}
#torch.manual_seed(1234)
model = Network(hypers, vocab_size=len(word_to_ix))

# optimizer
learning_rate = 0.001
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# define a custom loss function
def loss_function(batch_outputs, batch_labels, batch_lengths):
    bceloss = nn.BCELoss()
    loss = bceloss(batch_outputs, batch_labels.float())

    loss = loss / batch_lengths.sum().float()
    return loss


# training loop
num_epochs = 100
for epoch in range(num_epochs):
    total_loss = 0
    for batch_inputs, batch_labels, batch_lengths in loader:
        optimizer.zero_grad()
        batch_outputs = model.forward(batch_inputs)
        loss = loss_function(batch_outputs, batch_labels, batch_lengths)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if epoch % 10 == 0:
        print(f'Epoch: {epoch}, loss: {total_loss}')

# test sentence
test_sentence = "తెలంగాణ రాష్ట్ర రాజధాని హైద్రాబాదు"
test_labels = [1, 0, 0, 1]
print(f'original test sentence: {test_sentence}')
test_sentence = [word_to_ix.get(token, word_to_ix[UNK_TOKEN]) for token in test_sentence.split()]
restored_test_sentence = [ix_to_word[index] for index in test_sentence]
print(f'restored test sentence: {restored_test_sentence}')

# window pad test sentence
padding = window_size*[word_to_ix[PAD_TOKEN]]
test_sentence = padding + test_sentence + padding
restored_test_sentence = [ix_to_word[index] for index in test_sentence]
print(f'padded test sentence: {restored_test_sentence}')

test_sentence = torch.tensor(test_sentence)
test_labels = torch.tensor(test_labels)

test_sentence = test_sentence.view(1, -1)
test_labels = test_labels.view(1, -1)
test_outputs = model.forward(test_sentence)
print('Predictions for test sentence:', test_outputs)