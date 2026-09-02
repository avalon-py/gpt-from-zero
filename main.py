with open('dataset/dataset.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
# >> chars = [ !$&',-.3:;?ABCDEF...]
vocab_size = len(chars)
# >> vocab_size = 65

# Tokenizing (str -> num)
stoi = {ch:i for i,ch in enumerate(chars)} # converts a character to integer (its position value in chars[])
itos = {i:ch for i,ch in enumerate(chars)} # converts an integer value to the character in that position

encode = lambda s: [stoi[c] for c in s] # take a sentence and stoi every character in it
# >> encode("hii") = [46, 47, 47]
decode = lambda i_list: ''.join(itos[i] for i in i_list) # take a list of ints and itos every ints in it
# >> decode(encode("hii")) = "hii"

import torch
data = torch.tensor(encode(text), dtype=torch.long) # wraps the full text from dataset.txt into a tensor (n-dim matrix)

# Train-Test Split
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
# Chunking
block_size = 8
train_data[:block_size+1]

