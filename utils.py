import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AttnLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, character):
        # character (str): set of the possible characters.
        # [GO] for the start token of the attention decoder. [s] for end-of-sentence token.
        list_token = ['[GO]', '[s]','[UNK]']
        #list_character = list(character)
        list_character = character
        self.character = list_token + list_character

        self.dict = {}
        for i, char in enumerate(self.character):
            self.dict[char] = i

    def encode(self, text, level, batch_max_length=25):
        """ convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
            batch_max_length: max length of text label in the batch. 25 by default

        output:
            text : the input of attention decoder. [batch_size x (max_length+2)] +1 for [GO] token and +1 for [s] token.
                text[:, 0] is [GO] token and text is padded with [GO] token after [s] token.
            length : the length of output of attention decoder, which count [s] token also. [3, 7, ....] [batch_size]
        """
        length = [len(s) + 1 for s in text]  # +1 for [s] at end of sentence.
        batch_max_length += 1
        batch_text = torch.LongTensor(len(text), batch_max_length + 1).fill_(0)
        for i, t in enumerate(text):
            if level == "char":
                text = list(t)
                text.append('[s]')
            else:
                text = t.split(' ')
                text.append('[s]')
                text = [' ' if j =='' else j for j in text]
                
            text = [int(self.dict.get(item,2)) for item in text]
            try:
                batch_text[i][1:1 + len(text)] = torch.LongTensor(text)  # batch_text[:, 0] = [GO] token
            except:
                idx = min(len(text), batch_max_length)
                batch_text[i][1:idx+1] = torch.LongTensor(text[:idx])  # batch_text[:, 0] = [GO] token
        return (batch_text.to(device), torch.IntTensor(length).to(device))

    def decode(self, text_index, length, level):
        """ convert text-index into text-label. """
        texts = []
        for index, l in enumerate(length):
            if level == "char":
                text = ''.join([self.character[i] for i in text_index[index, :]])
            else:
                text = ' '.join([self.character[i] for i in text_index[index, :]])
            texts.append(text)
        return texts


class Averager(object):
    """Compute average for torch.Tensor, used for loss average."""

    def __init__(self):
        self.reset()

    def add(self, v):
        count = v.data.numel()
        v = v.data.sum()
        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res
