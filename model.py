import torch
from typing import Type
from torch import nn
from dataset import TextDataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LanguageModel(nn.Module):
    def __init__(self, dataset: TextDataset, embed_size: int = 256, hidden_size: int = 256,
                 rnn_type: Type = nn.RNN, rnn_layers: int = 1):
        """
        Model for text generation
        :param dataset: text data dataset (to extract vocab_size and max_length)
        :param embed_size: dimensionality of embeddings
        :param hidden_size: dimensionality of hidden state
        :param rnn_type: type of RNN layer (nn.RNN or nn.LSTM)
        :param rnn_layers: number of layers in RNN
        """
        super().__init__()
        self.dataset = dataset  # required for decoding during inference
        self.vocab_size_en = dataset.vocab_size_en
        self.vocab_size_de = dataset.vocab_size_de
        self.max_length = dataset.max_length

        """
        YOUR CODE HERE (⊃｡•́‿•̀｡)⊃━✿✿✿✿✿✿
        Create necessary layers
        """
        self.embedding_en = nn.Embedding(self.vocab_size_en, embed_size, padding_idx=self.dataset.pad_id_en)
        self.embedding_de = nn.Embedding(self.vocab_size_de, embed_size, padding_idx=self.dataset.pad_id_de)
        self.rnn_encoder = rnn_type(embed_size, hidden_size, rnn_layers, batch_first=True)
        self.rnn_decoder = rnn_type(embed_size + hidden_size, hidden_size, rnn_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, self.vocab_size_en)

    def forward(self, indices: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Compute forward pass through the model and
        return logits for the next token probabilities
        :param indices: LongTensor of encoded tokens of size (batch_size, input length)
        :param lengths: LongTensor of lengths of size (batch_size, )
        :return: FloatTensor of logits of shape (batch_size, output length, vocab_size)
        """
        """
        YOUR CODE HERE (⊃｡•́‿•̀｡)⊃━✿✿✿✿✿✿
        Convert indices to embeddings, pass them through recurrent layers
        and apply output linear layer to obtain the logits
        """
        emb = pack_padded_sequence(self.embedding_de(indices[0]), lengths[0], batch_first=True, enforce_sorted=False)

        out_enc, h = self.rnn_encoder(emb)
        
        out_enc, _ = pad_packed_sequence(out_enc, batch_first=True, total_length=indices[1].shape[1])
        emb_en = self.embedding_en(indices[1])

        emb_en = torch.concat((emb_en, torch.tile(out_enc[:, -1:, :], (1, emb_en.shape[1], 1))), axis=2)
        emb_en = pack_padded_sequence(emb_en, lengths[1], batch_first=True, enforce_sorted=False)
        
        out, _ = self.rnn_decoder(emb_en, h)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=lengths[1].max())
        logits = self.linear(out)
        
        return logits

    @torch.inference_mode()
    def inference(self, prefix: str = '', temp: float = 1.) -> str:
        """
        Generate new text with an optional prefix
        :param prefix: prefix to start generation
        :param temp: sampling temperature
        :return: generated text
        """
        self.eval()
        # This is a placeholder, you may remove it.
        generated = prefix + ', а потом купил мужик шляпу, а она ему как раз.'
        """
        YOUR CODE HERE (⊃｡•́‿•̀｡)⊃━✿✿✿✿✿✿
        Encode the prefix (do not forget the BOS token!),
        pass it through the model to accumulate RNN hidden state and
        generate new tokens sequentially, sampling from categorical distribution,
        until EOS token or reaching self.max_length.
        Do not forget to divide predicted logits by temperature before sampling
        """
        with torch.no_grad():
            tokenized_prefix = self.dataset.text2ids(prefix)
            emb = self.embedding_de(torch.tensor(tokenized_prefix).unsqueeze(0).to(device = next(self.parameters()).device))
            eos_emb = self.dataset.eos_id_en
            tokenized_translation = []
            out_enc, h = self.rnn_encoder(emb)

            beginings = torch.tensor([[self.dataset.bos_id_en]])
            emb = self.embedding_de(beginings.to(device = next(self.parameters()).device))
            
            emb = torch.concat((emb, out_enc[:, -1:, :]), axis=2)
            for _ in range(self.dataset.max_length):
                out, h = self.rnn_decoder(emb, h)
                logits = self.linear(out)[:, -1:, :] / temp
                probs = torch.softmax(logits[0], dim=-1)
                new_token = torch.multinomial(probs, num_samples=1)
                if new_token == eos_emb:
                    break
                tokenized_translation.append(new_token.item())
                
                emb = self.embedding_en(new_token)
                emb = torch.concat((emb, out_enc[:, -1:, :]), axis=2)

            return self.dataset.ids2text(tokenized_translation)
