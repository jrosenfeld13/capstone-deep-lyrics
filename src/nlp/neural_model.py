from fastai import *
from fastai.text import *


class MultiModalRNN(RNNCore):
    
    def __init__(self, audio_sz, output_p, bias, vocab_sz:int, emb_sz:int,
                 n_hid:int, n_layers:int, pad_token:int, bidir:bool=False,
                 hidden_p:float=0.2, input_p:float=0.6, embed_p:float=0.1,
                 weight_p:float=0.5, qrnn:bool=False, tie_encoder:bool=True):
        
        super().__init__(vocab_sz=vocab_sz, emb_sz=emb_sz,
                         n_hid=n_hid, n_layers=n_layers, pad_token=pad_token,
                         bidir=bidir,hidden_p=hidden_p, input_p=input_p,
                         embed_p=embed_p, weight_p=weight_p, qrnn=qrnn)
        self.rnns = None
        self.audio_sz = audio_sz
        self.multimode = [nn.LSTM(emb_sz + audio_sz if l == 0 else n_hid,
                                  (n_hid if l != n_layers - 1 else emb_sz)//self.ndir,
                                  1, bidirectional=bidir) for l in range(n_layers)]
        self.multimode = [WeightDropout(rnn, weight_p) for rnn in self.multimode]
        self.multimode = torch.nn.ModuleList(self.multimode)
        
        if tie_encoder:
            enc = self.encoder
        else:
            enc = None
        
        self.multidecoder = LinearDecoder(vocab_sz,
                                          emb_sz,
                                          output_p,
                                          tie_encoder=enc,
                                          bias=bias)
        
    def forward(self, input:LongTensor, input_audio:Tensor)->Tuple[Tensor,Tensor,Tensor]:
        sl,bs = input.size()
        if bs!=self.bs:
            self.bs=bs
            self.reset()
        raw_output = self.input_dp(self.encoder_dp(input))
        raw_output = torch.cat([raw_output, input_audio], dim=2)
        new_hidden,raw_outputs,outputs = [],[],[]
        for l, (rnn,hid_dp) in enumerate(zip(self.multimode, self.hidden_dps)):
            raw_output, new_h = rnn(raw_output, self.hidden[l])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if l != self.n_layers - 1: raw_output = hid_dp(raw_output)
            outputs.append(raw_output)
        self.hidden = to_detach(new_hidden)
        
        output = self.multidecoder.output_dp(outputs[-1])
        decoded = self.multidecoder.decoder(output.view(output.size(0)*output.size(1),
                                                        output.size(2)))
        
        return decoded, raw_outputs, outputs
    
    def _one_hidden(self, l:int)->Tensor:
        "Return one hidden state."
        nh = (self.n_hid if l != self.n_layers - 1 else self.emb_sz)//self.ndir
        return self.weights.new(self.ndir, self.bs, nh).zero_()

    def reset(self):
        "Reset the hidden states."
        [r.reset() for r in self.multimode if hasattr(r, 'reset')]
        self.weights = next(self.parameters()).data
        if self.qrnn: self.hidden = [self._one_hidden(l) for l in range(self.n_layers)]
        else: self.hidden = [(self._one_hidden(l), self._one_hidden(l)) for l in range(self.n_layers)]

class MultiLinearDecoder(nn.Module):
    "To go on top of a RNNCore module and create a Language Model."

    initrange=0.1

    def __init__(self, n_out:int, n_hid:int, audio_sz:int,
                 output_p:float, tie_encoder:nn.Module=None,
                 bias:bool=True):
        super().__init__()
        self.decoder = nn.Linear(n_hid + audio_sz, n_out, bias=bias)
        self.decoder.weight.data.uniform_(-self.initrange, self.initrange)
        self.output_dp = RNNDropout(output_p)
        if bias: self.decoder.bias.data.zero_()
        if tie_encoder: self.decoder.weight = tie_encoder.weight

    def forward(self, input:Tuple[Tensor,Tensor])->Tuple[Tensor,Tensor,Tensor]:
        raw_outputs, outputs = input
        output = self.output_dp(outputs[-1])
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded, raw_outputs, outputs
    
class MultiModalPostRNN(RNNCore):
    
    def __init__(self, audio_sz, output_p, bias, vocab_sz:int, emb_sz:int,
                 n_hid:int, n_layers:int, pad_token:int, bidir:bool=False,
                 hidden_p:float=0.2, input_p:float=0.6, embed_p:float=0.1,
                 weight_p:float=0.5, qrnn:bool=False, tie_encoder:bool=False):
        
        assert tie_encoder==False,\
        "Encoder and Decoder cannot be tied for this architecture. Set `tie_encoder=False`"
        
        super().__init__(vocab_sz=vocab_sz, emb_sz=emb_sz,
                         n_hid=n_hid, n_layers=n_layers, pad_token=pad_token,
                         bidir=bidir, hidden_p=hidden_p, input_p=input_p,
                         embed_p=embed_p, weight_p=weight_p, qrnn=qrnn)
        
        self.rnns = None
        self.multimode = [nn.LSTM(emb_sz if l == 0 else n_hid,
                                (n_hid if l != n_layers - 1 else emb_sz)//self.ndir,
                                1, bidirectional=bidir) for l in range(n_layers)]
        self.multimode = [WeightDropout(rnn, weight_p) for rnn in self.multimode]
        self.multimode = torch.nn.ModuleList(self.multimode)
        self.audio_sz = audio_sz
        
        if tie_encoder:
            enc = self.encoder
        else:
            enc = None
        
        self.multidecoder = MultiLinearDecoder(vocab_sz,
                                               emb_sz,
                                               audio_sz,
                                               output_p,
                                               tie_encoder=enc,
                                               bias=bias)
        
    def forward(self, input:LongTensor, input_audio:Tensor)->Tuple[Tensor,Tensor,Tensor]:
        sl,bs = input.size()
        if bs!=self.bs:
            self.bs=bs
            self.reset()
        raw_output = self.input_dp(self.encoder_dp(input))
        new_hidden,raw_outputs,outputs = [],[],[]
        for l, (rnn,hid_dp) in enumerate(zip(self.multimode, self.hidden_dps)):
            raw_output, new_h = rnn(raw_output, self.hidden[l])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if l != self.n_layers - 1: raw_output = hid_dp(raw_output)
            outputs.append(raw_output)
        self.hidden = to_detach(new_hidden)
        
        output = self.multidecoder.output_dp(outputs[-1])
        output = torch.cat([raw_output, input_audio], dim=2)
        decoded = self.multidecoder.decoder(output.view(output.size(0)*output.size(1),
                                                        output.size(2)))
        
        return decoded, raw_outputs, outputs
    
    def _one_hidden(self, l:int)->Tensor:
        "Return one hidden state."
        nh = (self.n_hid if l != self.n_layers - 1 else self.emb_sz)//self.ndir
        return self.weights.new(self.ndir, self.bs, nh).zero_()

    def reset(self):
        "Reset the hidden states."
        [r.reset() for r in self.multimode if hasattr(r, 'reset')]
        self.weights = next(self.parameters()).data
        if self.qrnn: self.hidden = [self._one_hidden(l) for l in range(self.n_layers)]
        else: self.hidden = [(self._one_hidden(l), self._one_hidden(l)) for l in range(self.n_layers)]
    