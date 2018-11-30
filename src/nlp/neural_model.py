from fastai import *
from fastai.text import *


class MultiModalRNN(RNNCore):
    def __init__(self, audio_sz, output_p, bias, tie_encoder:bool=True, **kwargs):
        super(MultiModalRNN, self).__init__(**kwargs)
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
    