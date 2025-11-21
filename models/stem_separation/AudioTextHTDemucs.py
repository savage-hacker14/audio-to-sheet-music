import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention

from demucs.htdemucs import HTDemucs
from transformers import AutoTokenizer, ClapModel, ClapTextModel

# from typing import Optional
from fractions import Fraction
from einops import rearrange

class AudioTextHTDemucs(nn.Module):
    """
    Wrapper around HTDemucs that adds a single audio<->text cross attention layer.
    CLAP text encoder produces text embeddings that serve as keys/values.
    Audio encoder features (bottleneck) serve as queries.
    """

    def __init__(
        self,
        htdemucs_model: HTDemucs,           # Preloaded HTDemucs instance
        clap_encoder: ClapModel,            # Preloaded CLAP model
        clap_tokenizer,                     # Tokenizer for CLAP model
        model_dim=512,                      # dim of HTDemucs bottleneck
        text_dim=512,                       # CLAP text dimension
        n_fft=4096,
        hop_length=1024,
        num_heads=8,
        sample_rate=44100                   # Sample rate [Hz]
    ):
        super().__init__()

        self.htdemucs = htdemucs_model
        self.clap_text = clap_encoder
        self.tokenizer = clap_tokenizer
        
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sample_rate = sample_rate

        # Project CLAP embedding -> model dim (for attention KV)
        self.text_proj = nn.Linear(text_dim, model_dim)

        # Normalize two streams before attention
        self.norm_a = nn.LayerNorm(model_dim)
        self.norm_t = nn.LayerNorm(model_dim)

        # Multi-head cross attention (audio Q → text KV)
        self.cross_attn = MultiheadAttention(
            embed_dim=model_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # FFN after attention
        self.ff = nn.Sequential(
            nn.Linear(model_dim, 4 * model_dim),
            nn.GELU(),
            nn.Linear(4 * model_dim, model_dim)
        )

        self.norm_post = nn.LayerNorm(model_dim)
        
        # Final project to singular stem 
        self.final_proj = nn.Conv1d(96, 4, kernel_size=1)       # TODO: Fix hard coded value

    def _htdemucs_full_enc(self, wav):
        """
        Helper function to do forward pass for all the encoding layers + cross attention
        """
        length = wav.shape[-1]
        length_pre_pad = None
        if self.htdemucs.use_train_segment:
            if self.training:
                self.segment = Fraction(wav.shape[-1], self.sample_rate)
            else:
                training_length = int(self.segment * self.sample_rate)
                if wav.shape[-1] < training_length:
                    length_pre_pad = wav.shape[-1]
                    wav = F.pad(wav, (0, training_length - length_pre_pad))
                    
        z = self.htdemucs._spec(wav)
        mag = self.htdemucs._magnitude(z).to(wav.device)
        x = mag

        B, C, Fq, T = x.shape

        # unlike previous Demucs, we always normalize because it is easier.
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        std = x.std(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) / (1e-5 + std)
        # x will be the freq. branch input.

        # Prepare the time branch input.
        xt = wav
        meant = xt.mean(dim=(1, 2), keepdim=True)
        stdt = xt.std(dim=(1, 2), keepdim=True)
        xt = (xt - meant) / (1e-5 + stdt)

        # Frequence and waveform encoder (with storage for skip connections) 
        saved = []  # skip connections, freq.
        saved_t = []  # skip connections, time.
        lengths = []  # saved lengths to properly remove padding, freq branch.
        lengths_t = []  # saved lengths for time branch.
        
        for idx, encode in enumerate(self.htdemucs.encoder):
            lengths.append(x.shape[-1])
            inject = None
            if idx < len(self.htdemucs.tencoder):
                # we have not yet merged branches.
                lengths_t.append(xt.shape[-1])
                tenc = self.htdemucs.tencoder[idx]
                xt = tenc(xt)
                if not tenc.empty:
                    # save for skip connection
                    saved_t.append(xt)
                else:
                    # tenc contains just the first conv., so that now time and freq.
                    # branches have the same shape and can be merged.
                    inject = xt
            x = encode(x, inject)
            if idx == 0 and self.htdemucs.freq_emb is not None:
                # add frequency embedding to allow for non equivariant convolutions
                # over the frequency axis.
                frs = torch.arange(x.shape[-2], device=x.device)
                emb = self.htdemucs.freq_emb(frs).t()[None, :, :, None].expand_as(x)
                x = x + self.htdemucs.freq_emb_scale * emb

            saved.append(x)
            
        if self.htdemucs.crosstransformer:
            if self.htdemucs.bottom_channels:
                b, c, f, t = x.shape
                x = rearrange(x, "b c f t-> b c (f t)")
                x = self.htdemucs.channel_upsampler(x)
                x = rearrange(x, "b c (f t)-> b c f t", f=f)
                xt = self.htdemucs.channel_upsampler_t(xt)

            x, xt = self.htdemucs.crosstransformer(x, xt)

            if self.htdemucs.bottom_channels:
                x = rearrange(x, "b c f t-> b c (f t)")
                x = self.htdemucs.channel_downsampler(x)
                x = rearrange(x, "b c (f t)-> b c f t", f=f)
                xt = self.htdemucs.channel_downsampler_t(xt)
        
        return x, xt

    # ------------------------------------------------------------------
    # Forward: waveform + text → conditioned separated stems
    # ------------------------------------------------------------------
    def forward(self, wav, text):
        """
        wav  : (B, C, T)
        text : List[str]
        Returns: (B, 1, C, T)    # separated 1-track (4 channels) GOAL
        """
        # NOTE: Copied from htdemucs forward function, then modified
        x, xt = self._htdemucs_full_enc(wav)
                    
        print(f"x shape: {x.shape}")
        print(f"xt shape: {xt.shape}")
        
        return torch.rand((1, 2, 44100))


# Example usage template:
if __name__ == "__main__":
    # Load pre-trained CLAP text encoder
    import torch
    from demucs import pretrained

    # Load pre-trained HTDemucs model
    htdemucs = pretrained.get_model('htdemucs').models[0]

    # Load CLAP model
    clap = ClapModel.from_pretrained("laion/clap-htsat-unfused")
    tokenizer = AutoTokenizer.from_pretrained("laion/clap-htsat-unfused")

    model = AudioTextHTDemucs(
        htdemucs,
        clap,
        tokenizer  
    )
    #print(f"Full Model:\n{model}")
    #model = model.to('cuda')

    B = 1
    C = 2
    T = 44100 * 3  # 3 seconds @ 16k
    waveform = torch.randn(B, C, T)
    prompts = ["extract viola", "extract drums"]
    out_waveform = model(waveform, prompts)
    print(f"output waveform shape: {out_waveform.shape}")