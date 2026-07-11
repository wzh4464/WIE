"""W6: peak GPU memory of a full-parameter BERT-base HVP (double backward),
with gradient checkpointing, to substantiate 'full 110M params, no last-layer
restriction'. Measures torch.cuda.max_memory_allocated for one HVP."""
import torch, sys
from transformers import AutoModelForSequenceClassification

dev = torch.device("cuda")
torch.manual_seed(0)
m = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2).to(dev)
try:
    m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
except TypeError:
    import torch.utils.checkpoint as _ckpt
    _orig = _ckpt.checkpoint
    _ckpt.checkpoint = lambda *a, **k: _orig(*a, **{**k, "use_reentrant": False})
    m.gradient_checkpointing_enable()  # bound activation memory
m.train()
# gradient checkpointing + double backward requires non-reentrant
try:
    m.config.use_cache = False
except Exception:
    pass

P = sum(p.numel() for p in m.parameters())
params = [p for p in m.parameters() if p.requires_grad]

B, L = int(sys.argv[1]) if len(sys.argv) > 1 else 8, int(sys.argv[2]) if len(sys.argv) > 2 else 256
ids = torch.randint(0, 30000, (B, L), device=dev)
mask = torch.ones(B, L, device=dev, dtype=torch.long)
y = torch.randint(0, 2, (B,), device=dev)
lossf = torch.nn.CrossEntropyLoss()

torch.cuda.reset_peak_memory_stats()
out = m(input_ids=ids, attention_mask=mask).logits
loss = lossf(out, y)
g = torch.autograd.grad(loss, params, create_graph=True)
# HVP with a random vector
v = [torch.randn_like(p) for p in params]
dot = sum((gi * vi).sum() for gi, vi in zip(g, v))
hv = torch.autograd.grad(dot, params, retain_graph=False)
torch.cuda.synchronize()

peak = torch.cuda.max_memory_allocated() / 1e9
reserved = torch.cuda.max_memory_reserved() / 1e9
print(f"params={P/1e6:.1f}M  batch={B} seqlen={L}  grad_checkpoint=ON")
print(f"PEAK_ALLOCATED_GB={peak:.2f}  PEAK_RESERVED_GB={reserved:.2f}")
print(f"hvp_norm={sum((h*h).sum() for h in hv).sqrt().item():.3f}")
