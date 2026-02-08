# Minimal Free Form Flows

Grokking this approach to kinda think through other ways to utilise the volume change estimate.

Encoder-decoder generative model. No invertibility constraint — any architecture works.

Volume change estimated via Hutchinson surrogate (mixed fwd/bwd AD), so training is O(d) not O(d^3).

1-NFE sampling: `z ~ N(0,I), x = decoder(z)`.

```
python fff.py
```

Paper: Draxler et al., "Free-form flows: Make Any Architecture a Normalizing Flow"
