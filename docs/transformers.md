# Understanding Q, K, V and Temporal Attention in Transformers

## From CNN Features to Attention

Suppose a CNN produces a feature map:

```math
(15, 15, 512)
```

This means:
- 15 × 15 spatial locations
- each location represented by a 512-dimensional feature vector

Flattening spatially gives:

```math
225 \text{ tokens} \times 512 \text{ features}
```

Each token now represents a spatial patch of the image.

---

# Why Transformers Use Attention

A transformer allows every token to dynamically interact with every other token.

Instead of fixed convolutions, each token can ask:

> “Which other tokens are relevant to me?”

This creates a flexible communication system between image regions, words, or time steps.

---

# Q, K, V: The Three Projections

From the same input tensor `X`, the transformer creates three different learned projections:

```math
Q = XW_Q
```

```math
K = XW_K
```

```math
V = XW_V
```

These are linear projections with different learned weights.

Even though they come from the same source tensor, they learn different roles.

---

# Intuition Behind Q, K, V

## Query (Q)

The Query represents:

> “What information am I looking for?”

Queries learn how to ask useful questions.

---

## Key (K)

The Key represents:

> “What kind of information do I contain?”

Keys act like semantic addresses or implicit identifiers.

They help other tokens determine whether this token is relevant.

---

## Value (V)

The Value represents:

> “What information do I actually provide?”

Values are the actual payload transferred between tokens.

Q and K decide who communicates.

V is the information being communicated.

---

# Attention Mechanism

The transformer compares Queries against Keys:

```math
QK^T
```

This produces similarity scores.

Then scaled dot-product attention computes:

```math
\mathrm{Attention}(Q,K,V)
=
\mathrm{softmax}\left(
\frac{QK^T}{\sqrt{d}}
\right)V
```

The process is:

1. Query searches for relevant Keys
2. Similarity scores are computed
3. Softmax converts scores into weights
4. Weighted combination of Values produces the output

---

# Important Insight

Q and K are mainly used for:

- matching
- routing
- retrieval

V is used for:

- transferring semantic information
- carrying features forward

This separation allows the transformer to learn:
- how to search
- how to advertise information
- what information to share

all independently.

---

# Attention as Communication

A powerful way to think about transformers:

## Attention is differentiable communication.

Each token:
- asks questions using Queries
- advertises itself using Keys
- sends information using Values

The model learns its own internal communication protocol.

---

# Multi-Head Attention

Transformers usually use multiple attention heads.

Different heads specialize in different relationships.

For example:
- one head may focus on edges
- another on object identity
- another on spatial relationships
- another on motion continuity

This allows multiple semantic interactions simultaneously.

---

# Adding Temporality

Now suppose we process video instead of a single image.

If:
- each frame has 225 spatial tokens
- there are K frames

Then total tokens become:

```math
225 \times K
```

Now attention operates across:
- space
- time

A token can attend:
- spatially within the same frame
- temporally across previous frames

---

# Temporal Attention

Example:

A token representing a cat paw in frame `t`
may attend to:
- the same paw in frame `t-1`
- nearby motion regions
- object boundaries from earlier frames

This allows the transformer to learn:
- motion
- continuity
- tracking
- temporal relationships

without explicit hand-designed motion rules.

---

# Why Scaling Exists

Dot products grow larger with higher dimensions.

Without scaling:

```math
QK^T
```

can become unstable.

So transformers divide by:

```math
\sqrt{d}
```

This stabilizes gradients and softmax behavior.

Hence the name:

## Scaled Dot-Product Attention (SDPA)

---

# KV Cache in Autoregressive Generation

During generation, transformers produce one token at a time.

At timestep `t`:
- previous Keys and Values are already known
- only the new Query must search them

So models cache:
- K tensors
- V tensors

This is called the KV Cache.

---

# Why Queries Are Not Cached

Queries are temporary.

A Query only exists to ask a question at the current timestep.

Once attention is computed:
- old Queries are no longer needed

But old:
- Keys remain searchable
- Values remain retrievable

So future tokens still need them.

---

# Deep Intuition

Transformers are not simply “looking things up.”

They are dynamically routing learned semantic information between tokens.

Over many layers:
- low-level features become objects
- objects become relationships
- relationships become abstractions

Attention creates an emergent communication system inside the network.

This is one reason transformers scale so effectively across:
- language
- vision
- video
- multimodal learning
- reasoning tasks