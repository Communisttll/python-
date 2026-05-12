<!DOCTYPE html>
<html lang="zh">
<head>
    <meta charset="UTF-8">
    <title>Transformer (machine learning model) - Wikipedia</title>
</head>
<body>
<h1>Transformer (machine learning model)</h1>

<p>In machine learning, the <strong>transformer</strong> is a deep learning architecture
introduced in 2017 by researchers at Google and University of Toronto. Transformers
are designed to handle sequential data, such as natural language, for tasks like
translation and text summarization. Unlike recurrent neural networks (RNNs),
transformers process the entire input sequence at once, enabling parallel computation
and capturing long-range dependencies more effectively.</p>

<h2>Architecture</h2>

<p>The transformer architecture consists of an encoder and a decoder. The encoder
processes the input sequence and creates a representation, while the decoder generates
the output sequence. Key components include:</p>

<ul>
<li><strong>Self-attention mechanism</strong>: Allows each position to attend to all positions
in the input sequence, computing attention weights based on relevance.</li>
<li><strong>Feed-forward neural networks</strong>: Applied point-wise to each position.</li>
<li><strong>Positional encoding</strong>: Adds information about token positions since
transformers have no inherent notion of order.</li>
<li><strong>Multi-head attention</strong>: Runs attention in parallel, allowing the model
to jointly attend to information from different representation subspaces.</li>
</ul>

<h2>History</h2>

<p>The transformer architecture was introduced in the 2017 paper "Attention Is All You Need"
by Ashish Vaswani et al. The paper was presented at the NeurIPS conference. This work
built upon earlier concepts of attention mechanisms and became the foundation for
subsequent models like BERT (2018) and GPT (2018).</p>

<p>In 2020, the introduction of GPT-3 demonstrated that large-scale transformers could
achieve remarkable few-shot learning capabilities. The year 2023 saw the release of
GPT-4, further advancing the capabilities of transformer-based models.</p>

<h2>Applications</h2>

<p>Transformers have been applied to a wide variety of tasks:</p>

<ul>
<li><strong>Natural Language Processing</strong>: Machine translation, text summarization,
question answering, sentiment analysis</li>
<li><strong>Computer Vision</strong>: Vision Transformers (ViT), DETR, image classification</li>
<li><strong>Multimodal Models</strong>: CLIP, DALL-E, GPT-4V, Gemini</li>
<li><strong>Audio Processing</strong>: Speech recognition, music generation</li>
<li><strong>Scientific Research</strong>: Protein structure prediction (AlphaFold),
drug discovery</li>
</ul>

<h2>Variants</h2>

<p>Since the original transformer, numerous variants have been developed:</p>

<ul>
<li><strong>Encoder-only</strong>: BERT, RoBERTa - used for understanding tasks</li>
<li><strong>Decoder-only</strong>: GPT series - used for generation tasks</li>
<li><strong>Encoder-decoder</strong>: T5, BART - used for seq2seq tasks</li>
<li><strong>Efficient Transformers</strong>: Longformer, Reformer, BigBird -
designed for processing longer sequences</li>
</ul>

<h2>See Also</h2>
<ul>
<li>Attention mechanism</li>
<li>BERT (language model)</li>
<li>GPT (language model)</li>
<li>Deep learning</li>
</ul>

</body>
</html>
