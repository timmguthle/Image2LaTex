
# Image2LaTex

This is an encoder-decoder based neural network to convert images of mathematical expressions into the corresponding LaTeX code. Because this system acts as an inverse of the LaTeX compiler, one could call it a decompiler.

The implementation is based on [What You Get Is What You See: A Visual Markup Decompiler](http://arxiv.org/pdf/1609.04938v1.pdf) by HarvardNLP.

## Sample Results
An example from the validation set:


![Example](/data/example.png)

Examples of the attention mechanism can be found in the `test_mk3.ipynb` notebook.
