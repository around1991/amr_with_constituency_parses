# amr_with_constituency_parses
Codebase for https://aclweb.org/anthology/papers/N/N19/N19-1223/

This has all the code necessary to run the model from the NAACL 2019 paper 'Factorising AMR generation through syntax'.

The primary entry point is ``model.py``. This contains the main model. Model implementation is all in ``seq2seq.py``. This also contains a very nice optimized batched beam search in PyTorch which may be helpful for others. 
