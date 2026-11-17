# ES 335 Project Proposal (submitted, revision 14)

**Title:** Investigating Syntax Learning and Generalization in MLP-Based Python Code Generation
**Members:** Buha Deep Maheshbhai (24110082), Panchal Heet Chirag (24110235)
**Expected tools:** Python, NumPy, pandas, scikit-learn, PyTorch, Jupyter Notebook, Matplotlib, Git, GitHub, HTML, CSS, JavaScript

## Research question
How does context length affect the next-token prediction performance, syntactic validity, and generalization of a fixed-context MLP trained on Python source code? How does its performance compare with an n-gram baseline, particularly when evaluated on repositories excluded from training?

## Problem and intended claim
Predicting the next token in source code is a self-supervised learning task that allows a model to learn patterns directly from existing programs. However, good next-token prediction does not necessarily imply that generated code is syntactically valid or generalizes to unfamiliar projects.

This project will train a fixed-context multilayer perceptron (MLP) to predict the next token in Python code. It will investigate how context length affects prediction quality and generated-code validity, and whether the MLP provides an advantage over a simpler n-gram model. A useful outcome will be an evidence-based account of which patterns the model learns, how well it generalizes to unseen repositories, and where its limitations remain.

## Data plan
The primary data source will be the publicly accessible CodeSearchNet Python dataset, obtained through Hugging Face Datasets. It contains Python code examples collected from public repositories. The dataset will be inspected before selecting a subset of roughly 100,000 functions suitable for training and evaluation.

Each original observation represents a code example, such as a function or code snippet. The main variables will be the source code, the repository name, and other metadata useful for filtering and splitting the data. Source code will be tokenized with Python's tokenize module, keeping the 10,000 most frequent tokens plus an <UNK> token, and the resulting sequences will be used for next-token prediction.

I will inspect code lengths, token frequencies, missing or empty examples, duplicates, and repository distribution. Training, validation, and test sets will be separated by repository to reduce contamination from similar code appearing across splits. Dataset size and final filtering decisions will be documented after inspection.

## Baseline and candidate methods
The baseline will be a smoothed n-gram language model that predicts the next token from a limited history of preceding tokens. The candidate model will be a fixed-context MLP with an embedding layer, a hidden layer with a nonlinear activation, and an output layer over the vocabulary. It will be trained using cross-entropy loss and gradient-based optimization.

Experiments will compare MLP context lengths of 3, 5, 10 and 20 tokens against n-gram orders 2 to 5, controlling other important settings as far as practical. Both models will share the same vocabulary and unknown-token handling so that their perplexities are comparable. The n-gram baseline is appropriate because it captures local token patterns without a neural network. The MLP provides a way to investigate whether learned embeddings and nonlinear transformations improve prediction beyond simple frequency-based patterns. Model complexity and training cost will be considered when interpreting improvements.

## Evaluation plan
The dataset will be divided into training, validation, and test sets by repository, using the repository name in the dataset metadata. Training data will be used to fit the models, validation data to select context length and hyperparameters, and the held-out test set only for final evaluation.

Next-token prediction will be evaluated using cross-entropy loss and perplexity. Generated code will also be assessed using the proportion of sampled completions that pass Python syntax parsing. Completions will be generated until the end of the function or a maximum token length, with the same procedure and stopping conditions used for all models. Syntax validity will be reported only for completions that reach a natural stopping point, and truncated samples will be counted and reported separately. Representative failures will be inspected and categorized.

The comparison will include both ordinary held-out evaluation and repository-separated evaluation if feasible. Vocabulary construction, learned preprocessing, and model selection will use training and validation data only; the test set will not be used for tuning. Results will be reported with the relevant split details and limitations.

## Likely challenge
A major challenge is data leakage caused by duplicated or highly similar code snippets from the same repository appearing in both training and testing. Where repository metadata is available, I will use repository-separated splits and inspect duplicates. Another challenge is that low next-token loss does not guarantee valid generated programs. I will therefore evaluate syntax validity separately and avoid treating syntactic correctness as proof that the generated code is functionally correct.

## Track-specific plan (ES 335)
The project website will include an interactive code-completion explorer where visitors can enter a Python prefix and inspect generated continuations from saved model outputs. Visitors will be able to compare selected context lengths and the MLP against the n-gram baseline. Additional interactive plots will show changes in prediction loss and syntax-validity rates across experiments. A failure inspector will present representative invalid or poorly generalized completions and explain the limitations revealed by the evaluation. The website will be published on GitHub Pages, with the source code, experiment settings, and reproduction instructions provided in the repository.
