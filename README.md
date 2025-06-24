## Sentiment analysis in low-resource languages such as Assamese poses significant challenges due to the
## unavailability of large, annotated datasets and language processing tools. 
The project focuses on leveraging advanced deep learning methods, particularly Conditional Generative Adversarial Networks
(cGAN), to accurately classify sentiment. The model is designed to enhance the understanding and
generation of sentiment-aware text, thereby supporting the broader goal of advancing regional language
processing in NLP. The main motive of this project is to build an efficient sentiment analysis system for
Assamese, a low-resource and less-explored language in the domain of Natural Language Processing
(NLP). It aims to assess how well generative models can learn complex sentiment cues in Assamese text
and to create a performance benchmark for future studies. A custom dataset of a total of 4800 was prepared,
consisting of Assamese text samples labelled as either positive or negative, which formed the basis for
training and evaluation. The cGAN generates sentiment-conditioned synthetic samples to aid
classification, improving performance in data-scarce conditions, achieving an Accuracy of 86%, and Precision
86%, Recall 86%, and F1-score 86%. The current implementation uses a basic cGAN in PyTorch,
demonstrating feasibility but not yet incorporating advanced methods like pretrained embeddings.
Gumbel-Softmax, or transformers, highlighting these as future enhancement areas
