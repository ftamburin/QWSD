# QWSD
Code and data for the paper "A Quantum-like approach to Word Sense Disambiguation" at
[RANLP2019](http://lml.bas.bg/ranlp2019/).

Tested using:

- Python 3.6.3
- Numpy 1.15.4
- Scipy 0.19.1
- NLTK 3.2.5
- lxml 4.3.0

For reproducing our results:

- Clone the repository.
- Install the [NLTK WordNet corpus package](http://www.nltk.org/howto/wordnet.html).
- Download complex embeddings from http://corpora.ficlit.unibo.it/UploadDIR/TestX9c.bin.
- Classify all the instances taken from the standard benchmark created by [Raganato et al. 2017].
```
    python3 QWSD.py TestX9c.bin Evaluation_Datasets/ALL/ALL.data.xml > Test
```
- Evaluate the results:
```
    bash ./ScoreAll.sh Test
```
In case of problems contact me at <fabio.tamburini@unibo.it>.
