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
- Download complex embeddings from http://corpora.ficlit.unibo.it/UploadDIR/GitHub/TestX9c.bin.
- Classify all the instances taken from the standard benchmark created by [Raganato et al. 2017].
```
    python3 QWSD.py TestX9c.bin Evaluation_Datasets/ALL/ALL.data.xml > Test
```
- Evaluate the results:
```
    bash ./ScoreAll.sh Test
```

If you find this code useful in your research, please cite:
```
@inproceedings{Tamburini:2019:RANLP,
  author    = {Fabio Tamburini},
  editor    = {Ruslan Mitkov and
               Galia Angelova},
  title     = {A Quantum-Like Approach to Word Sense Disambiguation},
  booktitle = {Proceedings of the International Conference on Recent Advances in
               Natural Language Processing, {RANLP} 2019, Varna, Bulgaria, September
               2-4, 2019},
  pages     = {1176--1185},
  publisher = {{INCOMA} Ltd.},
  year      = {2019},
  url       = {https://doi.org/10.26615/978-954-452-056-4\_135},
  doi       = {10.26615/978-954-452-056-4\_135},
}
```

In case of problems contact me at <fabio.tamburini@unibo.it>.
