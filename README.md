# Text Clustering

This repository contains tools to easily embed and cluster texts as well as label clusters and produce visualizations of those labeled clusters. 

## Install 

Install the library to get started:

```bash
pip install --upgrade bocluster
```

## Usage

The pipeline can be used following the code block below.

```python
from datasets import load_dataset
from bocluster.cluster import BoClusterClassifier

# load a Tibetan language text dataset
ds = load_dataset('billingsmoore/LotsawaHouse-bo-en', split='train')

# initilialize a BoClusterClassifier object
bcc = BoClusterClassifier()

# fit the classifier on a set of texts
bcc.fit(ds['bo'][:1000])

# if you want to treat all data points as members of clusters, with no data treated as outliers
bcc.classify_outliers()

# show a visualization of results
bcc.show()
```