# Project Title

Ontology-based Recommendation System for Few-Chapter Books

## Getting Started

Basically clone this whole project using Git terminal command


### Package Prerequisites
Please make sure these packages have already been installed before you run the code. Otherwise, you will get compiling errors
```
_ Python 3 
_ NLTK framework. If you do not have NLTK, install them on your machine 
_ matplotlib package
_ numpy package

```

### File Prerequisites
```
[long_book.json] : This file contains information of the many-book chapters (i.e. title, abstract and keywords). The metadata is stored in JSON format
[short_book.json] : This file contains information of the few-book chapters (i.e. title, abstract and keywords). The metadata is stored in JSON format
[iswc_topic]: Text file includes all of the "Topic of interest" pulled from ISWC 2017 website 
[ontology]: This folder contains Computer Science Ontology that is used to extract topics from input strings (publications/ conferences). Make sure that ComputerScienceOntology_v2.csv is included in this folder
```
## Running the tests

There are 2 tests that you can run from these scripts: Extended Few-Chapter Book test vs Few-Chapter Book Test
# Few-Chapter Book Test:
This test basically computes different similarity measures (Cosine Similarity, Jaccard Similarity and Semantic Ranked Recommendation Similarity). There are two ways to run this test by passing arguments

1.  This test is run in a cumulative fashion. This means that newly considered chapter is appended to the previously considered chapters
```
python3 few_chapter_similarity.py cumulative

```     
2. The second test is run in a separate fashion. This means that newly considered chapter is treated as its own.
to the previously considered chapters
```
python3 few_chapter_similarity.py separate

```     

# Extended Few-Chapter Book Test:
This test  computes different similarity measures (Cosine Similarity, Jaccard Similarity and Semantic Ranked Recommendation Similarity) when we build more expanded knowledge base by extracting topics from many-chapter book 

1.  This test is run in a cumulative fashion. 
```
python3 extended_few_chapter_similarity.py cumulative

```     
2. The second test is run in a separate fashion. This means that newly considered chapter is treated as its own.
to the previously considered chapters
```
python3 extended_few_chapter_similarity.py separate

```     

## Author

Hoang Nguyen - *Initial work* 

## Acknowledgments

* [cso-classifier]: Inspired from here https://github.com/angelosalatino/cso-classifier


