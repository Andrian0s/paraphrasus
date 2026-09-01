# Benchmarking library and reproduction code for [PARAPHRASUS: A Comprehensive Benchmark for Evaluating Paraphrase Detection Models](https://aclanthology.org/2025.coling-main.585/)
<img height="48" alt="COLING2025 Abu Dhabi" src="https://github.com/user-attachments/assets/d84f99e0-ac15-443c-b9ef-a196b64e1da7"/> ![License: AGPLV3+](https://img.shields.io/badge/License-AGPLV3+-brightgreen.svg)

This repository contains the code and datasets for benchmarking a paraphrase detector, as described in our COLING 2025 paper *PARAPHRASUS: A Comprehensive Benchmark for Evaluating Paraphrase Detection Models*. It also has the scripts that allow reproduction and extension of the results that are displayed in the paper.

It additionally contains an extension for benchmarking pre-trained embedding models with supervised calibration, as described in *Efficient Paraphrase Detection With Embeddings* ([ACL Anthology](TODO-ANTHOLOGY-LINK)). See [Embedding models](#embedding-models).

## Quick start

To evaluate a model on the full PARAPHRASUS benchmark, you simply need to wrap it into a binary prediction method that accepts a list with pairs of texts, and returns a list with boolean True/False predictions.

For example, here are two dummy prediction methods, `predict_method1` and `predict_method2`. Then:

```python
from benchmarking import bench

def predict_method1(pairs):
    return [False for _ in pairs]

def predict_method2(pairs):
    return [False for _ in pairs]
methods = {
        "m1": predict_method1,
        "m2": predict_method2
    }

bench(methods, bench_id="mybench")
```

## Configuration
Running by configuration is also supported. 
Assuming the dummy prediction functions above are located at the file my_funcs.py, a configuration file should look like this:
```json
{
  "bench_id": "mybench",
  "methods": [
    {
      "name": "m1",
      "module": "my_funcs",
      "function": "predict_method1"
    },
    {
      "name": "m2",
      "module": "my_funcs",
      "function": "predict_method2"
    }
  ]
}
```
Then, assuming the above configuration is saved as configs/my_config.json, one can run the benchmark like so:
```bash
python3 benchmarking.py configs/my_config.json
```

Finally, the results can be extracted by running:
```bash
python3 extract_results.py configs/my_config.json
```

which will save the error rates at: benches/mybench/results.json

## Table of Contents

- [Overview](#overview)
- [Repository Organization](#repository-organization)
- [Reproducing the Experiments](#reproducing-the-experiments)
- [Embedding models](#embedding-models)
- [Further Experimentation](#further-experimentation)
- [BibTeX Reference](#bibtex-reference)
- [Datasets and Licenses](#datasets-and-licenses)
- [Further Support](#further-support)
- [About Impresso](#about-impresso)

## Overview

This repository allows replication of the experiments from the research titled "PARAPHRASUS: A Comprehensive Benchmark for Evaluating Paraphrase Detection Models" and is extendable to allow further experimentation, qualitative analysis. It includes:

- Original predictions generated using the models described in the paper, useful for further qualitative or quantitative analysis.
- Scripts and configuration files to reproduce the results.
- Utility scripts to reproduce statistics plots and so on.
- An extension for benchmarking pre-trained embedding models, described in [Embedding models](#embedding-models).

## Repository Organization:

The repository is organized as follows:

```
├── original_reproduction_code
│   └── The initial version of the repository.
├── datasets_no_results
│   └── Contains the datasets used in the experiments, in a JSON format. Copied for every new benchmark
├── models
│   └── Empty models file used in the experiments.
├── configs
│   └──paper_config.json
│      └──Benchmark configuration for the methods used in the paper.
│   └──llama3_3_70b_config.json
│      └──Benchmark configuration for running the benchmark using Llama3.3 70b Q8 and 8b Q4
│   └──ablation_a_config.json, ablation_b_config.json
│      └──Benchmark configurations for the ablations in the paper.
│   └──full_embed_config.json
│      └──Benchmark configuration for the embedding models. See Embedding models.
├── benchmarking.py
│   └── Main benchmarking code, for running predictions on the datasets using specified methods.
├── extract_results.py
│   └── Script for extracting result of a benchmark.
├── lm_studio_templates
|   └──templates.py
|      └──Sample functions for making prediction functions using LM Studio
|   └──paper_methods.py
|      └──methods used in the paper to run benchmarks using LM Studio
|   └──l70b_methods.py
|      └──methods to run the benchmark using Llama3.3 70b Q8 and 8b Q4
├── logger.py
│   └── Utility for managing logging: all events are logged both to stdout and to a local logs.log file.
├── benchmarking_embed.py
│   └── Benchmarking code for embedding models. Counterpart to benchmarking.py.
├── embedders.py, embedding_store.py, similarity.py, scorers.py, calibrate_embeddings_clf.py
│   └── Supporting modules for the embedding benchmark. See README_embeddings.md.
├── labels.py
│   └── Group definitions and ground truth labels, shared by extract_results.py and the embedding benchmark.
├── README_embeddings.md
│   └── Full documentation for the embedding extension.
```

## Reproducing the Experiments

Predictions using the LLMs in the paper can be run locally (provided LM Studio is running and serving the model meta-llama-3-8b-instruct (Meta-Llama-3-8B-Instruct-Q4_K_M.gguf)) like so:
```bash
python3 benchmarking.py configs/paper_config.json
```

The predictions of the methods mentioned in the paper are given as a benchmark with the identifier 'paper'.
That means, the results (error rates) can be extracted like so:
```bash
python3 extract_results.py configs/paper_config.json
```
At benches/paper the file results.json is generated:
```json
{
    "Classify!": {
        "PAWSX": {
            "XLM-RoBERTa-EN-ORIG": "15.2%",
            "Llama3 zero-shot P1": "44.7%",
            "Llama3 zero-shot P2": "40.7%",
            "Llama3 zero-shot P3": "38.1%",
            "Llama3 ICL_4 P1": "39.0%",
            "Llama3 ICL_4 P2": "34.1%",
            "Llama3 ICL_4 P3": "33.2%"
        },
        "STS-H": {
            "XLM-RoBERTa-EN-ORIG": "54.1%",
            "Llama3 zero-shot P1": "56.2%",
            "Llama3 zero-shot P2": "37.6%",
            "Llama3 zero-shot P3": "41.7%",
            "Llama3 ICL_4 P1": "44.7%",
            "Llama3 ICL_4 P2": "41.7%",
            "Llama3 ICL_4 P3": "39.1%"
        },
        "MRPC": {
            "XLM-RoBERTa-EN-ORIG": "33.4%",
            "Llama3 zero-shot P1": "23.6%",
            "Llama3 zero-shot P2": "45.9%",
            "Llama3 zero-shot P3": "37.5%",
            "Llama3 ICL_4 P1": "33.2%",
            "Llama3 ICL_4 P2": "45.2%",
            "Llama3 ICL_4 P3": "46.7%"
        }
    },
    "Minimize!": {
        "SNLI": {
            "XLM-RoBERTa-EN-ORIG": "32.4%",
            "Llama3 zero-shot P1": "7.3%",
            "Llama3 zero-shot P2": "1.0%",
            "Llama3 zero-shot P3": "1.3%",
            "Llama3 ICL_4 P1": "1.9%",
            "Llama3 ICL_4 P2": "0.8%",
            "Llama3 ICL_4 P3": "0.5%"
        },
        "ANLI": {
            "XLM-RoBERTa-EN-ORIG": "7.2%",
            "Llama3 zero-shot P1": "13.0%",
            "Llama3 zero-shot P2": "1.2%",
            "Llama3 zero-shot P3": "1.7%",
            "Llama3 ICL_4 P1": "2.0%",
            "Llama3 ICL_4 P2": "0.8%",
            "Llama3 ICL_4 P3": "0.8%"
        },
        "XNLI": {
            "XLM-RoBERTa-EN-ORIG": "26.7%",
            "Llama3 zero-shot P1": "12.3%",
            "Llama3 zero-shot P2": "1.4%",
            "Llama3 zero-shot P3": "1.3%",
            "Llama3 ICL_4 P1": "2.8%",
            "Llama3 ICL_4 P2": "0.3%",
            "Llama3 ICL_4 P3": "0.3%"
        },
        "STS": {
            "XLM-RoBERTa-EN-ORIG": "46.6%",
            "Llama3 zero-shot P1": "12.9%",
            "Llama3 zero-shot P2": "2.4%",
            "Llama3 zero-shot P3": "3.5%",
            "Llama3 ICL_4 P1": "3.5%",
            "Llama3 ICL_4 P2": "3.1%",
            "Llama3 ICL_4 P3": "2.4%"
        },
        "SICK": {
            "XLM-RoBERTa-EN-ORIG": "37.0%",
            "Llama3 zero-shot P1": "0.9%",
            "Llama3 zero-shot P2": "0.1%",
            "Llama3 zero-shot P3": "0.0%",
            "Llama3 ICL_4 P1": "0.3%",
            "Llama3 ICL_4 P2": "0.0%",
            "Llama3 ICL_4 P3": "0.0%"
        }
    },
    "Maximize!": {
        "TRUE": {
            "XLM-RoBERTa-EN-ORIG": "31.4%",
            "Llama3 zero-shot P1": "9.0%",
            "Llama3 zero-shot P2": "34.7%",
            "Llama3 zero-shot P3": "35.3%",
            "Llama3 ICL_4 P1": "29.9%",
            "Llama3 ICL_4 P2": "40.1%",
            "Llama3 ICL_4 P3": "50.9%"
        },
        "SIMP": {
            "XLM-RoBERTa-EN-ORIG": "5.3%",
            "Llama3 zero-shot P1": "14.7%",
            "Llama3 zero-shot P2": "47.3%",
            "Llama3 zero-shot P3": "37.5%",
            "Llama3 ICL_4 P1": "33.3%",
            "Llama3 ICL_4 P2": "42.3%",
            "Llama3 ICL_4 P3": "45.5%"
        }
    },
    "Averages": {
        "XLM-RoBERTa-EN-ORIG": {
            "Classify!": "34.2%",
            "Minimize!": "30.0%",
            "Maximize!": "18.3%",
            "Overall Average": "27.5%"
        },
        "Llama3 zero-shot P1": {
            "Classify!": "41.5%",
            "Minimize!": "9.3%",
            "Maximize!": "11.8%",
            "Overall Average": "20.9%"
        },
        "Llama3 zero-shot P2": {
            "Classify!": "41.4%",
            "Minimize!": "1.2%",
            "Maximize!": "41.0%",
            "Overall Average": "27.9%"
        },
        "Llama3 zero-shot P3": {
            "Classify!": "39.1%",
            "Minimize!": "1.6%",
            "Maximize!": "36.4%",
            "Overall Average": "25.7%"
        },
        "Llama3 ICL_4 P1": {
            "Classify!": "39.0%",
            "Minimize!": "2.1%",
            "Maximize!": "31.6%",
            "Overall Average": "24.2%"
        },
        "Llama3 ICL_4 P2": {
            "Classify!": "40.3%",
            "Minimize!": "1.0%",
            "Maximize!": "41.2%",
            "Overall Average": "27.5%"
        },
        "Llama3 ICL_4 P3": {
            "Classify!": "39.7%",
            "Minimize!": "0.8%",
            "Maximize!": "48.2%",
            "Overall Average": "29.6%"
        }
    }
}
```

## Embedding models

Pre-trained embedding models can be benchmarked with a separate entry point. Rather than
asking a model to judge each pair, it embeds every text once and turns the resulting
vectors into decisions, which is what makes the approach cheap at scale. This is the
methodology of *Efficient Paraphrase Detection With Embeddings*.

An embedder is any function that takes a list of texts and returns one vector per text,
which is what `SentenceTransformer.encode` already does:

```python
from benchmarking_embed import bench_embed
from scorers import get_scorer
from sentence_transformers import SentenceTransformer

embedders = {"M-MiniLM": SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2").encode}
scorers = {"thr": get_scorer({"method": "threshold"})}

bench_embed(embedders, scorers, bench_id="myembedbench")
```

Two calibration strategies are provided: a single threshold on cosine similarity, and a
logistic regression on the unsigned element-wise difference between the two embeddings.
Both are fitted leave-one-dataset-out, so predictions for a group come from a fit that
never saw that group.

Running by configuration works the same way as for the main benchmark:

```bash
pip install -r requirements.txt
python3 benchmarking_embed.py configs/full_embed_config.json
python3 extract_results.py configs/full_embed_config.json
```

`configs/full_embed_config.json` covers the nine embedding models of the paper, the
logistic regression feature ablation, and the instruction-prompt variants: 12 embedders
by 5 scorers, written into one benchmark at `benches/full_embed_results`. Results appear
in the same `results.json` format as the main benchmark, so embedding methods and the
LLM baselines can be compared in one table.

Embeddings are cached to disk and both stages resume, so an interrupted run picks up
where it stopped.

Full documentation, including the calibration protocol, the supported scorers, and how
to reproduce the published numbers exactly, is in
[README_embeddings.md](README_embeddings.md).

## Further Experimentation

You can run your own experiments using any prediction methods of your choosing.

## BibTeX Reference

If you would like to cite this project, or the associated papers, here are the bibtex entries.

For the benchmark:

```bibtex
@inproceedings{michail-etal-2025-paraphrasus,
    title = "{PARAPHRASUS}: A Comprehensive Benchmark for Evaluating Paraphrase Detection Models",
    author = "Michail, Andrianos  and
      Clematide, Simon  and
      Opitz, Juri",
    editor = "Rambow, Owen  and
      Wanner, Leo  and
      Apidianaki, Marianna  and
      Al-Khalifa, Hend  and
      Eugenio, Barbara Di  and
      Schockaert, Steven",
    booktitle = "Proceedings of the 31st International Conference on Computational Linguistics",
    month = jan,
    year = "2025",
    address = "Abu Dhabi, UAE",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.coling-main.585/",
    pages = "8749--8762"
}
```

For the embedding extension:

```bibtex
@inproceedings{rocci-etal-2026-efficient,
    title = "Efficient Paraphrase Detection With Embeddings",
    author = "Rocci, Giovanni  and
      Michail, Andrianos  and
      Loizides, Andreas  and
      Clematide, Simon  and
      Opitz, Juri",
    booktitle = "TODO",
    year = "TODO",
    address = "TODO",
    publisher = "Association for Computational Linguistics",
    url = "TODO-ANTHOLOGY-LINK",
    pages = "TODO"
}
```

## Datasets and Licenses

This repository inherits its license from the original release, and all datasets used are publicly available from the following links (among many others):

1. **PAWS-X**  
   Link: [PAWS-X Dataset](https://github.com/google-research-datasets/paws/tree/master/pawsx)

2. **SICK-R**  
   Link: [SICK-R Dataset](https://zenodo.org/records/2787612)

3. **MSRPC**  
   Link: [Microsoft Research Paraphrase Corpus](https://www.microsoft.com/en-us/download/details.aspx?id=52398)

4. **XNLI**  
   Link: [XNLI Dataset](https://cims.nyu.edu/~sbowman/xnli/)

5. **ANLI**  
   Link: [Adversarial NLI (ANLI)](https://github.com/facebookresearch/anli) 

6. **Stanford NLI (SNLI)**  
   Link: [SNLI Dataset](https://nlp.stanford.edu/projects/snli/)

7. **STS Benchmark**  
   Link: [STS Benchmark](https://ixa2.si.ehu.eus/stswiki/index.php/STSbenchmark)

8. **OneStopEnglish Corpus**  
   Link: [OneStopEnglish Corpus](https://github.com/nishkalavallabhi/OneStopEnglish)


Within this work, we introduce a dataset (and an annotation on an existing one) which are also available within our repository under the same license as the source dataset

1. **AMR True Paraphrases** Source: [AMR GUIDELINES](https://github.com/amrisi/amr-guidelines/blob/master/amr.md) Dataset: [AMR-True-Paraphrases](https://huggingface.co/datasets/impresso-project/amr-true-paraphrases).

2. **STS Benchmark (Scores 4-5) (STS-H) with Paraphrase Label** 
   Link: [STS Hard](https://huggingface.co/datasets/impresso-project/sts-h-paraphrase-detection)

## Further Support
This repository's benchmarking codebase was voluntary developed ad-hoc by Andreas Loizides.
In the future, we will work towards adding more datasets (also multilingual) and to make the benchmark more compute efficient. If you are interested in contributing or need further support reproducing/recreating/extending the results, please reach out to andrianos.michail@cl.uzh.ch.

## About Impresso

### Impresso project

[Impresso - Media Monitoring of the Past](https://impresso-project.ch) is an interdisciplinary research project that aims to develop and consolidate tools for processing and exploring large collections of media archives across modalities, time, languages and national borders. The first project (2017-2021) was funded by the Swiss National Science Foundation under grant No. [CRSII5_173719](http://p3.snf.ch/project-173719) and the second project (2023-2027) by the SNSF under grant No. [CRSII5_213585](https://data.snf.ch/grants/grant/213585) and the Luxembourg National Research Fund under grant No. 17498891.

### Copyright

Copyright (C) 2024 The Impresso team.

### License

This program is provided as open source under the [GNU Affero General Public License](https://github.com/impresso/impresso-pyindexation/blob/master/LICENSE) v3 or later.

---

<p align="center">
  <img src="https://github.com/impresso/impresso.github.io/blob/master/assets/images/3x1--Yellow-Impresso-Black-on-White--transparent.png?raw=true" width="350" alt="Impresso Project Logo"/>
</p>
