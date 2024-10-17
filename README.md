# Quantifying Language Confusion
This is the official repository for our paper __Large Language Models are Easily Confused:
A Quantitative Metric, Security Implications and Typological Analysis__.




## Installation

1. Download the repository to local:

2. Create a new conda environment

`conda create -n envlc python==3.12`

`conda activate envlc`

3. Install pytorch and packages from requirements

`pip3 install torch torchvision torchaudio`

`pip install -r requirements.txt`

4. Specifics 
- Tokenize Japanese, after installing `fugashi[unidic]`:

`python -m unidic download`

## Language Similarities

### By features
- Lang2Lang Pointwise Mutual Information by Overlapping Colexification Patterns (CLICS3):
  - `datasets/lang2lang/clics3.csv`
- Lang2Lang Euclidean Distance from Grambank 
  - (data source and code: https://github.com/WPoelman/typ-div-survey)
  - `datasets/lang2lang/grambank_sim.csv`
- Lang2Lang Euclidean Distance from WALS
  - `datasets/lang2lang/wals_sim.csv`



## Preprocessing datasets

* `notebooks/processing_colex_data.ipynb` 
  * get language to language graph based on colexification patterns, where nodes are languages, edges are the weights of shared colexifications
  * CLICS3:
    * input: `datasets/colexifications/colex_clics3.csv`
    * output: `datasets/lang2lang/clics3.csv`
  * WordNet from BabelNet (5.0)
    * input: `datasets/colexifications/colex_wn_bn.csv`
    * output: `datasets/lang2lang/wn_colex.csv`
  
* `notebooks/processing_typology_features.ipynb`
  * processing features from WALS and Grambank
  * into `lang:vec_of_features`
  * output: 
    * `datasets/data_for_graph/WALS`
      * language and their vectors of features
      * edge weights for training language graph embeddings
      
    * `datasets/data_for_graph/Grambank`
      * each language have at least 25% coverage of all features.
      * edge weights and vector of features
    

* preprocessing data for analyzing the contribution of typological features for inversion language confusion
  * `notebooks/language_confusion_typ_inversion_results.ipynb`
    * outputs in `datasets/inversion_language_confusion`:
      * `wals_features.csv`
      * `gb_features.csv`
  * `src/processing_data/processing_inversion_results_typology.py`
    * get the typological features for regression 
    * output: 
      * `datasets/inversion_language_confusion/word_level/mono/train_data_clics3.csv`


## Language Confusion

### For Prompting datasets

1. Language Identification
`src/languageConfusion/post_eval_prompt_datasets.py`

2. Measuring language confusion



##  Language Graph Embeddings


- Colexifications
  - node2vec
- Grambank
  - GraphSAGE
- WALS
  - GraphSAGE

## Analysis

`src/analysis_language_confusion`
