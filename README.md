# Quantifying Language Confusion
This is the official repository for our paper __Large Language Models are Easily Confused:
A Quantitative Metric, Security Implications and Typological Analysis__.


## Data and Results 
Please refer to [zenodo](https://zenodo.org/records/13946031) for datasets, language graphs, and results:

DATA include the following datasets:

i) Raw __Language Graphs__ and

ii) The calculated __Language Similarities__ from the Language Graphs,

iii) __MTEI__: the files from the experimental results of multilingual inversion attacks, and calculated language confusion entropy from the data;

iv) __LCB__: the files from the language confusion benchmark and calculated language confusion entropy from the data 

 

Results include aggregated results for further analysis:

i) inversion_language_confusion: results from __MTEI__

ii) prompting_language_confusion: results from LCB




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


[//]: # ()
[//]: # (## Preprocessing datasets)

[//]: # ()
[//]: # (* `notebooks/processing_colex_data.ipynb` )

[//]: # (  * get language to language graph based on colexification patterns, where nodes are languages, edges are the weights of shared colexifications)

[//]: # (  * CLICS3:)

[//]: # (    * input: `datasets/colexifications/colex_clics3.csv`)

[//]: # (    * output: `datasets/lang2lang/clics3.csv`)

[//]: # (  * WordNet from BabelNet &#40;5.0&#41;)

[//]: # (    * input: `datasets/colexifications/colex_wn_bn.csv`)

[//]: # (    * output: `datasets/lang2lang/wn_colex.csv`)

[//]: # (  )
[//]: # (* `notebooks/processing_typology_features.ipynb`)

[//]: # (  * processing features from WALS and Grambank)

[//]: # (  * into `lang:vec_of_features`)

[//]: # (  * output: )

[//]: # (    * `datasets/data_for_graph/WALS`)

[//]: # (      * language and their vectors of features)

[//]: # (      * edge weights for training language graph embeddings)

[//]: # (      )
[//]: # (    * `datasets/data_for_graph/Grambank`)

[//]: # (      * each language have at least 25% coverage of all features.)

[//]: # (      * edge weights and vector of features)

[//]: # (    )
[//]: # ()
[//]: # (* preprocessing data for analyzing the contribution of typological features for inversion language confusion)

[//]: # (  * `notebooks/language_confusion_typ_inversion_results.ipynb`)

[//]: # (    * outputs in `datasets/inversion_language_confusion`:)

[//]: # (      * `wals_features.csv`)

[//]: # (      * `gb_features.csv`)

[//]: # (  * `src/processing_data/processing_inversion_results_typology.py`)

[//]: # (    * get the typological features for regression )

[//]: # (    * output: )

[//]: # (      * `datasets/inversion_language_confusion/word_level/mono/train_data_clics3.csv`)


## Language Confusion Analysis 
`src/analysis_language_confusion`

[//]: # ()
[//]: # (### For Prompting datasets)

[//]: # ()
[//]: # (1. Language Identification)

[//]: # (`src/languageConfusion/post_eval_prompt_datasets.py`)

[//]: # ()
[//]: # (2. Measuring language confusion)

[//]: # ()
[//]: # ()
[//]: # ()
[//]: # (##  Language Graph Embeddings)

[//]: # ()
[//]: # ()
[//]: # (- Colexifications)

[//]: # (  - node2vec)

[//]: # (- Grambank)

[//]: # (  - GraphSAGE)

[//]: # (- WALS)

[//]: # (  - GraphSAGE)

[//]: # ()
[//]: # (## Analysis)

[//]: # ()
[//]: # (`src/analysis_language_confusion`)
