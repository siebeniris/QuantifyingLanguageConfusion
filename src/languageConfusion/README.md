# Post evaluation and Language Confusion


## Tokenization and evaluate sequence token metrics and language detection


```
python language_confusion/post_eval.py multilingual inverter

python language_confusion/post_eval.py multilingual corrector

python language_confusion/post_eval.py monolingual inverter

python language_confusion/post_eval.py monolingual corrector

```


## Collect token set evaluations

```

python -m language_confusion.collect_token_set_eval multilingual token_set_f1 

python -m language_confusion.collect_token_set_eval multilingual num_true_words 

python -m language_confusion.collect_token_set_eval multilingual num_pred_words 

python -m language_confusion.collect_token_set_eval monolingual token_set_f1 results/mt5_mono

python -m language_confusion.collect_token_set_eval monolingual num_true_words results/mt5_mono

python -m language_confusion.collect_token_set_eval monolingual num_pred_words results/mt5_mono

```

## Get Dataset2language dictionary

```
python language_confusion/get_dataset2langdist.py mono

python language_confusion/get_dataset2langdist.py multi

python language_confusion/get_dataset2langdist.py mono+multi

```

### Get langdist to dataframe.
three mode: multi, mono, mono+multi
```
python language_confusion/langdist2df.py
```


## Prepare data for regression training 

```
python language_confusion/preprocessing_for_regression.py
```

## Predict languages using Random Forest

```

python language_confusion/pred_lang_regression.py

```

## Collect results from Regression

```
python collect_results_regression.py

```


##  plot.













