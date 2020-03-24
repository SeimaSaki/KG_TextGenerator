# KG_TextGenerator
* Training:
```sh
allennlp train experiments/kglm.jsonnet -s ms_train --include-package kglm
 ```
* Discriminator model training:
```sh
allennlp train experiments/kglm-disc.jsonnet -s ms_train_perplexity --include-package kglm
```
* Perplexity evaluation:

```sh
python -m kglm.run evaluate-perplexity --include-package kglm ms_train/model.tar.gz ms_train_perplexity/model.tar.gz data/linked-wikitext-2/valid.jsonl
```

Note: Had to comment out the line #iterator.eval() on line 163 in file: kglm/commands/evaluate_perplexity.py

* Complete the sentence:

```sh
python -m kglm.run complete-the-sentence ms_train/model.tar.gz ms_train_perplexity/model.tar.gz experiments/complete_the_sentence.jsonl --include-package kglm
```

Note: changes to the file: kglm/commands/complete_the_sentence.py
