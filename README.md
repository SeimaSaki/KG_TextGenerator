# KG_TextGenerator
* Training:
```sh
allennlp train experiments/kglm.jsonnet -s model --include-package kglm
 ```
Note: Had to comment out the line #iterator.eval() on line 163 in file: kglm/commands/evaluate_perplexity.py

* Complete the sentence:

```sh
python -m kglm.run complete-the-sentence  model/model.tar.gz experiments/complete_the_sentence.jsonl --output-file output_data/predictions.txt --include-package kglm
```

Note: changes to the file: kglm/commands/complete_the_sentence.py
