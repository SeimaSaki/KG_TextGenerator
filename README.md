# KG_TextGenerator
1. Training:
allennlp train experiments/kglm.jsonnet -s ms_train --include-package kglm

2. Discriminator model training:
allennlp train experiments/kglm-disc.jsonnet -s ms_train_perplexity --include-package kglm

3. Perplexity evaluation:
#### Note: Had to comment out the line #iterator.eval() on line 163 in file - kglm/commands/evaluate_perplexity.py

python -m kglm.run evaluate-perplexity --include-package kglm ms_train/model.tar.gz ms_train_perplexity/model.tar.gz data/linked-wikitext-2/valid.jsonl

4. Complete the sentence:
##### Note: changes to the file - kglm/commands/complete_the_sentence.py

python -m kglm.run complete-the-sentence ms_train/model.tar.gz ms_train_perplexity/model.tar.gz backup/data/manojs2.jsonl --include-package kglm
