This project is a part of the course CE7455, Deep Learning for NLP, Nanyang Technological University.
We have enhanced the original KG-BERT Markup :  [https://github.com/yao8839836/kg-bert] and evaluated it against the KGLM https://github.com/rloganiv/kglm-model
# KG_TextGenerator

##To use KGLM, follow the below steps: 
* Training:
```sh
allennlp train experiments/kglm.jsonnet -s model --include-package kglm
 ```
Note: Had to comment out the line #iterator.eval() on line 163 in file: kglm/commands/evaluate_perplexity.py

* To evaluate the model on cloze type sentence completion task:

```sh
python -m kglm.run complete-the-sentence  model/model.tar.gz experiments/complete_the_sentence.jsonl --output-file output_data/predictions.txt --include-package kglm
```
##To use KG-BERT on KG completion tasks, follow the below steps:

* Triple Classification:
```sh
python triple_classifier.py  --task_name kg --do_train  --do_eval  --do_predict --data_dir ./data/linked_wikitext --bert_model bert-base-uncased --max_seq_length 128 --train_batch_size 32 --learning_rate 5e-5 --num_train_epochs 3.0 --output_dir ./triple_output_linked_wikitext/  --gradient_accumulation_steps 1 --eval_batch_size 512

```
* Relation Prediction:
```sh
python relation_prediction.py --task_name kg  --do_train  --do_eval --do_predict --data_dir ./data/linked_wikitext --bert_model bert-base-uncased --max_seq_length 128 --train_batch_size 32 --learning_rate 5e-5 --num_train_epochs 3.0 --output_dir ./output_relation_prediction/  --gradient_accumulation_steps 1 --eval_batch_size 512

```

* Link Prediction:
```sh
python link_pred.py  --task_name kg --do_train  --do_eval  --do_predict --data_dir ./data/linked_wikitext --bert_model bert-base-uncased --max_seq_length 20 --train_batch_size 32 --learning_rate 5e-5 --num_train_epochs 3.0 --output_dir ./output_linked_wikitext/  --checkpoint_dir ./checkpoint/ --gradient_accumulation_steps 1 --eval_batch_size 512

```
