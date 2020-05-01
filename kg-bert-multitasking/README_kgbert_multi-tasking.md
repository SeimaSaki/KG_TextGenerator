
## Introduction

Code for [BERT and PALs](https://arxiv.org/abs/1902.02671) using which we adapted multi-task learning for link and relation prediction; 
`modeling.py` contains the BERT model formulation and `run_bert_link_prediction_multi.py` and `run_bert_relation_prediction_multi.py` perform multi-task training on the GLUE benchmark and KG optimization.

## Requirements

This code was tested on Python 3.5+. The requirements are:

- PyTorch (>= 0.4.1)
- tqdm
- scikit-learn (0.20.0)
- numpy (1.15.4)


Some basic details of the parts of the code used for multi-task learning were adapted from [BERT and PALs](https://arxiv.org/abs/1902.02671)):

We implement our multi-task sampling methods (annealed, proportional etc.) with `np.random.choice`. 

The [GLUE data](https://gluebenchmark.com/tasks) can be downloaded with
[this script](https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e). This README assumes it is located in `glue/glue_data`.

## Getting the pretrained weights

You can convert any TensorFlow checkpoint for BERT (in particular [the pre-trained models released by Google](https://github.com/google-research/bert#pre-trained-models)) in a PyTorch save file by using the [`./pytorch_pretrained_bert/convert_tf_checkpoint_to_pytorch.py`](convert_tf_checkpoint_to_pytorch.py) script.

This CLI takes as input a TensorFlow checkpoint (three files starting with `bert_model.ckpt`) and the associated configuration file (`bert_config.json`), and creates a PyTorch model for this configuration, loads the weights from the TensorFlow checkpoint in the PyTorch model and saves the resulting model in a standard PyTorch save file that can be imported using `torch.load()`

You only need to run this conversion script **once** to get a PyTorch model. You can then disregard the TensorFlow checkpoint (the three files starting with `bert_model.ckpt`) but be sure to keep the configuration file (`bert_config.json`) and the vocabulary file (`vocab.txt`) as these are needed for the PyTorch model too.

To run this specific conversion script you will need to have TensorFlow and PyTorch installed (`pip install tensorflow`). The rest of the repository only requires PyTorch.

Here is an example of the conversion process for a pre-trained `BERT-Base Uncased` model:

```shell
export BERT_BASE_DIR=/path/to/bert/uncased_L-12_H-768_A-12

pytorch_pretrained_bert convert_tf_checkpoint_to_pytorch \
  $BERT_BASE_DIR/bert_model.ckpt \
  $BERT_BASE_DIR/bert_config.json \
  $BERT_BASE_DIR/pytorch_model.bin
```

You can download Google's pre-trained models for the conversion [here](https://github.com/google-research/bert#pre-trained-models).
We use the BERT-base uncased: `uncased_L-12_H-768_A-12` model for all experiments. 

## KG-BERT Multi-Tasking

The bert config file contain the settings neccesary to reproduce the results of our work. 

`/configs/pals_config.json`: Contains the configuration with small hidden size 204.


Choose the `sample` argument to be 'anneal', 'sqrt', 'prop' or 'rr' for the various sampling methods. Choose 'anneal' to reproduce the best results. 

Here's an example of how to run the KG-BERT multi-tasking method with annealed sampling:

```shell
export BERT_BASE_DIR=/path/to/uncased_L-12_H-768_A-12
export BERT_PYTORCH_DIR=/path/to/uncased_L-12_H-768_A-12
export GLUE_DIR=/path/to/glue/glue_data
export SAVE_DIR=/tmp/saved

Train and evaluate relation prediction with multi-tasking

python3 run_bert_relation_prediction_multi.py --seed 55 --output_dir ./pals/ --tasks all --sample 'anneal' --multi --do_train --do_eval --data_dir glue_data/ --vocab_file ./uncased_L-12_H-768_A-12/vocab.txt --bert_config_file ./configs/pals_config.json --init_checkpoint ./uncased_L-12_H-768_A-12/pytorch_model.bin --max_seq_length 128 --train_batch_size 32 --learning_rate 3e-5 --num_train_epochs 5.0 --gradient_accumulation_steps 2

Test the model trained above

python3 run_bert_relation_prediction_multi.py --seed 42 --output_dir ./pals/ --tasks all --sample 'anneal' --multi --do_predict --data_dir glue_data/ --vocab_file ./uncased_L-12_H-768_A-12/vocab.txt --bert_config_file ./configs/pals_config.json --init_checkpoint ./pals/saved model --max_seq_length 128 --train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 5.0 --gradient_accumulation_steps 2

Train and evaluate link prediction with multi-tasking

python3 run_bert_link_prediction_multi.py --seed 50 --output_dir ./pals/linkpred/ --tasks all --sample 'anneal' --multi --do_train --do_eval --data_dir glue_data/ --vocab_file ./uncased_L-12_H-768_A-12/vocab.txt --bert_config_file ./configs/pals_config.json --init_checkpoint ./uncased_L-12_H-768_A-12/pytorch_model.bin --max_seq_length 128 --train_batch_size 32 --learning_rate 3e-5 --num_train_epochs 5.0 --gradient_accumulation_steps 2

Test the model trained above

python3 run_bert_link_prediction_multi.py --seed 85 --output_dir ./pals/linkpred/ --tasks all --sample 'anneal' --multi --do_predict --data_dir glue_data/ --vocab_file ./uncased_L-12_H-768_A-12/vocab.txt --bert_config_file ./configs/pals_config.json --init_checkpoint ./pals/saved model --max_seq_length 128 --train_batch_size 32 --learning_rate 3e-5 --num_train_epochs 5.0 --gradient_accumulation_steps 2

```
