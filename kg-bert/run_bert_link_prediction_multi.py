# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import sys
from itertools import cycle

import numpy as np

import torch
print(torch.cuda.is_available())

from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from torch.nn import CrossEntropyLoss, MSELoss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score
from sklearn import metrics

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
##including bertformultitask to train bert on two tasks - link prediction and textual entailment
# from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertConfig, BertForMultiTask

import tokenization
from modeling import BertForSequenceClassification, BertConfig, BertForMultiTask


from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule

os.environ['CUDA_VISIBLE_DEVICES']= '1'
#torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
print('device ', device, n_gpu)

logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, text_c=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            text_c: (Optional) string. The untokenized text of the third sequence.
            Only must be specified for sequence triple tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.text_c = text_c
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self, data_dir):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines


class KGProcessor(DataProcessor):
    """Processor for knowledge graph data set."""
    def __init__(self):
        self.labels = set()
    
    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train", data_dir)

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev", data_dir)

    def get_test_examples(self, data_dir):
      """See base class."""
      return self._create_examples(
          self._read_tsv(os.path.join(data_dir, "test.tsv")), "test", data_dir)

    def get_relations(self, data_dir):
        """Gets all labels (relations) in the knowledge graph."""
        # return list(self.labels)
        with open(os.path.join(data_dir, "relations.tsv"), 'r') as f:
            lines = f.readlines()
            relations = []
            for line in lines:
                relations.append(line.strip())
        return relations

    def get_labels(self, data_dir):
        """Gets all labels (0, 1) for triples in the knowledge graph."""
        return ["0", "1"]

    def get_entities(self, data_dir):
        """Gets all entities in the knowledge graph."""
        # return list(self.labels)
        with open(os.path.join(data_dir, "entities.txt"), 'r') as f:
            lines = f.readlines()
            entities = []
            for line in lines:
                entities.append(line.strip())
        return entities

    def get_train_triples(self, data_dir):
        """Gets training triples."""
        return self._read_tsv(os.path.join(data_dir, "train.tsv"))

    def get_dev_triples(self, data_dir):
        """Gets validation triples."""
        return self._read_tsv(os.path.join(data_dir, "dev.tsv"))

    def get_test_triples(self, data_dir):
        """Gets test triples."""
        return self._read_tsv(os.path.join(data_dir, "test.tsv"))

    def _create_examples(self, lines, set_type, data_dir):
        """Creates examples for the training and dev sets."""
        # entity to text
        ent2text = {}
        with open(os.path.join(data_dir, "entity2text.txt"), 'r') as f:
            ent_lines = f.readlines()
            for line in ent_lines:
                temp = line.strip().split('\t')
                if len(temp) == 2:
                    end = temp[1]#.find(',')
                    ent2text[temp[0]] = temp[1]#[:end]
  
        if data_dir.find("FB15") != -1:
            with open(os.path.join(data_dir, "entity2textlong.txt"), 'r') as f:
                ent_lines = f.readlines()
                for line in ent_lines:
                    temp = line.strip().split('\t')
                    #first_sent_end_position = temp[1].find(".")
                    ent2text[temp[0]] = temp[1]#[:first_sent_end_position + 1] 

        entities = list(ent2text.keys())

        rel2text = {}
        with open(os.path.join(data_dir, "relation2text.txt"), 'r') as f:
            rel_lines = f.readlines()
            for line in rel_lines:
                temp = line.strip().split('\t')
                rel2text[temp[0]] = temp[1]      

        lines_str_set = set(['\t'.join(line) for line in lines])
        examples = []
        for (i, line) in enumerate(lines):
            
            head_ent_text = ent2text[line[0]]
            tail_ent_text = ent2text[line[2]]
            relation_text = rel2text[line[1]]

            if set_type == "dev" or set_type == "test":

                label = "1"

                guid = "%s-%s" % (set_type, i)
                text_a = head_ent_text
                text_b = relation_text
                text_c = tail_ent_text 
                self.labels.add(label)
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, text_c = text_c, label=label))
                
            elif set_type == "train":
                guid = "%s-%s" % (set_type, i)
                text_a = head_ent_text
                text_b = relation_text
                text_c = tail_ent_text 
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, text_c = text_c, label="1"))

                rnd = random.random()
                guid = "%s-%s" % (set_type + "_corrupt", i)
                if rnd <= 0.5:
                    # corrupting head
                    for j in range(5):
                        tmp_head = ''
                        while True:
                            tmp_ent_list = set(entities)
                            tmp_ent_list.remove(line[0])
                            tmp_ent_list = list(tmp_ent_list)
                            tmp_head = random.choice(tmp_ent_list)
                            tmp_triple_str = tmp_head + '\t' + line[1] + '\t' + line[2]
                            if tmp_triple_str not in lines_str_set:
                                break                    
                        tmp_head_text = ent2text[tmp_head]
                        examples.append(
                            InputExample(guid=guid, text_a=tmp_head_text, text_b=text_b, text_c = text_c, label="0"))       
                else:
                    # corrupting tail
                    tmp_tail = ''
                    for j in range(5):
                        while True:
                            tmp_ent_list = set(entities)
                            tmp_ent_list.remove(line[2])
                            tmp_ent_list = list(tmp_ent_list)
                            tmp_tail = random.choice(tmp_ent_list)
                            tmp_triple_str = line[0] + '\t' + line[1] + '\t' + tmp_tail
                            if tmp_triple_str not in lines_str_set:
                                break
                        tmp_tail_text = ent2text[tmp_tail]
                        examples.append(
                            InputExample(guid=guid, text_a=text_a, text_b=text_b, text_c = tmp_tail_text, label="0"))                                                  
        return examples


## data processor for QNLI q&a data
class QNLIProcessor(DataProcessor):
    """Processor for the QNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "test")

    def get_test_triples(self, data_dir):
        """Gets test triples."""
        return self._read_tsv(os.path.join(data_dir, "dev.tsv"))

    def get_dev_triples(self, data_dir):
        """Gets dev triples."""
        return self._read_tsv(os.path.join(data_dir, "dev.tsv"))

    def get_train_triples(self, data_dir):
        """Gets training triples."""
        return self._read_tsv(os.path.join(data_dir, "train.tsv"))

    def get_labels(self):
        """See base class."""
        return ["not_entailment", "entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
           
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(line[1])
            text_b = tokenization.convert_to_unicode(line[2])
            label = tokenization.convert_to_unicode(line[3])

            #print(text_a)
            #print(text_b)
            #print(label)

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, print_info = True):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {}
    for (i, label) in enumerate(label_list):
        # print(i)
        # print(label)
        label_map[label] = i
    ##print ('total examples: ', len(examples))
    features = []
    for (ex_index, example) in enumerate(examples):
        # print(example)
        # print ('Example ', example.text_a)
        # print(example.text_b)
        # print(example.text_c)
        # print ('Example label ', example.label)
        if ex_index % 10000 == 0 and print_info:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        tokens_c = None

        if example.text_b and example.text_c:
            tokens_b = tokenizer.tokenize(example.text_b)
            tokens_c = tokenizer.tokenize(example.text_c)
            # Modifies `tokens_a`, `tokens_b` and `tokens_c`in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP], [SEP] with "- 4"
            #_truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
            _truncate_seq_triple(tokens_a, tokens_b, tokens_c, max_seq_length - 4)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        # (c) for sequence triples:
        #  tokens: [CLS] Steve Jobs [SEP] founded [SEP] Apple Inc .[SEP]
        #  type_ids: 0 0 0 0 1 1 0 0 0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence or the third sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)
        if tokens_c:
            tokens += tokens_c + ["[SEP]"]
            segment_ids += [0] * (len(tokens_c) + 1)        

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]

        # ### prepare input sequence file
        # ip_file_str = ' '.join([str(x) for x in tokens])
        # ip_file_str += ' Actual label: '
        # ip_file_str += example.label
        # ip_file_str += '\n'
        # ip_file.write(ip_file_str)

        if ex_index < 5 and print_info:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def _truncate_seq_triple(tokens_a, tokens_b, tokens_c, max_length):
    """Truncates a sequence triple in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    ###> changed to >= for multi tasking with linked wikitext
    while True:
        total_length = len(tokens_a) + len(tokens_b) + len(tokens_c)
        if total_length <= max_length:
            break
        if len(tokens_a) >= len(tokens_b) and len(tokens_a) >= len(tokens_c):
            tokens_a.pop()
        elif len(tokens_b) >= len(tokens_a) and len(tokens_b) >= len(tokens_c):
            tokens_b.pop()
        elif len(tokens_c) >= len(tokens_a) and len(tokens_c) >= len(tokens_b):
            tokens_c.pop()
        else:
            tokens_c.pop()

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def accuracy(out, labels):
  outputs = np.argmax(out, axis=1)
  return np.sum(outputs==labels)

def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "kg":
        return {"acc": simple_accuracy(preds, labels)}
    else:
        raise KeyError(task_name)


def do_eval(model, logger, args, tr_loss, nb_tr_steps, global_step, processor, 
            label_list, tokenizer, eval_dataloader, task_id, i):

        model.eval()
        ## adding evaluation accuracy as a metric
        eval_loss, eval_accuracy = 0, 0
        ## initialising count of evaluation examples
        nb_eval_steps, nb_eval_examples = 0, 0
        preds = []

        for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)

            with torch.no_grad():
                ##logits = model(input_ids, segment_ids, input_mask, labels=None)
                
                ### change for multiple tasks
                tmp_eval_loss, logits = model(input_ids, segment_ids, input_mask, i, task_id, label_ids)
                
                #print('model op eval : ', logits)

            ## create eval loss and other metric required by each task
            if task_id == 'qnli':
              logits = logits.detach().cpu().numpy()
              ## moving label ids to cpu
              label_ids = label_ids.to('cpu').numpy()
              ## computing eval accuracy in addition to eval loss
              tmp_eval_accuracy = accuracy(logits, label_ids)
            else:   ## for kg
              logits = logits.detach().cpu().numpy()
              ## moving label ids to cpu
              label_ids = label_ids.to('cpu').numpy()
              ## computing eval accuracy in addition to eval loss
              tmp_eval_accuracy = accuracy(logits, label_ids)
              
         #   print(label_ids.view(-1))
            
            eval_loss += tmp_eval_loss.mean().item()
            
            ## assigning num of eval examples
            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1

            # if len(preds) == 0:
            #     preds.append(logits.detach().cpu().numpy())
            # else:
            #     preds[0] = np.append(
            #         preds[0], logits.detach().cpu().numpy(), axis=0)

            if task_id == 'cola' or task_id == 'sts':
              eval_accuracy += tmp_eval_accuracy * input_ids.size(0)
            else:
              eval_accuracy += tmp_eval_accuracy


        eval_loss = eval_loss / nb_eval_steps
        ## computing eval accuracy
        eval_accuracy = eval_accuracy / nb_eval_examples
        
        # preds = preds[0]

        # preds = np.argmax(preds, axis=1)

        ##convert prediction ids to prediction textand insert it into ip_op_file
        # with open('eval_ip.txt', 'r') as istr:
        #   with open('eval_ip_op.txt', 'w') as ostr:
        #     for i, line in enumerate(istr):
        #         # Get rid of the trailing newline (if any).
        #         line = line.rstrip('\n')
        #         pred_text = label_map[preds[i-1]]
        #         print('input sequence: ', line, ' predicted label: ', pred_text, file=ostr)
        #         print('\n', file=ostr)

        
        # result = compute_metrics(task_name, preds, all_label_ids.numpy())
        loss = tr_loss/nb_tr_steps if args.do_train else None

        result = {'eval_loss': eval_loss,
                  'eval_accuracy': eval_accuracy,
                  'global_step': global_step,
                  'loss': loss}

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
          logger.info("***** Num of Eval results *****")
          #logger.info(len(result.keys()))
          for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))
        
        return eval_accuracy

def do_kg_test(model, logger, args, data_dir, tr_loss, nb_tr_steps, global_step, processor, 
            label_list, tokenizer, test_dataloader, task_id, i, all_label_ids, test_triples, str_set, entity_list):
            
    test_loss = 0
    nb_eval_steps = 0
    nb_test_examples = 0
    test_accuracy = 0
    preds = []
        
    for input_ids, input_mask, segment_ids, label_ids in tqdm(test_dataloader, desc="Testing"):
             
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)
        
        #print('label ids ', label_ids)
        

        with torch.no_grad():
            tmp_test_loss, logits = model(input_ids, segment_ids, input_mask, i, task_id, label_ids)
            # print ('model op during predictions : ', logits)
             
        logits = logits.detach().cpu().numpy()
        ## moving label ids to cpu
        label_ids = label_ids.to('cpu').numpy()
        ## computing eval accuracy in addition to eval loss

        #print('logits ', logits)

        tmp_test_accuracy = accuracy(logits, label_ids)

        ##loss_fct = CrossEntropyLoss()
        ##tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
            
        test_loss += tmp_test_loss.mean().item()
        nb_eval_steps += 1
        nb_test_examples += input_ids.size(0)
        #print('loop test ex ', nb_test_examples)

        if len(preds) == 0:
            preds.append(logits)
        else:
            preds[0] = np.append(preds[0], logits, axis=0)
        
        if task_id == 'cola' or task_id == 'sts':
           test_accuracy += tmp_test_accuracy * input_ids.size(0)
        else:
           test_accuracy += tmp_test_accuracy

    test_loss = test_loss / nb_eval_steps
    preds = preds[0]

    test_accuracy = test_accuracy / nb_test_examples

    preds = np.argmax(preds, axis=1)

    # ##convert prediction ids to prediction textand insert it into ip_op_file

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[i] = label
        
    # print('label map ', len(label_map))
    

    # with open('test_ip_0.txt', 'r') as istr:
    #     with open('test_ip_op_0.txt', 'w') as ostr:
    #         for j, line in enumerate(istr):
    #             # Get rid of the trailing newline (if any).
    #             line = line.rstrip('\n')
    #             print('preds ', preds[j-1])
    #             pred_text = label_map[preds[j-1]]
    #             print('input sequence: ', line, ' predicted label: ', pred_text, file=ostr)
    #             print('\n', file=ostr)
   
        # ####

    #result = compute_metrics(task_id, preds, all_label_ids)
    loss = tr_loss/nb_tr_steps if args.do_train else None

    # result['eval_loss'] = eval_loss
    # result['global_step'] = global_step
    ##result['loss'] = loss
    result = {'test_loss': test_loss,
              'test_accuracy': test_accuracy,
              'loss': loss,
              'global_step': global_step}

    output_eval_file = os.path.join(args.output_dir, "test_results_0.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Test results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))
    ## relation prediction, raw
    print("Link prediction hits@1, raw...")
    print(metrics.accuracy_score(all_label_ids, preds))


    # run link prediction
    ranks = []
    ranks_left = []
    ranks_right = []

    hits_left = []
    hits_right = []
    hits = []

    top_ten_hit_count = 0

    for j in range(10):
      hits_left.append([])
      hits_right.append([])
      hits.append([])

    test_triples = test_triples[0:1]
    for test_triple in test_triples:
      head = test_triple[0]
      relation = test_triple[1]
      tail = test_triple[2]
      #print(test_triple, head, relation, tail)

      head_corrupt_list = [test_triple]
      for corrupt_ent in entity_list:
        if corrupt_ent != head:
          tmp_triple = [corrupt_ent, relation, tail]
          tmp_triple_str = '\t'.join(tmp_triple)
          if tmp_triple_str not in str_set:
            # may be slow
            head_corrupt_list.append(tmp_triple)

      tmp_examples = processor._create_examples(head_corrupt_list, "test", data_dir)
      print(len(tmp_examples))
      tmp_features = convert_examples_to_features(tmp_examples, label_list, args.max_seq_length, tokenizer, print_info = False)
      all_input_ids = torch.tensor([f.input_ids for f in tmp_features], dtype=torch.long)
      all_input_mask = torch.tensor([f.input_mask for f in tmp_features], dtype=torch.long)
      all_segment_ids = torch.tensor([f.segment_ids for f in tmp_features], dtype=torch.long)
      all_label_ids = torch.tensor([f.label_id for f in tmp_features], dtype=torch.long)

      eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
      # Run prediction for temp data
      eval_sampler = SequentialSampler(eval_data)
      eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
      model.eval()

      preds = []
            
      for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Testing"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)
                
        with torch.no_grad():
          tmp_test_loss, logits = model(input_ids, segment_ids, input_mask, i, task_id, label_ids)
        batch_logits = logits.detach().cpu().numpy()
        if len(preds) == 0:
          preds.append(batch_logits)
        else:
          preds[0] = np.append(preds[0], batch_logits, axis=0)       

      preds = preds[0]
      # get the dimension corresponding to current label 1
      #print(preds, preds.shape)
      rel_values = preds[:, all_label_ids[0]]
      rel_values = torch.tensor(rel_values)
      #print(rel_values, rel_values.shape)
      _, argsort1 = torch.sort(rel_values, descending=True)
      #print(max_values)
      #print(argsort1)
      argsort1 = argsort1.cpu().numpy()
      rank1 = np.where(argsort1 == 0)[0][0]
      print('left: ', rank1)
      ranks.append(rank1+1)
      ranks_left.append(rank1+1)
      if rank1 < 10:
        top_ten_hit_count += 1

      tail_corrupt_list = [test_triple]
      for corrupt_ent in entity_list:
        if corrupt_ent != tail:
          tmp_triple = [head, relation, corrupt_ent]
          tmp_triple_str = '\t'.join(tmp_triple)
          if tmp_triple_str not in str_set:
            # may be slow
            tail_corrupt_list.append(tmp_triple)

      tmp_examples = processor._create_examples(tail_corrupt_list, "test", data_dir)
      #print(len(tmp_examples))
      tmp_features = convert_examples_to_features(tmp_examples, label_list, args.max_seq_length, tokenizer, print_info = False)
      all_input_ids = torch.tensor([f.input_ids for f in tmp_features], dtype=torch.long)
      all_input_mask = torch.tensor([f.input_mask for f in tmp_features], dtype=torch.long)
      all_segment_ids = torch.tensor([f.segment_ids for f in tmp_features], dtype=torch.long)
      all_label_ids = torch.tensor([f.label_id for f in tmp_features], dtype=torch.long)

      eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
      # Run prediction for temp data
      eval_sampler = SequentialSampler(eval_data)
      eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
      model.eval()
      preds = []        

      for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Testing"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)
                
        with torch.no_grad():
          tmp_test_loss, logits = model(input_ids, segment_ids, input_mask, i, task_id, label_ids)
        batch_logits = logits.detach().cpu().numpy()
        if len(preds) == 0:
          preds.append(batch_logits)
        else:
          preds[0] = np.append(preds[0], batch_logits, axis=0) 

      preds = preds[0]
      # get the dimension corresponding to current label 1
      rel_values = preds[:, all_label_ids[0]]
      rel_values = torch.tensor(rel_values)
      _, argsort1 = torch.sort(rel_values, descending=True)
      argsort1 = argsort1.cpu().numpy()
      rank2 = np.where(argsort1 == 0)[0][0]
      ranks.append(rank2+1)
      ranks_right.append(rank2+1)
      print('right: ', rank2)
      print('mean rank until now: ', np.mean(ranks))
      if rank2 < 10:
        top_ten_hit_count += 1
      print("hit@10 until now: ", top_ten_hit_count * 1.0 / len(ranks))

      file_prefix = str(data_dir[7:]) + "_" + str(args.train_batch_size) + "_" + str(args.learning_rate) + "_" + str(args.max_seq_length) + "_" + str(args.num_train_epochs)
      #file_prefix = str(args.data_dir[7:])
      f = open(file_prefix + '_ranks.txt','a')
      f.write(str(rank1) + '\t' + str(rank2) + '\n')
      f.close()
      # this could be done more elegantly, but here you go
      for hits_level in range(10):
        if rank1 <= hits_level:
          hits[hits_level].append(1.0)
          hits_left[hits_level].append(1.0)
        else:
          hits[hits_level].append(0.0)
          hits_left[hits_level].append(0.0)

        if rank2 <= hits_level:
          hits[hits_level].append(1.0)
          hits_right[hits_level].append(1.0)
        else:
          hits[hits_level].append(0.0)
          hits_right[hits_level].append(0.0)
    

    for i in [0,2,9]:
      logger.info('Hits left @{0}: {1}'.format(i+1, np.mean(hits_left[i])))
      logger.info('Hits right @{0}: {1}'.format(i+1, np.mean(hits_right[i])))
      logger.info('Hits @{0}: {1}'.format(i+1, np.mean(hits[i])))
    logger.info('Mean rank left: {0}'.format(np.mean(ranks_left)))
    logger.info('Mean rank right: {0}'.format(np.mean(ranks_right)))
    logger.info('Mean rank: {0}'.format(np.mean(ranks)))
    logger.info('Mean reciprocal rank left: {0}'.format(np.mean(1./np.array(ranks_left))))
    logger.info('Mean reciprocal rank right: {0}'.format(np.mean(1./np.array(ranks_right))))
    logger.info('Mean reciprocal rank: {0}'.format(np.mean(1./np.array(ranks)))) 
        
    return test_accuracy


def do_glue_test(model, logger, args, tr_loss, nb_tr_steps, global_step, 
            label_list, test_dataloader, task_id, i, all_label_ids):
            
    test_loss = 0
    nb_eval_steps = 0
    nb_test_examples = 0
    test_accuracy = 0
    preds = []
        
    for input_ids, input_mask, segment_ids, label_ids in tqdm(test_dataloader, desc="Testing"):
             
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)
        
        #print('label ids ', label_ids)
        

        with torch.no_grad():
            tmp_test_loss, logits = model(input_ids, segment_ids, input_mask, i, task_id, label_ids)
            # print ('model op during predictions : ', logits)
             
        logits = logits.detach().cpu().numpy()
        ## moving label ids to cpu
        label_ids = label_ids.to('cpu').numpy()
        ## computing eval accuracy in addition to eval loss

        #print('logits ', logits)

        tmp_test_accuracy = accuracy(logits, label_ids)

        ##loss_fct = CrossEntropyLoss()
        ##tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
            
        test_loss += tmp_test_loss.mean().item()
        nb_eval_steps += 1
        nb_test_examples += input_ids.size(0)
        #print('loop test ex ', nb_test_examples)

        if len(preds) == 0:
            preds.append(logits)
        else:
            preds[0] = np.append(preds[0], logits, axis=0)
        
        if task_id == 'cola' or task_id == 'sts':
           test_accuracy += tmp_test_accuracy * input_ids.size(0)
        else:
           test_accuracy += tmp_test_accuracy

    test_loss = test_loss / nb_eval_steps
    preds = preds[0]

    test_accuracy = test_accuracy / nb_test_examples
    # print('test ex ', nb_test_examples)
    #print('acc ', test_accuracy)
       
        # print('predictions ', preds, preds.shape)
        
    preds = np.argmax(preds, axis=1)

    # ##convert prediction ids to prediction textand insert it into ip_op_file

    label_map = {}
    for (j, label) in enumerate(label_list):
        label_map[j] = label
        
    #print('label map ', label_map)
    #print('preds ', preds)

    # with open('test_ip_1.txt', 'r') as istr:
    #     with open('test_ip_op_1.txt', 'w') as ostr:
    #         for j, line in enumerate(istr):
    #             # Get rid of the trailing newline (if any).
    #             line = line.rstrip('\n')
    #             pred_text = label_map[preds[j-1]]
    #             print('input sequence: ', line, ' predicted label: ', pred_text, file=ostr)
    #             print('\n', file=ostr)
   
        # ####

    #result = compute_metrics(task_id, preds, all_label_ids)
    loss = tr_loss/nb_tr_steps if args.do_train else None

    # result['eval_loss'] = eval_loss
    # result['global_step'] = global_step
    ##result['loss'] = loss
    result = {'test_loss': test_loss,
              'test_accuracy': test_accuracy,
              'loss': loss,
              'global_step': global_step}

    output_eval_file = os.path.join(args.output_dir, "test_results_1.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Test results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))
    ## relation prediction, raw
    # print("Relation prediction hits@1, raw...")
    print(metrics.accuracy_score(all_label_ids, preds))
        
    return test_accuracy
        

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    # parser.add_argument("--bert_model", default=None, type=str, required=True,
    #                     help="Bert pre-trained model selected in the list: bert-base-uncased, "
    #                     "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
    #                     "bert-base-multilingual-cased, bert-base-chinese.")
    # parser.add_argument("--task_name",
    #                     default=None,
    #                     type=str,
    #                     required=True,
    #                     help="The name of the task to train.")

    ## including bert config and vocab file
    parser.add_argument("--bert_config_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The config json file corresponding to the pre-trained BERT model. \n"
                             "This specifies the model architecture.")
    parser.add_argument("--vocab_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The vocabulary file that the BERT model was trained on.")

    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters

    ##added for multi-tasking
    parser.add_argument("--init_checkpoint",
                        default=None,
                        type=str,
                        help="Initial checkpoint (usually from a pre-trained BERT model).")


    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")

    ## added for multi-tasking
    parser.add_argument("--multi",
                        default=False,
                        help="Whether to add adapter modules",
                        action='store_true')
    parser.add_argument("--optim",
                        default='normal',
                        help="Whether to split up the optimiser between adapters and not adapters.")
    parser.add_argument("--sample",
                        default='rr',
                        help="How to sample tasks, other options 'prop', 'sqrt' or 'anneal'")


    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict",
                        action='store_true',
                        help="Whether to run eval on the test set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")


    ## added for multi-tasking
    parser.add_argument("--h_aug",
                        default="n/a",
                        help="Size of hidden state for adapters..")
    parser.add_argument("--tasks",
                        default="all",
                        help="Which set of tasks to train on.")
    parser.add_argument("--task_id",
                        default=1,
                        help="ID of single task to train on if using that setting.")


    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")

    ## added for multi-tasking
    parser.add_argument("--save_checkpoints_steps",
                        default=1000,
                        type=int,
                        help="How often to save the model checkpoint.")
    parser.add_argument("--freeze",
                        default=False,
                        action='store_true',
                        help="Freeze base network weights")


    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    # parser.add_argument('--fp16',
    #                     action='store_true',
    #                     help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    args = parser.parse_args()

    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    processors = {
        "kg": KGProcessor,
        "qnli": QNLIProcessor,
    }
    
    # print('cuda avail ', torch.cuda.is_available())
    # if args.local_rank == -1 or args.no_cuda:
    #     print('cuda avail ', torch.cuda.is_available())
    #     print('cuda backend enabled ', torch.backends.cudnn.enabled)
    #     device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    #     n_gpu = torch.cuda.device_count()
    # else:
    #     torch.cuda.set_device(args.local_rank)
    #     device = torch.device("cuda", args.local_rank)
    #     n_gpu = 1
    #     # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    #     torch.distributed.init_process_group(backend='nccl')

    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    # logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
    #     device, n_gpu, bool(args.local_rank != -1), args.fp16))

    ## for multi-tasking
    logger.info("device %s n_gpu %d distributed training %r", device, n_gpu, bool(args.local_rank != -1))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
    args.seed = random.randint(1, 200)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval and not args.do_predict:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    ## added for multi-tasking
    bert_config = BertConfig.from_json_file(args.bert_config_file)
    if args.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length {} because the BERT model was only trained up to sequence length {}".format(
            args.max_seq_length, bert_config.max_position_embeddings))


    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # task_name = args.task_name.lower()

    # if task_name not in processors:
    #     raise ValueError("Task not found: %s" % (task_name))

    # processor = processors[task_name]()

    # label_list = processor.get_labels(args.data_dir)

  ### for multi-tasking
    if args.tasks == 'all':
        task_names =['kg', 'qnli']
        data_dirs = ['./data/linked_wikitext', './glue_data/QNLI']
    elif args.tasks == 'single':
        task_names = ['kg', 'qnli']
        data_dirs = ['./data/linked_wikitext', './glue_data/QNLI']
        task_names = [task_names[int(args.task_id)]]
        data_dirs = [data_dirs[int(args.task_id)]]
    if task_names[0] not in processors:
        raise ValueError("Task not found: %s" % (task_name))


    processor_list = []
    label_list = []
    entity_list = []
    
    for task_name in task_names:
      processor_list.append(processors[task_name]())
      if task_name == 'kg':
        label_list.append(processors[task_name]().get_labels('./data/linked_wikitext/'))
        tokenizer = tokenization.FullTokenizer(vocab_file=args.vocab_file, do_lower_case=args.do_lower_case)
        entity_list.append(processors[task_name]().get_entities('./data/linked_wikitext/'))
        ##print('label list kg ', label_list)    
      else:
        label_list.append(processors[task_name]().get_labels())
        tokenizer = tokenization.FullTokenizer(vocab_file=args.vocab_file, do_lower_case=args.do_lower_case)
        ##print('label list qqp ', label_list)    
       

    num_labels = len(label_list)

    print('label list kg ', label_list[0])
    print('label list QNLI ', label_list[1])
    
    
    #print(entity_list)

    # tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    label_map = {i : label for i, label in enumerate(label_list)}

    train_examples = None
    num_train_optimization_steps = 0

    
    ## added for multi-tasking
    num_train_steps = None
    num_tasks = len(task_names)
    total_tr = 0.

    if args.do_train:
        # train_examples = processor.get_train_examples(args.data_dir)

         ## for multi-tasking
        train_examples = [processor.get_train_examples(data_dir) for processor, data_dir in zip(processor_list, data_dirs)]


        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        
        ## added for multi-tasking
        num_train_steps = int(
            len(train_examples[0]) / args.train_batch_size * args.num_train_epochs)
        
        if args.tasks == 'all':
            total_tr = 300 * num_tasks * args.num_train_epochs
        else:
            total_tr = int(0.5 * num_train_steps)

        if args.tasks == 'all':
          #steps_per_epoch = args.gradient_accumulation_steps * 300 * num_tasks
          steps_per_epoch = args.gradient_accumulation_steps * 50 * num_tasks
        else:
          steps_per_epoch = int(num_train_steps/(2. * args.num_train_epochs)) 
        print('steps per epoch ', steps_per_epoch) 
        
        # if args.local_rank != -1:
        #     num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    # Prepare model
    # cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(args.local_rank))
    # model = BertForSequenceClassification.from_pretrained(args.bert_model,
    #           cache_dir=cache_dir,
    #           num_labels=num_labels)
    # if args.fp16:
    #     model.half()

    # Prepare model FOR MULTI TASKING
    bert_config.num_tasks = num_tasks
    if args.h_aug is not 'n/a':
        bert_config.hidden_size_aug = int(args.h_aug)
    model = BertForMultiTask(bert_config, [len(labels) for labels in label_list])

    ## added for multi-tasking
    if args.init_checkpoint is not None:
      if args.multi:
           partial = torch.load(args.init_checkpoint, map_location='cpu')
          ##  model.load_state_dict(torch.load(partial))
           model_dict = model.bert.state_dict()
           update = {}
           for n, p in model_dict.items():
                if 'aug' in n or 'mult' in n:
                    ##print('in aug')
                    update[n] = p
                    if 'pooler.mult' in n and 'bias' in n:
                        update[n] = partial['pooler.dense.bias']
                    if 'pooler.mult' in n and 'weight' in n:
                        update[n] = partial['pooler.dense.weight']
                elif 'gamma' in n and '.bin' in args.init_checkpoint:
                  ##print('in gamma')
                  new_n = n.replace("gamma", "weight")
                  update[n] = partial['bert.' + new_n]
                elif 'beta' in n and '.bin' in args.init_checkpoint:
                  ##print('in beta')
                  new_n = n.replace("beta", "bias")
                  update[n] = partial['bert.' + new_n]
                elif 'classifier' in n:
                  update[n] = None
                else:
                  ##print('in last')
                  update[n] = partial['bert.'+ n]

           model.bert.load_state_dict(update)
      else:
           model.bert.load_state_dict(torch.load(args.init_checkpoint, map_location='cpu'))

    if args.freeze:
        for n, p in model.bert.named_parameters():
            if 'aug' in n or 'classifier' in n or 'mult' in n or 'gamma' in n or 'beta' in n:
                continue
            p.requires_grad = False
 
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)
        #model = torch.nn.parallel.data_parallel(model)
    # Prepare optimizer
    # param_optimizer = list(model.named_parameters())
    # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    #     {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    #     ]
    # if args.fp16:
    #     try:
    #         from apex.optimizers import FP16_Optimizer
    #         from apex.optimizers import FusedAdam
    #     except ImportError:
    #         raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

    #     optimizer = FusedAdam(optimizer_grouped_parameters,
    #                           lr=args.learning_rate,
    #                           bias_correction=False,
    #                           max_grad_norm=1.0)
    #     if args.loss_scale == 0:
    #         optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
    #     else:
    #         optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
    #     warmup_linear = WarmupLinearSchedule(warmup=args.warmup_proportion,
    #                                          t_total=num_train_optimization_steps)        

    # else:
    #     optimizer = BertAdam(optimizer_grouped_parameters,
    #                          lr=args.learning_rate,
    #                          warmup=args.warmup_proportion,
    #                          t_total=num_train_optimization_steps)

    ### PREPARE OPTIMIZER FOR MULTI TASKING
    if args.optim == 'normal':
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_parameters = [
                {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
                {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
                ]
        optimizer = BertAdam(optimizer_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=total_tr)
    else:
        no_decay = ['bias', 'gamma', 'beta']
        base = ['attn']
        optimizer_parameters = [
                {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and not any(nd in n for nd in base)], 'weight_decay_rate': 0.01},
                {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and not any(nd in n for nd in base)], 'weight_decay_rate': 0.0}
                ]
        optimizer = BertAdam(optimizer_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=total_tr)
        optimizer_parameters_mult = [
                {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and any(nd in n for nd in base)], 'weight_decay_rate': 0.01},
                {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and any(nd in n for nd in base)], 'weight_decay_rate': 0.0}
                ]
        optimizer_mult = BertAdam(optimizer_parameters_mult,
                             lr=3e-4,
                             warmup=args.warmup_proportion,
                             t_total=total_tr)


    ## moved do eval preparation from after do train to before train for multi-tasking
    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        
        ## for multit-tasking
        eval_loaders = []
        for i, task in enumerate(task_names):
          print('do eval for task ', i, ' ', task)
          print('label list ', label_list[i])
          eval_examples = processor_list[i].get_dev_examples(data_dirs[i])
          # ip_file = open("eval_ip_{0}.txt".format(i), "w")
          eval_features = convert_examples_to_features(eval_examples, label_list[i], args.max_seq_length, tokenizer, task)
          # ip_file.close()

          logger.info("***** Running evaluation *****")
          logger.info("  Num examples = %d", len(eval_examples))
          logger.info("  Batch size = %d", args.eval_batch_size)
          all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
          all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
          all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)

          all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)

          eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
          # Run prediction for full data
          if args.local_rank == -1:
                eval_sampler = SequentialSampler(eval_data)
          else:
                eval_sampler = DistributedSampler(eval_data)
          eval_loaders.append(DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size))


    global_step = 0
    nb_tr_steps = 0
    tr_loss = [0. for i in range(num_tasks)]

    if args.do_train:
        loaders = []
        logger.info("  Num Tasks = %d", len(train_examples))
        ## for multi-tasking
        for i, task in enumerate(task_names):
          print('train for task ', i, ' ', task)
          print('label list ', label_list[i])
          train_features = convert_examples_to_features(
              train_examples[i], label_list[i], args.max_seq_length, tokenizer)
          logger.info("***** Running training *****")
          print('For task: ', task)
          logger.info("  Num examples = %d", len(train_examples))
          logger.info("***** training data for %s *****", task)
          logger.info("  Data size = %d", len(train_features))


          all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
          all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
          all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)

          all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)

          train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
          if args.local_rank == -1:
              train_sampler = RandomSampler(train_data)
          else:
              train_sampler = DistributedSampler(train_data)
          # train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
          ## change for multi-tasking
          loaders.append(iter(DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)))



        ## 
        total_params = sum(p.numel() for p in model.parameters())
        logger.info("  Num param = {}".format(total_params))
        loaders = [cycle(it) for it in loaders]


        model.train()
        #print(model)

        ## for multi-tasking
        best_score = 0.
        if args.sample == 'sqrt' or args.sample == 'prop':
            probs = [306798, 284257]
            if args.sample == 'prop':
                alpha = 1.
            if args.sample == 'sqrt':
                alpha = 0.5
            probs = [p**alpha for p in probs]
            tot = sum(probs)
            probs = [p/tot for p in probs]
        task_id = 0
        epoch = 0

        for _ in trange(int(args.num_train_epochs), desc="Epoch"):

            ## for multi-tasking
            if args.sample == 'anneal':
                probs = [306798, 284257]
                alpha = 1. - 0.8 * epoch / (args.num_train_epochs - 1)
                probs = [p**alpha for p in probs]
                tot = sum(probs)
                probs = [p/tot for p in probs]

            # tr_loss = 0
            # nb_tr_examples, nb_tr_steps = 0, 0

            ## change for multi-tasking
           
            nb_tr_examples, nb_tr_steps = 0, 0


            # for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            ## changded for multi-tasking
            for step in range(steps_per_epoch):
                ## added for multi-tasking
                if args.sample != 'rr':
                    if step % args.gradient_accumulation_steps == 0:
                        task_id = np.random.choice(2, p=probs)
                else:
                    task_id = task_id % num_tasks
                batch = next(loaders[task_id])

                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch

                # define a new function to compute loss values for both output_modes
                # logits = model(input_ids, segment_ids, input_mask, labels=None)
                # #print(logits, logits.shape)

                # loss_fct = CrossEntropyLoss()
                # loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
                
                ##ADDED FOR MULTI TASKING
                loss, logits = model(input_ids, segment_ids, input_mask, task_id, task_names[task_id], label_ids)

                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                # if args.fp16:
                #     optimizer.backward(loss)
                # else:
                loss.backward()

                # tr_loss += loss.item()
                tr_loss[task_id] += loss.item()

                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                ## for multi-tasking
                if step % 1000 < num_tasks:
                    logger.info("Task: {}, Step: {}".format(task_id, step))
                    logger.info("Loss: {}".format(tr_loss[task_id]/nb_tr_steps))

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    # if args.fp16:
                    #     # modify learning rate with special warm up BERT uses
                    #     # if args.fp16 is False, BertAdam is used that handles this automatically
                    #     lr_this_step = args.learning_rate * warmup_linear.get_lr(global_step/num_train_optimization_steps,
                    #                                                              args.warmup_proportion)
                    #     for param_group in optimizer.param_groups:
                    #         param_group['lr'] = lr_this_step
                    optimizer.step()
                    
                    ##FOR MULTI TASKING
                    if args.optim != 'normal':
                        optimizer_mult.step()

                    # optimizer.zero_grad()
                    model.zero_grad()

                    global_step += 1

                    ##FOR MULTI TASKING
                    if not args.sample:
                        task_id += 1
            # print("Training loss: ", tr_loss, nb_tr_examples)

            ### added for multi-tasking
            epoch += 1
            ev_acc = 0.
            for i, task in enumerate(task_names):
                print('eval for task ', task)
                ev_acc += do_eval(model, logger, args, tr_loss[i], nb_tr_steps, global_step, processor_list[i], 
                                  label_list[i], tokenizer, eval_loaders[i], task, i)
            logger.info("Total acc: {}".format(ev_acc))
            if ev_acc > best_score:
                best_score = ev_acc
                model_dir = os.path.join(args.output_dir, "best_model.pth")
                torch.save(model.state_dict(), model_dir)
            logger.info("Best Total acc: {}".format(best_score))

        for i, task in enumerate(task_names):
            print("evaluation for task: ", task)
            eval_acc += do_eval(model, logger, args, tr_loss[i], nb_tr_steps, global_step, processor_list[i], 
                              label_list[i], tokenizer, eval_loaders[i], task, i)
        logger.info("Total acc: {}".format(eval_acc))  


## testing the fine-tuned model
    if args.do_predict and (args.local_rank == -1 or torch.distributed.get_rank() == 0):

        ## for multit-tasking
        test_loaders = []
        
        # Load a trained model and vocabulary that you have fine-tuned
        #model = BertForMultiTask.from_pretrained(args.output_dir, num_labels=num_labels)
        #tokenizer = BertTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        #model.to(device)
        
        model.eval()
        all_test_triples = []
        all_str_set = []
        label_ids_coll = []
        for i, task in enumerate(task_names):
          print('do test for task ', i, ' ', task)
        #if task == 'kg':
          test_examples = processor_list[i].get_test_examples(data_dirs[i])
          #else:
           # test_examples = processor_list[i].get_dev_examples(data_dirs[i])
          # ip_file = open("test_ip_{0}.txt".format(i), "w")
          test_features = convert_examples_to_features(
            test_examples, label_list[i], args.max_seq_length, tokenizer, task)
          # ip_file.close()
  
          test_triples = processor_list[i].get_test_triples(data_dirs[i])
          train_triples = processor_list[i].get_train_triples(data_dirs[i])
          #if (task == 'kg'):
          dev_triples = processor_list[i].get_dev_triples(data_dirs[i])
          #else:
           # dev_triples = []
        
          all_triples = train_triples + dev_triples + test_triples

      
          all_triples_str_set = set()
          for triple in all_triples:
            triple_str = '\t'.join(triple)
            all_triples_str_set.add(triple_str)

        ##eval_examples = processor.get_test_examples(args.data_dir)
        ##ip_file = open("test_ip.txt", "w")
        ##eval_features = convert_examples_to_features(
          ##  ip_file, eval_examples, label_list, args.max_seq_length, tokenizer)
        ##ip_file.close()

          logger.info("***** Running Prediction *****")
          logger.info("  Num examples = %d", len(test_examples))
          logger.info("  Batch size = %d", args.eval_batch_size)
          all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
          all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
          all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)

          all_label_ids = torch.tensor([f.label_id for f in test_features], dtype=torch.long)
          label_ids_coll.append(all_label_ids)

          # print('ip ids ', )
          
          all_test_triples.append(test_triples)
          all_str_set.append(all_triples_str_set)

         ## print('ip ids size ', all_label_ids.size())
          test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        
        # Run prediction for full data
          if args.local_rank == -1:
                test_sampler = SequentialSampler(test_data)
          else:
                test_sampler = DistributedSampler(test_data)
          test_loaders.append(DataLoader(test_data, sampler=test_sampler, batch_size=args.eval_batch_size))
        

        for i, task in enumerate(task_names):
            print('running test for ', task, i)
            test_acc = 0
            if task == 'kg':
              test_acc = 0
              test_acc += do_kg_test(model, logger, args, data_dirs[i], tr_loss[i], nb_tr_steps, global_step, processor_list[i], 
                                  label_list[i], tokenizer, test_loaders[i], task, i, label_ids_coll[i], all_test_triples[i], all_str_set[i], entity_list[i])
            
            else:
                # print('glue predictions')
                # glue predictions
                test_acc += do_glue_test(model, logger, args, tr_loss[i], nb_tr_steps, global_step,
                                  label_list[i], test_loaders[i], task, i, label_ids_coll[i])
            print('test accuracy for ', task, ': ', test_acc)
            

if __name__ == "__main__":
    main()  

    # if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
    #     # Save a trained model, configuration and tokenizer
    #     model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

    #     # If we save using the predefined names, we can load using `from_pretrained`
    #     output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
    #     output_config_file = os.path.join(args.output_dir, CONFIG_NAME)

    #     torch.save(model_to_save.state_dict(), output_model_file)
    #     model_to_save.config.to_json_file(output_config_file)
    #     tokenizer.save_vocabulary(args.output_dir)

    #     # Load a trained model and vocabulary that you have fine-tuned
    #     model = BertForSequenceClassification.from_pretrained(args.output_dir, num_labels=num_labels)
    #     tokenizer = BertTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
    # else:
    #     model = BertForSequenceClassification.from_pretrained(args.bert_model, num_labels=num_labels)
    # model.to(device)

    # if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        
    #     eval_examples = processor.get_dev_examples(args.data_dir)
    #     eval_features = convert_examples_to_features(
#     #         eval_examples, label_list, args.max_seq_length, tokenizer)
#     #     logger.info("***** Running evaluation *****")
#     #     logger.info("  Num examples = %d", len(eval_examples))
#     #     logger.info("  Batch size = %d", args.eval_batch_size)
#     #     all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
#     #     all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
#     #     all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)

#     #     all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)

#     #     eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
#     #     # Run prediction for full data
#     #     eval_sampler = SequentialSampler(eval_data)
#     #     eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
        
#     #     # Load a trained model and vocabulary that you have fine-tuned
#     #     model = BertForSequenceClassification.from_pretrained(args.output_dir, num_labels=num_labels)
#     #     tokenizer = BertTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
#     #     model.to(device)

#     #     model.eval()
#     #     eval_loss = 0
#     #     nb_eval_steps = 0
#     #     preds = []

#         for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
#             input_ids = input_ids.to(device)
#             input_mask = input_mask.to(device)
#             segment_ids = segment_ids.to(device)
#             label_ids = label_ids.to(device)

#             with torch.no_grad():
#                 logits = model(input_ids, segment_ids, input_mask, labels=None)

#             # create eval loss and other metric required by the task
#             loss_fct = CrossEntropyLoss()
#             tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
#             print(label_ids.view(-1))
            
#             eval_loss += tmp_eval_loss.mean().item()
#             nb_eval_steps += 1
#             if len(preds) == 0:
#                 preds.append(logits.detach().cpu().numpy())
#             else:
#                 preds[0] = np.append(
#                     preds[0], logits.detach().cpu().numpy(), axis=0)

#         eval_loss = eval_loss / nb_eval_steps
#         preds = preds[0]

#         preds = np.argmax(preds, axis=1)
#         result = compute_metrics(task_name, preds, all_label_ids.numpy())
#         loss = tr_loss/nb_tr_steps if args.do_train else None

#         result['eval_loss'] = eval_loss
#         result['global_step'] = global_step
#         result['loss'] = loss

#         output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
#         with open(output_eval_file, "w") as writer:
#             logger.info("***** Eval results *****")
#             for key in sorted(result.keys()):
#                 logger.info("  %s = %s", key, str(result[key]))
#                 writer.write("%s = %s\n" % (key, str(result[key])))

#     if args.do_predict and (args.local_rank == -1 or torch.distributed.get_rank() == 0):

#         train_triples = processor.get_train_triples(args.data_dir)
#         dev_triples = processor.get_dev_triples(args.data_dir)
#         test_triples = processor.get_test_triples(args.data_dir)
#         all_triples = train_triples + dev_triples + test_triples

#         all_triples_str_set = set()
#         for triple in all_triples:
#             triple_str = '\t'.join(triple)
#             all_triples_str_set.add(triple_str)

#         eval_examples = processor.get_test_examples(args.data_dir)
#         eval_features = convert_examples_to_features(
#             eval_examples, label_list, args.max_seq_length, tokenizer)
#         logger.info("***** Running Prediction *****")
#         logger.info("  Num examples = %d", len(eval_examples))
#         logger.info("  Batch size = %d", args.eval_batch_size)
#         all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
#         all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
#         all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)

#         all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)

#         eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
#         # Run prediction for full data
#         eval_sampler = SequentialSampler(eval_data)
#         eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
#         # Load a trained model and vocabulary that you have fine-tuned
#         model = BertForSequenceClassification.from_pretrained(args.output_dir, num_labels=num_labels)
#         tokenizer = BertTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
#         model.to(device)
#         model.eval()
#         eval_loss = 0
#         nb_eval_steps = 0
#         preds = []

#         for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Testing"):
#             input_ids = input_ids.to(device)
#             input_mask = input_mask.to(device)
#             segment_ids = segment_ids.to(device)
#             label_ids = label_ids.to(device)

#             with torch.no_grad():
#                 logits = model(input_ids, segment_ids, input_mask, labels=None)

#             loss_fct = CrossEntropyLoss()
#             tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
            
#             eval_loss += tmp_eval_loss.mean().item()
#             nb_eval_steps += 1
#             if len(preds) == 0:
#                 preds.append(logits.detach().cpu().numpy())
#             else:
#                 preds[0] = np.append(
#                     preds[0], logits.detach().cpu().numpy(), axis=0)

#         eval_loss = eval_loss / nb_eval_steps
#         preds = preds[0]
#         print(preds, preds.shape)
        
#         all_label_ids = all_label_ids.numpy()

#         preds = np.argmax(preds, axis=1)

#         result = compute_metrics(task_name, preds, all_label_ids)
#         loss = tr_loss/nb_tr_steps if args.do_train else None

#         result['eval_loss'] = eval_loss
#         result['global_step'] = global_step
#         result['loss'] = loss

#         output_eval_file = os.path.join(args.output_dir, "test_results.txt")
#         with open(output_eval_file, "w") as writer:
#             logger.info("***** Test results *****")
#             for key in sorted(result.keys()):
#                 logger.info("  %s = %s", key, str(result[key]))
#                 writer.write("%s = %s\n" % (key, str(result[key])))
#         print("Triple classification acc is : ")
#         print(metrics.accuracy_score(all_label_ids, preds))

#         # run link prediction
#         ranks = []
#         ranks_left = []
#         ranks_right = []

#         hits_left = []
#         hits_right = []
#         hits = []

#         top_ten_hit_count = 0

#         for i in range(10):
#             hits_left.append([])
#             hits_right.append([])
#             hits.append([])
#         '''
#         file_prefix = str(args.data_dir[7:])
#         f = open(file_prefix + '_ranks.txt','r')
#         lines = f.readlines()
#         for line in lines:
#             temp = line.strip().split()
#             rank1 = int(temp[0])
#             ranks_left.append(rank1+1)
#             print('left: ', rank1)
#             ranks.append(rank1+1)
#             if rank1 < 10:
#                 top_ten_hit_count += 1
#             rank2 = int(temp[1])
#             ranks.append(rank2+1)
#             ranks_right.append(rank2+1)
#             print('right: ', rank2)
#             print('mean rank until now: ', np.mean(ranks))
#             if rank2 < 10:
#                 top_ten_hit_count += 1
#             print("hit@10 until now: ", top_ten_hit_count * 1.0 / len(ranks))                
#             for hits_level in range(10):
#                 if rank1 <= hits_level:
#                     hits[hits_level].append(1.0)
#                     hits_left[hits_level].append(1.0)
#                 else:
#                     hits[hits_level].append(0.0)
#                     hits_left[hits_level].append(0.0)

#                 if rank2 <= hits_level:
#                     hits[hits_level].append(1.0)
#                     hits_right[hits_level].append(1.0)
#                 else:
#                     hits[hits_level].append(0.0)
#                     hits_right[hits_level].append(0.0)
    
#         '''
#         for test_triple in test_triples:
#             head = test_triple[0]
#             relation = test_triple[1]
#             tail = test_triple[2]
#             #print(test_triple, head, relation, tail)

#             head_corrupt_list = [test_triple]
#             for corrupt_ent in entity_list:
#                 if corrupt_ent != head:
#                     tmp_triple = [corrupt_ent, relation, tail]
#                     tmp_triple_str = '\t'.join(tmp_triple)
#                     if tmp_triple_str not in all_triples_str_set:
#                         # may be slow
#                         head_corrupt_list.append(tmp_triple)

#             tmp_examples = processor._create_examples(head_corrupt_list, "test", args.data_dir)
#             print(len(tmp_examples))
#             tmp_features = convert_examples_to_features(tmp_examples, label_list, args.max_seq_length, tokenizer, print_info = False)
#             all_input_ids = torch.tensor([f.input_ids for f in tmp_features], dtype=torch.long)
#             all_input_mask = torch.tensor([f.input_mask for f in tmp_features], dtype=torch.long)
#             all_segment_ids = torch.tensor([f.segment_ids for f in tmp_features], dtype=torch.long)
#             all_label_ids = torch.tensor([f.label_id for f in tmp_features], dtype=torch.long)

#             eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
#             # Run prediction for temp data
#             eval_sampler = SequentialSampler(eval_data)
#             eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
#             model.eval()

#             preds = []
            
#             for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Testing"):

#                 input_ids = input_ids.to(device)
#                 input_mask = input_mask.to(device)
#                 segment_ids = segment_ids.to(device)
#                 label_ids = label_ids.to(device)
                
#                 with torch.no_grad():
#                     logits = model(input_ids, segment_ids, input_mask, labels=None)
#                 if len(preds) == 0:
#                     batch_logits = logits.detach().cpu().numpy()
#                     preds.append(batch_logits)

#                 else:
#                     batch_logits = logits.detach().cpu().numpy()
#                     preds[0] = np.append(preds[0], batch_logits, axis=0)       

#             preds = preds[0]
#             # get the dimension corresponding to current label 1
#             #print(preds, preds.shape)
#             rel_values = preds[:, all_label_ids[0]]
#             rel_values = torch.tensor(rel_values)
#             #print(rel_values, rel_values.shape)
#             _, argsort1 = torch.sort(rel_values, descending=True)
#             #print(max_values)
#             #print(argsort1)
#             argsort1 = argsort1.cpu().numpy()
#             rank1 = np.where(argsort1 == 0)[0][0]
#             print('left: ', rank1)
#             ranks.append(rank1+1)
#             ranks_left.append(rank1+1)
#             if rank1 < 10:
#                 top_ten_hit_count += 1

#             tail_corrupt_list = [test_triple]
#             for corrupt_ent in entity_list:
#                 if corrupt_ent != tail:
#                     tmp_triple = [head, relation, corrupt_ent]
#                     tmp_triple_str = '\t'.join(tmp_triple)
#                     if tmp_triple_str not in all_triples_str_set:
#                         # may be slow
#                         tail_corrupt_list.append(tmp_triple)

#             tmp_examples = processor._create_examples(tail_corrupt_list, "test", args.data_dir)
#             #print(len(tmp_examples))
#             tmp_features = convert_examples_to_features(tmp_examples, label_list, args.max_seq_length, tokenizer, print_info = False)
#             all_input_ids = torch.tensor([f.input_ids for f in tmp_features], dtype=torch.long)
#             all_input_mask = torch.tensor([f.input_mask for f in tmp_features], dtype=torch.long)
#             all_segment_ids = torch.tensor([f.segment_ids for f in tmp_features], dtype=torch.long)
#             all_label_ids = torch.tensor([f.label_id for f in tmp_features], dtype=torch.long)

#             eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
#             # Run prediction for temp data
#             eval_sampler = SequentialSampler(eval_data)
#             eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
#             model.eval()
#             preds = []        

#             for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Testing"):
            
#                 input_ids = input_ids.to(device)
#                 input_mask = input_mask.to(device)
#                 segment_ids = segment_ids.to(device)
#                 label_ids = label_ids.to(device)
                
#                 with torch.no_grad():
#                     logits = model(input_ids, segment_ids, input_mask, labels=None)
#                 if len(preds) == 0:
#                     batch_logits = logits.detach().cpu().numpy()
#                     preds.append(batch_logits)

#                 else:
#                     batch_logits = logits.detach().cpu().numpy()
#                     preds[0] = np.append(preds[0], batch_logits, axis=0) 

#             preds = preds[0]
#             # get the dimension corresponding to current label 1
#             rel_values = preds[:, all_label_ids[0]]
#             rel_values = torch.tensor(rel_values)
#             _, argsort1 = torch.sort(rel_values, descending=True)
#             argsort1 = argsort1.cpu().numpy()
#             rank2 = np.where(argsort1 == 0)[0][0]
#             ranks.append(rank2+1)
#             ranks_right.append(rank2+1)
#             print('right: ', rank2)
#             print('mean rank until now: ', np.mean(ranks))
#             if rank2 < 10:
#                 top_ten_hit_count += 1
#             print("hit@10 until now: ", top_ten_hit_count * 1.0 / len(ranks))

#             file_prefix = str(args.data_dir[7:]) + "_" + str(args.train_batch_size) + "_" + str(args.learning_rate) + "_" + str(args.max_seq_length) + "_" + str(args.num_train_epochs)
#             #file_prefix = str(args.data_dir[7:])
#             f = open(file_prefix + '_ranks.txt','a')
#             f.write(str(rank1) + '\t' + str(rank2) + '\n')
#             f.close()
#             # this could be done more elegantly, but here you go
#             for hits_level in range(10):
#                 if rank1 <= hits_level:
#                     hits[hits_level].append(1.0)
#                     hits_left[hits_level].append(1.0)
#                 else:
#                     hits[hits_level].append(0.0)
#                     hits_left[hits_level].append(0.0)

#                 if rank2 <= hits_level:
#                     hits[hits_level].append(1.0)
#                     hits_right[hits_level].append(1.0)
#                 else:
#                     hits[hits_level].append(0.0)
#                     hits_right[hits_level].append(0.0)
    

#         for i in [0,2,9]:
#             logger.info('Hits left @{0}: {1}'.format(i+1, np.mean(hits_left[i])))
#             logger.info('Hits right @{0}: {1}'.format(i+1, np.mean(hits_right[i])))
#             logger.info('Hits @{0}: {1}'.format(i+1, np.mean(hits[i])))
#         logger.info('Mean rank left: {0}'.format(np.mean(ranks_left)))
#         logger.info('Mean rank right: {0}'.format(np.mean(ranks_right)))
#         logger.info('Mean rank: {0}'.format(np.mean(ranks)))
#         logger.info('Mean reciprocal rank left: {0}'.format(np.mean(1./np.array(ranks_left))))
#         logger.info('Mean reciprocal rank right: {0}'.format(np.mean(1./np.array(ranks_right))))
#         logger.info('Mean reciprocal rank: {0}'.format(np.mean(1./np.array(ranks))))            
              
# if __name__ == "__main__":
#     main()
