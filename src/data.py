"""
Modification of the original data.py file from the original repository
https://github.com/tonyzhaozh/few-shot-learning/blob/main/data_utils.py

"""


import pandas as pd
import json
import numpy as np
from tqdm import tqdm

def load_sst2(root_dir):
    def process_raw_data_sst(lines):
        """from lines in dataset to two lists of sentences and labels respectively"""
        labels = []
        sentences = []
        for line in lines:
            labels.append(int(line[0]))
            sentences.append(line[2:].strip())
        return sentences, labels

    with open(f"{root_dir}/data/sst2/stsa.binary.train", "r") as f:
        train_lines = f.readlines()
    with open(f"{root_dir}/data/sst2/stsa.binary.test", "r") as f:
        test_lines = f.readlines()
    train_sentences, train_labels = process_raw_data_sst(train_lines)
    test_sentences, test_labels = process_raw_data_sst(test_lines)

    data = {
        'train_sentences': train_sentences,
        'train_labels': train_labels,
        'test_sentences': test_sentences,
        'test_labels': test_labels
    }
    return data

def load_agnews(root_dir):
    train_data = pd.read_csv(f'{root_dir}/data/agnews/train.csv')
    test_data = pd.read_csv(f'{root_dir}/data/agnews/test.csv')

    train_sentences = train_data['Title'] + ". " + train_data['Description']
    train_sentences = list(
        [item.replace(' #39;s', '\'s').replace(' quot;', "\"").replace('\\', " ").replace(' #39;ll', "'ll") for item
         in train_sentences]) # some basic cleaning
    train_labels = list(train_data['Class Index'])
    test_sentences = test_data['Title'] + ". " + test_data['Description']
    test_sentences = list(
        [item.replace(' #39;s', '\'s').replace(' quot;', "\"").replace('\\', " ").replace(' #39;ll', "'ll") for item
         in test_sentences]) # some basic cleaning
    test_labels = list(test_data['Class Index']) 
    train_labels = [l - 1 for l in train_labels] # make them 0, 1, 2, 3 instead of 1, 2, 3, 4
    test_labels = [l - 1 for l in test_labels]

    data = {
        'train_sentences': train_sentences,
        'train_labels': train_labels,
        'test_sentences': test_sentences,
        'test_labels': test_labels
    }
    return data

def load_trec(root_dir):
    inv_label_dict = {'NUM': 0, 'LOC': 1, 'HUM': 2, 'DESC': 3, 'ENTY': 4, 'ABBR': 5}
    train_sentences = []
    train_labels = []
    with open(f'{root_dir}/data/trec/train.txt', 'r') as train_data:
        for line in train_data:
            train_label = line.split(' ')[0].split(':')[0]
            train_label = inv_label_dict[train_label]
            train_sentence = ' '.join(line.split(' ')[1:]).strip()
            # basic cleaning
            train_sentence = train_sentence.replace(" 's", "'s").replace('`` ', '"').replace(" ''",'"').replace(' ?','?').replace(' ,',',')
            train_labels.append(train_label)
            train_sentences.append(train_sentence)

    test_sentences = []
    test_labels = []
    with open(f'{root_dir}/data/trec/test.txt', 'r') as test_data:
        for line in test_data:
            test_label = line.split(' ')[0].split(':')[0]
            test_label = inv_label_dict[test_label]
            test_sentence = ' '.join(line.split(' ')[1:]).strip()
            test_sentence = test_sentence.replace(" 's", "'s").replace('`` ', '"').replace(" ''",'"').replace(' ?','?').replace(' ,',',')
            test_labels.append(test_label)
            test_sentences.append(test_sentence)
    
    data = {
        'train_sentences': train_sentences,
        'train_labels': train_labels,
        'test_sentences': test_sentences,
        'test_labels': test_labels
    }
    return data

def get_cb(root_dir):
    train_questions = []
    train_answers = []
    with open(f"{root_dir}/data/cb/train.jsonl", "r") as f:
        for line in f:
            myjson = json.loads(line)
            q = myjson['hypothesis']
            p = myjson['premise']
            curr_label = myjson['label']
            if curr_label == 'contradiction':
                train_answers.append(0)
            elif curr_label == 'neutral':
                train_answers.append(1)
            elif curr_label == 'entailment':
                train_answers.append(2)
            # being a bit lazy here. We put the "question: " into the input and treat it like single sentence classification.
            train_questions.append(p.strip() + '\n' + 'question: ' + q + '. true, false, or neither?')

    test_questions = []
    test_answers = []
    with open(f"{root_dir}/data/cb/val.jsonl", "r") as f:
        for line in f:
            myjson = json.loads(line)
            q = myjson['hypothesis']
            p = myjson['premise']
            if myjson['label'] == 'contradiction':
                test_answers.append(0)
            elif myjson['label'] == 'neutral':
                test_answers.append(1)
            elif myjson['label'] == 'entailment':
                test_answers.append(2)
            else:
                exit('answer')
            test_questions.append(p.strip() + '\n' + 'question: ' + q + '. true, false, or neither?')

    data = {
        'train_sentences': train_questions,
        'train_labels': train_answers,
        'test_sentences': test_questions,
        'test_labels': test_answers
    }
    return data

def load_dbpedia(root_dir):
    train_data = pd.read_csv(f'{root_dir}/data/dbpedia/train_subset.csv')
    test_data = pd.read_csv(f'{root_dir}/data/dbpedia/test.csv')

    train_sentences = train_data['Text']
    train_sentences = list([item.replace('""', '"') for item in train_sentences])
    train_labels = list(train_data['Class'])

    test_sentences = test_data['Text']
    test_sentences = list([item.replace('""', '"') for item in test_sentences])
    test_labels = list(test_data['Class'])
    
    train_labels = [l - 1 for l in train_labels] # make them 0, 1, 2, 3 instead of 1, 2, 3, 4...
    test_labels = [l - 1 for l in test_labels]

    data = {
        'train_sentences': train_sentences,
        'train_labels': train_labels,
        'test_sentences': test_sentences,
        'test_labels': test_labels
    }
    return data

def load_rte(root_dir):
    train_questions = []
    train_answers = []
    with open(f"{root_dir}/data/rte/train.jsonl", "r") as f:
        for line in f:
            myjson = json.loads(line)
            q = myjson['hypothesis']
            p = myjson['premise']
            if myjson['label'] == 'not_entailment':
                train_answers.append(0)
            elif myjson['label'] == 'entailment':
                train_answers.append(1)
            else:
                exit('answer')
            train_questions.append(p + '\n' + 'question: ' + q + ' True or False?')

    test_questions = []
    test_answers = []
    with open(f"{root_dir}/data/rte/val.jsonl", "r") as f:
        for line in f:
            myjson = json.loads(line)
            q = myjson['hypothesis']
            p = myjson['premise']
            if myjson['label'] == 'not_entailment':
                test_answers.append(0)
            elif myjson['label'] == 'entailment':
                test_answers.append(1)
            else:
                exit('answer')
            test_questions.append(p + '\n' + 'question: ' + q + ' True or False?')

    data = {
        'train_sentences': train_questions,
        'train_labels': train_answers,
        'test_sentences': test_questions,
        'test_labels': test_answers
    }
    return data


class ClassificationDataset:

    def __init__(self,root_dir,dataset="agnews",n_shot=2,random_state=None):
        self.root_dir = root_dir
        self.dataset = dataset
        self._load_data(dataset)
        self.n_shot = n_shot
        self.prompt_func = None
        
        # Remove the prompt shots from the training set
        self._rs = np.random.RandomState(random_state)
        train_shots_idx = self._rs.permutation(len(self._data["train_sentences"]))[:self.n_shot]
        all_train_sentences = self._data['train_sentences']
        all_train_labels = self._data['train_labels']
        new_train_sentences = []
        new_train_labels = []
        prompt_shots_sentences = []
        prompt_shots_labels = []
        for idx, (sentence, label) in enumerate(zip(all_train_sentences, all_train_labels)):
            if idx in train_shots_idx:
                prompt_shots_sentences.append(sentence)
                prompt_shots_labels.append(label)
            else:
                new_train_sentences.append(sentence)
                new_train_labels.append(label)
        self._data['train_sentences'] = new_train_sentences
        self._data['train_labels'] = new_train_labels
        self.prompt_shots_sentences = prompt_shots_sentences
        self.prompt_shots_labels = prompt_shots_labels


    def _load_data(self,dataset="agnews"):
        if dataset == "agnews":
            self._data = load_agnews(self.root_dir)
            self.prompt_prefix = "Classify the news articles into the categories of World, Sports, Business, and Technology.\n\n"
            self.q_prefix = "Article: "
            self.a_prefix = "Answer: "
            self.label_dict = {0: ['World'], 1: ['Sports'], 2: ['Business'], 3: ['Technology', 'Science']}
            self.inv_label_dict = {'World': 0, 'Sports': 1, 'Business': 2, 'Technology': 3, 'Science': 3} # notice index start from 1 here
            self.num_user_input = None
            self.task_format = 'classification'
            self.num_tokens_to_predict = 1

        elif dataset == "trec":
            self._data = load_trec(self.root_dir)
            self.prompt_prefix = "Classify the questions based on whether their answer type is a Number, Location, Person, Description, Entity, or Abbreviation.\n\n"
            self.q_prefix = "Question: "
            self.a_prefix = "Answer Type: "
            self.label_dict = {0: ['Number'], 1: ['Location'], 2: ['Person'], 3: ['Description'], 4: ['Entity'], 5: ['Ab']}
            self.inv_label_dict = {'Number': 0, 'Location': 1, 'Person': 2, 'Description': 3, 'Entity': 4, 'Ab': 5}
            self.num_user_input = None
            self.task_format = 'classification'
            self.num_tokens_to_predict = 1

        elif dataset == "cb":
            self._data = get_cb(self.root_dir)
            self.prompt_prefix = ""
            self.q_prefix = ""
            self.a_prefix = "answer: "
            self.label_dict = {0: ['false'], 1: ['neither'], 2: ['true']}
            self.inv_label_dict = {'false': 0, 'neither': 1, 'true': 2}
            self.num_user_input = None
            self.task_format = 'classification'
            self.num_tokens_to_predict = 1

        elif dataset == "rte":
            self._data = load_rte(self.root_dir)
            self.prompt_prefix = ""
            self.q_prefix = " "
            self.a_prefix = "answer: "
            self.label_dict = {0: ['False'], 1: ['True']}
            self.inv_label_dict = {'False': 0, 'True': 1}
            self.num_user_input = 2
            self.task_format = 'classification'
            self.num_tokens_to_predict = 1

        elif dataset == "sst2":
            self._data = load_sst2(self.root_dir)
            self.prompt_prefix = ""
            self.q_prefix = "Review: "
            self.a_prefix = "Sentiment: "
            self.label_dict = {0: ['Negative'], 1: ['Positive']}
            self.inv_label_dict = {'Negative': 0, 'Positive': 1}
            self.num_user_input = None
            self.task_format = 'classification'
            self.num_tokens_to_predict = 1

        elif self.dataset == "dbpedia":
            self._data = load_dbpedia(self.root_dir)
            self.prompt_prefix = "Classify the documents based on whether they are about a Company, School, Artist, Athlete, Politician, Transportation, Building, Nature, Village, Animal, Plant, Album, Film, or Book.\n\n"
            self.q_prefix = "Article: "
            self.a_prefix = "Answer: "
            self.label_dict = {0: ['Company'], 1: ['School'], 2: ['Artist'], 3: ['Ath'], 4: ['Polit'], 5: ['Transportation'], 6: ['Building'], 7: ['Nature'], 8: ['Village'], 9: ['Animal'], 10: ['Plant'], 11: ['Album'], 12: ['Film'], 13: ['Book']}
            self.inv_label_dict = {'Company': 0, 'School': 1, 'Artist': 2, 'Ath': 3, 'Polit': 4, 'Transportation': 5, 'Building': 6, 'Nature': 7, 'Village': 8, 'Animal': 9, 'Plant': 10, 'Album': 11, 'Film': 12, 'Book': 13}
            self.num_user_input = None
            self.task_format = 'classification'
            self.num_tokens_to_predict = 1
        else:
            raise NotImplementedError
        
    def construct_prompt(self, train_sentences, train_labels, test_sentence, prompt_func=None):
        """construct a single prompt to be fed into the model"""
        # special case when the user defines a custom prompt function. 
        if prompt_func is not None:
            return prompt_func(
                train_sentences, 
                train_labels, 
                test_sentence,
                prompt_prefix=self.prompt_prefix,
                q_prefix=self.q_prefix,
                a_prefix=self.a_prefix,
                label_dict=self.label_dict,
                inv_label_dict=self.inv_label_dict,
                num_user_input=self.num_user_input,
                task_format=self.task_format,
                num_tokens_to_predict=self.num_tokens_to_predict
            )

        # take the prompt template and fill in the training and test example
        MAX_CHARS_PER_SENTENCE = 2500 // (len(train_sentences) + 1) # 2500 is the max number of tokens allowed in the model multiplied by 2.5 (appox, chars per token)
        prompt = self.prompt_prefix
        q_prefix = self.q_prefix
        a_prefix = self.a_prefix
        for s, l in zip(train_sentences, train_labels):
            prompt += q_prefix
            prompt += s[:MAX_CHARS_PER_SENTENCE] + "\n"
            if isinstance(l, int) or isinstance(l, np.int32) or isinstance(l, np.int64): # integer labels for classification
                assert self.task_format == 'classification'
                l_str = self.label_dict[l][0] if isinstance(self.label_dict[l], list) else self.label_dict[l]
            else:
                assert isinstance(l, str) # string labels
                assert self.task_format == 'qa'
                l_str = l

            prompt += a_prefix
            prompt += l_str + "\n\n"

        prompt += q_prefix
        prompt += test_sentence[:MAX_CHARS_PER_SENTENCE] + "\n"
        assert a_prefix[-1] == ' '
        prompt += a_prefix[:-1] # GPT models do not want a trailing space, so we cut off -1
        return prompt
    
    def sample(self,idx,split="test"):

        if split == "test":
            all_sentences = self._data['test_sentences']
            all_labels = self._data['test_labels']
        elif split == "dev":
            ## TODO: implement dev
            raise NotImplementedError
        elif split == "train":
            all_sentences = self._data['train_sentences']
            all_labels = self._data['train_labels']
        
        prompt = self.construct_prompt_with_train_shots(all_sentences[idx], prompt_func=self.prompt_func)
        return {
            'prompt': prompt,
            'label': all_labels[idx]
        }
    
    def construct_prompt_with_train_shots(self, sentence, prompt_func=None):
        prompt = self.construct_prompt(self.prompt_shots_sentences, self.prompt_shots_labels, sentence, prompt_func=prompt_func)
        return prompt

    def batch_iter(self,split="test",num_eval=100,batch_size=32):
        if split == "test":
            num_samples = len(self._data["test_sentences"])
        elif split == "dev":
            num_samples = len(self._data["dev_sentences"])
        elif split == "train":
            num_samples = len(self._data["train_sentences"])

        if num_eval is None:
            num_eval = num_samples
        else:
            num_eval = min(num_eval,num_samples)
        test_idx = self._rs.permutation(num_samples)[:num_eval]

        for i in tqdm(range(0, num_eval, batch_size),total=num_eval//batch_size):
            batch_idx = test_idx[i:i+batch_size]
            batch = {'prompt': [], 'label': []}
            for idx in batch_idx:
                sample = self.sample(idx,split)
                batch['prompt'].append(sample['prompt'])
                batch['label'].append(sample['label'])
            yield batch