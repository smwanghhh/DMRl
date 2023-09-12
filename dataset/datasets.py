import torch
import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader, TensorDataset
from transformers import BertTokenizer, XLNetTokenizer

class IEMCOAP_DATA(Dataset):
    def __init__(self, path, mode = 'train'):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        if mode == 'train':
            dataset = data['train']
        elif mode == 'valid':
            dataset = data['valid']
        else:
            dataset = data['test']

        text = dataset['text'].astype(np.float32)
        text[text == -np.inf] = 0
        self.text = torch.tensor(text)
        audio = dataset['audio'].astype(np.float32)
        audio[audio == -np.inf] = 0
        self.audio = torch.tensor(audio)
        vision = dataset['vision'].astype(np.float32)
        self.vision = torch.tensor(vision)
        self.label = dataset['labels'].astype(np.float32)   #happy, sad, angry, neutral

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = self.text[index]
        vision = self.vision[index]
        audio = self.audio[index]
        label = torch.argmax(torch.tensor(self.label[index]), -1)
        return text, audio, vision, label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, visual, acoustic, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.visual = visual
        self.acoustic = acoustic
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

class MOSI_DATA(object):
    def __init__(self,args) -> None:
        with open(f"path to dataset.pkl", "rb") as handle:
            data = pickle.load(handle)
        
        super().__init__()
        self.args = args
        self.train_data =  data['train']
        self.dev_data = data['dev']
        self.test_data = data['test']



    def get_tokenizer(self,model):
        if model == "bert-base-uncased":
            return BertTokenizer.from_pretrained(model)
        elif model == "xlnet-base-cased":
            return XLNetTokenizer.from_pretrained(model)
        else:
            raise ValueError(
                "Expected 'bert-base-uncased' or 'xlnet-base-cased, but received {}".format(
                    model
                )
            )
    def prepare_bert_input(self, tokens, visual, acoustic, tokenizer):
        CLS = tokenizer.cls_token
        SEP = tokenizer.sep_token
        tokens = [CLS] + tokens + [SEP]

        # Pad zero vectors for acoustic / visual vectors to account for [CLS] / [SEP] tokens
        acoustic_zero = np.zeros((1, self.args.ACOUS_DIM))
        acoustic = np.concatenate((acoustic_zero, acoustic, acoustic_zero))
        visual_zero = np.zeros((1, self.args.VISUAL_DIM))
        visual = np.concatenate((visual_zero, visual, visual_zero))

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        segment_ids = [0] * len(input_ids)
        input_mask = [1] * len(input_ids)

        pad_length = self.args.max_seq_length - len(input_ids)

        acoustic_padding = np.zeros((pad_length, self.args.ACOUS_DIM))
        acoustic = np.concatenate((acoustic, acoustic_padding))

        visual_padding = np.zeros((pad_length, self.args.VISUAL_DIM))
        visual = np.concatenate((visual, visual_padding))

        padding = [0] * pad_length

        # Pad inputs
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        return input_ids, visual, acoustic, input_mask, segment_ids



    def conver_to_features(self,data,max_seq_length,tokenizer):
        features = []

        for (ex_index, example) in enumerate(data):

            (words, visual, acoustic), label_id, segment = example

            tokens, inversions = [], []
            for idx, word in enumerate(words):
                tokenized = tokenizer.tokenize(word)
                tokens.extend(tokenized)
                inversions.extend([idx] * len(tokenized))

            # Check inversion
            assert len(tokens) == len(inversions)

            aligned_visual = []
            aligned_audio = []

            for inv_idx in inversions:
                aligned_visual.append(visual[inv_idx, :])
                aligned_audio.append(acoustic[inv_idx, :])

            visual = np.array(aligned_visual)
            acoustic = np.array(aligned_audio)

            # Truncate input if necessary
            if len(tokens) > max_seq_length - 2:
                tokens = tokens[: max_seq_length - 2]
                acoustic = acoustic[: max_seq_length - 2]
                visual = visual[: max_seq_length - 2]

            if self.args.model == "bert-base-uncased":
                prepare_input = self.prepare_bert_input
            elif self.args.model == "xlnet-base-cased":
                prepare_input = prepare_xlnet_input

            input_ids, visual, acoustic, input_mask, segment_ids = prepare_input(
                tokens, visual, acoustic, tokenizer
            )

            # Check input length
            max_seq_length = self.args.max_seq_length
            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert acoustic.shape[0] == max_seq_length
            assert visual.shape[0] == max_seq_length

            features.append(
                InputFeatures(
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    visual=visual,
                    acoustic=acoustic,
                    label_id=label_id,
                )
            )
        return features


    def get_dataset(self,data):
        self.tokenizer = self.get_tokenizer(self.args.model)
        features = self.conver_to_features(data, self.args.max_seq_length, self.tokenizer)
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)

        all_visual = torch.tensor([f.visual for f in features], dtype=torch.float)
        all_acoustic = torch.tensor([f.acoustic for f in features], dtype=torch.float)
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

        return TensorDataset(
            all_input_ids, all_visual, all_acoustic, \
            all_input_mask, all_segment_ids, all_label_ids
        )

        


    def create_dataloader(self):

        train_dataset = self.get_dataset(self.train_data)
        dev_dataset = self.get_dataset(self.dev_data)
        test_dataset = self.get_dataset(self.test_data)
    

        return DataLoader(train_dataset,batch_size=self.args.train_batch_size,shuffle=True),\
                DataLoader(dev_dataset,batch_size=self.args.dev_batch_size,shuffle=True),\
                DataLoader(test_dataset,batch_size=self.args.test_batch_size,shuffle=True)


if __name__ == '__main__':
    data_train = IEMCOAP_DATA(path = 'path to dataset',mode='test')
    print(data_train.__len__())

