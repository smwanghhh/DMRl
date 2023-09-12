



def get_tokenizer(model):
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
    
class MultimodalConfig(object):
    def __init__(self, dropout_prob, emb_size, dim_t, dim_a, dim_v, seqlength, mode):
        self.dropout_prob = dropout_prob
        self.emb_size = emb_size
        self.dim_t = dim_t
        self.dim_a = dim_a
        self.dim_v = dim_v
        self.seqlength = seqlength
        self.mode = mode


