import numpy as np
from gensim.models import Word2Vec
from transformers import AutoModel, BertTokenizer, FeatureExtractionPipeline
import torch

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class FeatureTransformer:
    def __init__(self, w2v_model_path, bert_model_paths, tokenizer_path, dimm=16):
        self.dimm = dimm
        # self.model_w2c = Word2Vec.load(w2v_model_path)
        # self.tokenizer = BertTokenizer(vocab_file=tokenizer_path)

        # self.std_bert = self._get_std(bert_model_paths['bert'])
        # self.std_xlm = self._get_std(bert_model_paths['xlm'])
        # self.std_deberta = self._get_std(bert_model_paths['deberta'])
        # self.std_PubChem10M = self._get_std(bert_model_paths['PubChem10M'])

    def _get_std(self, model_path):
        model = AutoModel.from_pretrained(model_path)
        bert_model = FeatureExtractionPipeline(model=model, tokenizer=self.tokenizer,device='cuda:0')
        std = {
            "A": 'Nc1ncnc2[nH]cnc12',
            "T": "CC1=CNC(=O)NC1=O",
            "C": "C1=C(NC(=O)N=C1)N",
            "G": "O=C1c2ncnc2nc(N)N1"
        }
        A = self._wins(np.array(bert_model(" ".join(list(std["A"]))))[:, 0, :].flatten())
        T = self._wins(np.array(bert_model(" ".join(list(std["T"]))))[:, 0, :].flatten())
        C = self._wins(np.array(bert_model(" ".join(list(std["C"]))))[:, 0, :].flatten())
        G = self._wins(np.array(bert_model(" ".join(list(std["G"]))))[:, 0, :].flatten())
        return {"A": A, "T": T, "C": C, "G": G}

    # def _wins(self, vec):
    #     return vec[:self.dimm]

    def NCP(self, seq):
        std = {"A": [1, 1, 1], "T": [0, 0, 1], "C": [0, 1, 0], "G": [1, 0, 0]}
        return np.array([std[x] for x in seq])

    # def word2vec(self, seq):
    #     return np.array([self.model_w2c.wv[x] for x in seq])

    def binary(self, seq):
        std = {"A": [1, 0, 0, 0], "T": [0, 1, 0, 0], "C": [0, 0, 1, 0], "G": [0, 0, 0, 1]}
        return np.array([std[x] for x in seq])

    def EIIP(self, seq):
        std = {"A": [0.12601, 0, 0, 0], "T": [0, 0.13400, 0, 0], "C": [0, 0, 0.08060, 0], "G": [0, 0, 0, 0.13350]}
        return np.array([std[x] for x in seq])

    def NAC(self, seq):
        std = ["A", "T", "C", "G"]
        res5 = np.array([seq.count(x) / len(seq) for x in std])
        nacseq = []
        for i in seq:
            if i=="A":
                nacseq.append(res5[0])
            if i=="T":
                nacseq.append(res5[1])
            if i=="C":
                nacseq.append(res5[2])
            if i=="G":
                nacseq.append(res5[3])
        return np.array(nacseq).reshape(-1,1)
    def PCP(self, seq):
        std = {
            "A": [37.03, 83.8, 279.9, 122.7, 14.68],
            "T": [29.71, 102.7, 251.3, 35.7, 11.77],
            "C": [27.30, 71.5, 206.3, 69.2, 10.82],
            "G": [35.46, 68.8, 229.6, 124.0, 14.06]
        }
        res = np.zeros((len(seq), 5))
        for i, x in enumerate(seq):
            res[i] = std[x]
        return res

    # def bert_encoding(self, seq):
    #     res = np.zeros((len(seq), self.dimm))
    #     for i, x in enumerate(seq):
    #         res[i] = self.std_bert[x]
    #     return res
    #
    # def xlm_encoding(self, seq):
    #     res = np.zeros((len(seq), self.dimm))
    #     for i, x in enumerate(seq):
    #         res[i] = self.std_xlm[x]
    #     return res
    #
    # def deberta_encoding(self, seq):
    #     res = np.zeros((len(seq), self.dimm))
    #     for i, x in enumerate(seq):
    #         res[i] = self.std_deberta[x]
    #     return res
    #
    # def PubChem10M_encoding(self, seq):
    #     res = np.zeros((len(seq), self.dimm))
    #     for i, x in enumerate(seq):
    #         res[i] = self.std_PubChem10M[x]
    #     return res


    def get_features(self, seq):
        EIIP = self.EIIP(seq)#4
        # res2 = self.binary(seq)#1kmer
        NCP = self.NCP(seq)#3 化学
        # res4 = self.word2vec(seq)
        NAC= self.NAC(seq)#密度信息 1
        PCP = self.PCP(seq)#5
        # res7 = self.bert_encoding(seq)
        # res8 = self.xlm_encoding(seq)
        # res9 = self.deberta_encoding(seq)
        # res10 = self.PubChem10M_encoding(seq)
        res = np.concatenate([EIIP,NCP,NAC,PCP], axis=1)
        # res = np.concatenate([EIIP], axis=1)
        return res
        # return np.concatenate([res1,res5,res6], axis=1)#.flatten()
        # return np.concatenate([res2], axis=1)#.flatten()

def process_dna_sequences(sequences, transformer):
    """
    Processes a list of DNA sequences to extract feature encodings.

    Parameters:
    - sequences: List of DNA sequences (each sequence is a string).
    - transformer: An instance of FeatureTransformer used for feature extraction.

    Returns:
    - A numpy array of shape (n_samples, feature_dim) containing feature encodings.
    """
    features = []
    for seq in sequences:
        features.append(transformer.get_features(seq))
    return np.array(features)


if __name__ == "__main__":
    # Paths to models and tokenizer
    bert_model_paths = {
        'bert': "./lib/bert_model/4mC-bert-base-cased",
        'xlm': "./lib/bert_model/4mC-xlm-roberta-base",
        'deberta': "./lib/bert_model/4mC-deberta-base",
        'PubChem10M': "./lib/bert_model/4mC-PubChem10M-SMILES-BPE-450k"
    }

    # Initialize FeatureTransformer
    featureTransformer = FeatureTransformer(
        w2v_model_path="./lib/w2c/4mC-word2vec.model",
        bert_model_paths=bert_model_paths,
        tokenizer_path='./lib/smile_token/vocab.txt'
    )

    # Example usage
    sequences = ["ATCGCGGG"*40, "GCTAAAA", "TTAA"]
    feature_encodings = process_dna_sequences(sequences, featureTransformer)
    print(feature_encodings)
