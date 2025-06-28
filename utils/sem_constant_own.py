
"""
Define constants.
"""
EMB_INIT_RANGE = 1.0
MAX_LEN = 100
INFINITY_NUMBER = 1e12

# vocab
PAD_TOKEN = '<PAD>'
PAD_ID = 0
UNK_TOKEN = '<UNK>'
UNK_ID = 1

VOCAB_PREFIX = [PAD_TOKEN, UNK_TOKEN]

SUBJ_NER_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'OTHER': 2, 'TITLE': 3, 'CAUSE_OF_DEATH': 4, 'CRIMINAL_CHARGE': 5, 'DATE': 6, 'MISC': 7, 'MONEY': 8, 'DURATION': 9, 'PERSON': 10, 'NUMBER': 11, 'IDEOLOGY': 12, 'COUNTRY': 13, 'TIME': 14, 'NATIONALITY': 15, 'ORGANIZATION': 16, 'STATE_OR_PROVINCE': 17, 'LOCATION': 18, 'SET': 19, 'RELIGION': 20}
OBJ_NER_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'OTHER': 2, 'TITLE': 3, 'CAUSE_OF_DEATH': 4, 'DATE': 5, 'CRIMINAL_CHARGE': 6, 'DURATION': 7, 'IDEOLOGY': 8, 'TIME': 9, 'SET': 10, 'MISC': 11, 'LOCATION': 12, 'PERSON': 13, 'RELIGION': 14, 'ORDINAL': 15, 'NUMBER': 16, 'COUNTRY': 17, 'NATIONALITY': 18}
NER_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'O': 2, 'OTHER': 3, 'TITLE': 4, 'CAUSE_OF_DEATH': 5, 'CRIMINAL_CHARGE': 6, 'DATE': 7, 'DURATION': 8, 'IDEOLOGY': 9, 'MISC': 10, 'TIME': 11, 'SET': 12, 'LOCATION': 13, 'PERSON': 14, 'MONEY': 15, 'NUMBER': 16, 'COUNTRY': 17, 'RELIGION': 18, 'NATIONALITY': 19, 'ORGANIZATION': 20, 'ORDINAL': 21, 'STATE_OR_PROVINCE': 22}
POS_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'NN': 2, 'DT': 3, 'IN': 4, 'JJ': 5, 'NNS': 6, '.': 7, 'VBN': 8, 'NNP': 9, 'VBD': 10, ',': 11, 'VBZ': 12, 'CC': 13, 'RB': 14, 'VBG': 15, 'TO': 16, 'PRP': 17, 'VBP': 18, 'VB': 19, 'CD': 20, 'PRP$': 21, 'WDT': 22, 'POS': 23, 'RP': 24, 'WRB': 25, 'WP': 26, 'JJR': 27, '-LRB-': 28, '-RRB-': 29, 'MD': 30, 'JJS': 31, ':': 32, '``': 33, '\'\'': 34, 'NNPS': 35, 'RBR': 36, 'EX': 37, 'RBS': 38, 'PDT': 39, 'FW': 40, '$': 41, 'WP$': 42, 'UH': 43, 'SYM': 44, 'LS': 45, '#': 46}
DEPREL_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'det': 2, 'case': 3, 'nmod': 4, 'punct': 5, 'amod': 6, 'nsubj': 7, 'ROOT': 8, 'compound': 9, 'dobj': 10, 'conj': 11, 'cc': 12, 'advmod': 13, 'mark': 14, 'nmod:poss': 15, 'auxpass': 16, 'nsubjpass': 17, 'acl': 18, 'cop': 19, 'aux': 20, 'advcl': 21, 'acl:relcl': 22, 'xcomp': 23, 'nummod': 24, 'dep': 25, 'ccomp': 26, 'appos': 27, 'compound:prt': 28, 'mwe': 29, 'neg': 30, 'nmod:tmod': 31, 'nmod:npmod': 32, '': 33, 'root': 34, 'parataxis': 35, 'expl': 36, 'det:predet': 37, 'csubj': 38, 'cc:preconj': 39, 'iobj': 40, 'csubjpass': 41, 'discourse': 42}
NEGATIVE_LABEL = 'Other'
LABEL_TO_ID = {'Other': 0, 'Entity-Destination(e1,e2)': 1, 'Cause-Effect(e1,e2)': 2, 'Member-Collection(e1,e2)': 3, 'Entity-Origin(e1,e2)': 4, 'Message-Topic(e1,e2)': 5, 'Component-Whole(e1,e2)': 6, 'Instrument-Agency(e1,e2)': 7, 'Product-Producer(e1,e2)': 8, 'Content-Container(e1,e2)': 9, 'Entity-Destination(e2,e1)': 10, 'Cause-Effect(e2,e1)': 11, 'Member-Collection(e2,e1)': 12, 'Entity-Origin(e2,e1)': 13, 'Message-Topic(e2,e1)': 14, 'Component-Whole(e2,e1)': 15, 'Instrument-Agency(e2,e1)': 16, 'Product-Producer(e2,e1)': 17, 'Content-Container(e2,e1)': 18}
# LABEL_TO_ID = {'Other': 0, 'Entity-Destination': 1, 'Cause-Effect': 2, 'Member-Collection': 3, 'Entity-Origin': 4, 'Message-Topic': 5, 'Component-Whole': 6, 'Instrument-Agency': 7, 'Product-Producer': 8, 'Content-Container': 9}
