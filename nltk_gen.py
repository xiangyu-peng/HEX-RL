import spacy
from pattern.en import conjugate
import argparse
class nltk_gen(object):
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')

    def tense_detect(self, prompt):
        """
        Check the tense of verb
        """
        tokens = self.nlp(prompt)
        for token in tokens:
            if 'VBD' == token.tag_:
                return 'past'
        return 'present'

    def verb_tense_gen(self, verb, tense, person, number):
        """

        :param verb: does not care whether it is lemma.
        :param tense: 'present', "past"
        :param person: 1 or 2 or 3
        :param number: "plural" or "singular"
        :return:
        """
        if len(verb.split(' ')) == 1:
            return conjugate(verb=verb,
                             tense=tense,    # INFINITIVE, PRESENT, PAST, FUTURE
                             person=person,  # 1, 2, 3 or None
                             number=number)  # SG, PL
        else:
            tokens = self.nlp(verb)
            res = ''
            for token in tokens:
                res = res + ' ' if res != '' else ''
                if 'V' in token.tag_:
                    res += conjugate(verb=token.lemma_,
                                     tense=tense,  # INFINITIVE, PRESENT, PAST, FUTURE
                                     person=person,  # 1, 2, 3 or None
                                     number=number)  # SG, PL
                else:
                    res += token.text
            return res

    def nounFind(self, prompt):
        tokens = self.nlp(prompt)
        res = []
        for token in tokens:
            if 'NN' in token.tag_:
                res.append(token.lemma_)
        return res if res else [prompt]

    def nounFind_calm(self, prompt):
        dirs = [
            "north",
            "south",
            "east",
            "west",
            "southeast",
            "southwest",
            "northeast",
            "northwest",
            "up",
            "down",
        ]
        tokens = self.nlp(prompt)
        res = []
        for token in tokens:
            # print(token.lemma_, str(token.lemma_) in dirs)
            if 'NN' in token.tag_ and token.lemma_ not in dirs:
                res.append(token.lemma_)
        return res if res else None

    def find_verb_lemma(self, prompt):
        # print('prompt', prompt)
        tokens = self.nlp(prompt)
        for token in tokens:
            if 'V' in token.tag_:
                return token.lemma_
        return None

    def find_verb_phrase(self, prompt):
        if prompt.split(' ')[0] == 'to':
            prompt = ' '.join(prompt.split(' ')[1:])
        return prompt
    def noun_checker(self, phrase=None):
        tokens = self.nlp(phrase)
        for token in tokens:
            if 'NN' in token.tag_:
                return True
        return False

    def adj_checker(self, word):
        tokens = self.nlp(word)
        for token in tokens:
            if 'J' in token.tag_:
                return True
        return False

    def noun_adj_adv_find(self, words, prompt):
        """
        Try to find the noun, adj and adv for verbatlas outputs
        :param prompt: sentence
        :return:
        """
        tokens = self.nlp(prompt)
        res = []
        prev = ('', '')
        outputs = []
        for token in tokens:
            if 'NN' in token.tag_:
                if 'NN' not in prev[-1] and token.lemma_ in words:  # noun 前面不是noun 比如 beautiful girl
                    res.append(token.lemma_)
                elif 'NN' in prev[-1] and token.lemma_ in words:  # noun phrase
                    res[-1] += ' ' + token.lemma_
                else:
                    pass
            if 'JJ' in prev[-1] or 'RB' in prev[-1]:
                if 'NN' in token.tag_:
                    outputs.append((token.lemma_, prev[0]))
                elif 'NN' not in token.tag_ and token.lemma_ in words:
                    res.append(token.lemma_)

            prev = (token.lemma_, token.tag_)

        return res, outputs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--example",
                        type=str,
                        default='apple',
                        help="a obj")
    args = parser.parse_args()
    nltk_gen = nltk_gen()
    # prompt = args.example
    prompt = 'Jenny heard earthquake about Florida'
    # print(nltk_gen.verb_tense_gen('loves', "past", 3, "singular"))
    # print(nltk_gen.nounFind(prompt))
    # print(nltk_gen.adj_checker('animate'))
    # print(nltk_gen.noun_adj_adv_find(words='a piece of paper', prompt='a piece of paper'))
    print(nltk_gen.nounFind_calm(prompt='east'))