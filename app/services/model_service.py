import spacy
from spacy.lang.pt.stop_words import STOP_WORDS
import pandas as pd
import string
import random
from spacy.training import Example


class ModelService:
    def __init__(self):
        self.pontuacoes = string.punctuation
        self.stop_words = STOP_WORDS
        self.pln = spacy.load("pt_core_news_sm")
        self.modelo = None

    def preprocessamento(self, texto):
        texto = texto.lower()
        documento = self.pln(texto)

        lista = [token.lemma_ for token in documento if
                 token.text not in self.stop_words and token.text not in self.pontuacoes]
        lista = ' '.join([str(elemento) for elemento in lista if not elemento.isdigit()])

        return lista

    @staticmethod
    def carregar_base_dados():
        base_dados = pd.read_csv('app/database/base_treinamento.txt', encoding='utf-8')
        base_dados_final = []

        for texto, emocao in zip(base_dados['texto'], base_dados['emocao']):
            dic = ({'ALEGRIA': emocao == 'alegria', 'MEDO': emocao == 'medo'})
            base_dados_final.append([texto, dic])

        return base_dados_final

    def treinar_modelo(self):
        base_dados_final = self.carregar_base_dados()
        modelo = spacy.blank('pt')
        textcat = modelo.add_pipe("textcat")
        textcat.add_label("ALEGRIA")
        textcat.add_label("MEDO")
        historico = []

        modelo.begin_training()
        for epoca in range(1000):
            random.shuffle(base_dados_final)
            losses = {}
            for batch in spacy.util.minibatch(base_dados_final, 30):
                textos = [modelo(texto) for texto, entities in batch]
                annotations = [{'cats': entities} for texto, entities in batch]
                examples = [Example.from_dict(doc, annotation) for doc, annotation in zip(
                    textos, annotations
                )]
                modelo.update(examples, losses=losses)
            if epoca % 100 == 0:
                print(losses)
                historico.append(losses)
        modelo.to_disk("app/modelo")

    def carregar_modelo(self):
        self.modelo = spacy.load("app/modelo")

    def usar_modelo(self, texto):
        if not self.modelo:
            self.carregar_modelo()

        texto_preprocessado = self.preprocessamento(texto)
        previsao = self.modelo(texto_preprocessado)
        payload = {"texto": texto,
                   "Previsão": previsao.cats}
        return payload
