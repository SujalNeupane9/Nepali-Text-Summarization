
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
import math
from dataclasses import dataclass

nltk.download('stopwords')
nltk.download('punkt')

@dataclass
class Nepali_summarizer:
  text:list = None

  def text_preprocessing(self,sentences):
    """
    Preprocessing text to remove unnecessary words.
    """
    stop_words = set(stopwords.words('nepali'))

    clean_sentences = []
    for sentence in sentences:
        # Remove punctuation
        translator = str.maketrans("", "", string.punctuation)
        sentence_no_punct = sentence.translate(translator)

        # Tokenize sentence into words
        words = sentence_no_punct.split()

        # Remove stop words
        words_no_stop = [word for word in words if word.lower() not in stop_words]

        # Join words to reconstruct sentence
        clean_sentence = ' '.join(words_no_stop)
        clean_sentences.append(clean_sentence)

    return clean_sentences

  def create_tf_matrix(self,sentences):
    preprocessed_sentences = self.text_preprocessing(sentences)
    tf_matrix = list()

    for sentence in preprocessed_sentences:
      words = sentence.split()
      total_words = len(words)
      words_frequency = {}

      for word in words:
        if word not in words_frequency:
          words_frequency[word] = 0
        words_frequency[word] += 1/total_words
      
      tf_matrix.append(words_frequency)
    return tf_matrix

  def create_idf_matrix(self,tf_matrix):
    unique_words = set(word for tf_dict in tf_matrix for word in tf_dict)

    doc_frequency = {word:sum(word in tf_dict for tf_dict in tf_matrix) for word in unique_words}

    total_docs = len(tf_matrix)
    idf_matrix = {word:math.log(total_docs/(1+doc_freq)) for word, doc_freq in doc_frequency.items()}
    return idf_matrix

  def create_tfidf_matrix(self,sentences):
    tf_mat = self.create_tf_matrix(sentences)
    idf_mat = self.create_idf_matrix(tf_mat)

    tfidf_matrix = list()

    for tf_dict in tf_mat:
      tfidf_dict = {}
      for word, tf_score in tf_dict.items():
        tfidf_dict[word] = tf_score * idf_mat.get(word,0)
      tfidf_matrix.append(tfidf_dict)
    
    return tfidf_matrix

  def calculate_sentence_scores(self,tfidf_matrix):
    sentence_scores = list()

    for tfidf_dict in tfidf_matrix:
      total_score = sum(tfidf_score for tfidf_score in tfidf_dict.values())
      distinct_words = len(tfidf_dict)

      if distinct_words != 0:
        avg_score = total_score/distinct_words
      else:
        avg_score = 0
      sentence_scores.append(avg_score)
    
    return sentence_scores

  def calculate_average_score(self, sentence_scores):
    total_score = sum(sentence_scores)
    num_sentences = len(sentence_scores)

    if num_sentences != 0:
      average_score = total_score / num_sentences
    else:
      average_score = 0
    
    return average_score

  def generate_summary(self,sentences):
    tfidf_matrix = self.create_tfidf_matrix(text)
    sentence_scores = self.calculate_sentence_scores(create_tfidf_matrix(text))
    average_score = self.calculate_average_score(sentence_scores)

    threshold = average_score

    summary_sentences = [sentences[i] for i, score in enumerate(sentence_scores) if score > threshold]
    summary = ' '.join(summary_sentences)

    return summary
