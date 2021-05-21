# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 20:30:37 2020

@author: axalo
"""

from random import randint, choice
from sklearn import tree
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import graphviz

class Create:
  blank_data = pd.DataFrame(columns=['Gender', 'Skill', 'Academics', 'Experience', 'Ambition'])
  total_applicants = 0
  hires = 0
  def __init__(self):
    print('This is an algorithm designed to generate a synthetic dataset and showcase the effects of training bias.')
    Create.total_applicants = int(input('Number of Total Applicants: '))
    Create.hires = int(input('Number of hires: '))
    print(' ')
  def biased_dataset(self, biased_data):
    print('Creating the biased dataset...')
    def define_ratios():
        go = True
        while go:
          try:
            male_ratio = int(input('Percentage of the dataset that is MALE (ex. 70): '))
            skill_skew = input('Are MALE or FEMALE stats going to be more appealing to an employer: ').lower()
            if male_ratio < 100 and male_ratio > 0 and bool(skill_skew == 'male' or skill_skew == 'female'):
              go = False
            else:
              raise TypeError
          except TypeError: 
            print('Error. Try Again.')
        return [male_ratio, skill_skew]
    def generate_cv(ratios):
      cv = None
      def generate_stats(gender, a, b):
        stats = [gender,None,None,None,None]
        for stat_loc in range(1, len(stats)):
          stats[stat_loc] = randint(a, b)
        return stats
      if ratios[1] == 'male':
        if randint(1, 100) < ratios[0]:
          cv = generate_stats(0, 4, 9)
        else:
          cv = generate_stats(1, 2, 6)
      else:
        if randint(1, 100) < ratios[0]:
          cv = generate_stats(1, 4, 9)
        else:
          cv = generate_stats(0, 2, 6)
      return cv

    ratios = define_ratios()
    for applicant in range(Create.total_applicants):
      cv = generate_cv(ratios)
      biased_data = biased_data.append({'Gender': cv[0], 'Skill': cv[1], 'Academics': cv[2], 'Experience': cv[3], 'Ambition': cv[4]}, ignore_index = True)
    target=functions.label_accepted(biased_data, Create.hires)
    print('Done. \n')
    return biased_data, [biased_data.values, target]
  def unbiased_dataset(self, unbiased_data):
    print('Creating unbiased dataset...')
    for applicant in range(Create.total_applicants):
      unbiased_data = unbiased_data.append({'Gender': choice([0, 1]), 'Skill': randint(2, 9), 'Academics': randint(2,9), 'Experience': randint(2,9), 'Ambition': randint(2,9)}, ignore_index = True)
    target=functions.label_accepted(unbiased_data, Create.hires)
    print('Done. \n')
    return unbiased_data, [unbiased_data.values, target]

class Functions:
  def label_accepted(self, data, hires):
    score = []
    accepted = []
    for row in data.itertuples():
        avg = sum([row.Skill, row.Academics, row.Experience, row.Ambition]) / 4
        score.append(avg)
    sorted_score = score
    sorted_score.sort()
    accepted_scores = sorted_score[(hires * -1):]
    for row in data.itertuples():
        avg = sum([row.Skill, row.Academics, row.Experience, row.Ambition]) / 4
        if accepted.count('Accepted') <= hires and avg in accepted_scores:
            accepted.append('Accepted')
        else:
            accepted.append('Rejected')
    return accepted
  def train(self, X_train, X_test, y_train, y_test):
      print(' ')
      print('Training algorithm... \n')
      classifier = tree.DecisionTreeClassifier(max_depth=5)
      classifier = classifier.fit(X_train, y_train)
      dot_data = tree.export_graphviz(classifier, out_file=None, impurity=False) 
      graphviz.Source(dot_data)
      return [classifier, X_test, y_test, y_train]
  def test(self, classifier, X_test, y_test, y_train):
      print('Testing algorithm... \n')
      y_predict = classifier.predict(X_test)
      print('Algorithm tested. Results obtained. /n')
      return y_predict, X_test, y_test, y_train

class Graph:
  def bar_graph(self, accepted_rejected_data, df, graph_type):
    df['Accepted_Rejected'] = accepted_rejected_data
    accepted_men, rejected_men, accepted_women, rejected_women = (0,0,0,0)
    for row in df.itertuples():
      if row.Gender == 0 and row.Accepted_Rejected == 'Accepted':
        accepted_men += 1
      elif row.Gender == 0 and row.Accepted_Rejected == 'Rejected':
        rejected_men += 1
      elif row.Gender == 1 and row.Accepted_Rejected == 'Accepted':
        accepted_women += 1
      elif row.Gender == 1 and row.Accepted_Rejected == 'Rejected':
        rejected_women += 1
      else:
        raise TypeError
    N = 2
    men_data = [accepted_men, rejected_men]
    women_data = [accepted_women, rejected_women]
    width = .25
    ind = np.arange(N)       
    plt.bar(ind, men_data, width, label='Men')
    plt.bar(ind + width, women_data, width,
        label='Women')
    plt.ylabel('Number of People')
    plt.title('Accepted/Rejected Applicants by Gender \n' + graph_type)
    plt.xticks(ind + width / 2, ('Accepted', 'Rejected'))
    plt.legend(loc='best')
    plt.show()
    
def run():
    template = create.blank_data
    biased_dataframe, bias_data = create.biased_dataset(template)
    unbiased_dataframe, no_bias_data = create.unbiased_dataset(template)
    choice_print = input('Print generated datasets? (Y/N): ').upper()
    if choice_print == 'Y':
      print('Biased data: \n \n', biased_dataframe, '\n')
      print('Unbiased data: \n \n', unbiased_dataframe, '\n')
    prep = True
    test_vars = []
    while prep:
      training_data = input('Should a biased dataset be used to train the algorithm? (Y/N): ').upper()
      if training_data == 'Y':
        prep = False
        test_vars=functions.train(bias_data[0], no_bias_data[0], bias_data[1], no_bias_data[1])
      elif training_data == 'N':
        prep = False
        test_vars=functions.train(no_bias_data[0], bias_data[0], no_bias_data[1], bias_data[1])
      else:
        print('Error')
    y_predict, X_test, y_test, y_train = functions.test(test_vars[0], test_vars[1], test_vars[2], test_vars[3])
    print('')
    print('Analyzing results... \n')
    print("The first graph shows the ideal, unbiased hiring decisions. The second shows the algorithm's decisions. \n")
    graph.bar_graph(y_train, biased_dataframe, "Algorithm's Training Data (Biased)")
    graph.bar_graph(y_test, unbiased_dataframe, 'Pre-Generated Unbiased Hiring')
    graph.bar_graph(y_predict, unbiased_dataframe, 'Algorithm Generated Hiring')

create = Create()
functions = Functions()
graph = Graph()
run()