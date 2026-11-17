#include <iostream>
#include <string>
#include "csvstream.hpp"
#include <map>
#include <set>
#include <cmath>
using namespace std;


// EFFECTS: Return a set of unique whitespace delimited words
set<string> unique_words(const string &str) {
  istringstream source(str);
  set<string> words;
  string word;
  while (source >> word) {
    words.insert(word);
  }
  return words;
}

class Classifier {

private:
  int rowCount = 0;
  set<string> words;
  map<string, string> row;
  map<string, int> labelsCount;
  map<string, int> word_counts;
  map<string, map<string, int>> labels_word_counts;

public:

  void train(csvstream &csvin) {

    while (csvin >> row) {
          cout << "  label = " << row["tag"] 
          << ", content = " << row["content"] << "\n";
          rowCount++;

          set<string> currentWords = unique_words(row["content"]);

          for (const string &word : currentWords) {
              words.insert(word);
              ++word_counts[word];
              ++labels_word_counts[row["tag"]][word];

          }

          ++labelsCount[row["tag"]];

      }

  }

  int get_words_size() {
    return words.size();
  }

  int get_rowCount() {
    return rowCount;
  }

  void print_classes() const {
    for (const auto &m : labelsCount) {
      cout << "  " << m.first << ", " << m.second << " examples, " 
      << "log-prior = ";
      double altCount = rowCount;
      cout << std::log(m.second / altCount) << endl;
    }
  }

  void print_classifier_parameters() const {
    for (const auto &m : labels_word_counts) {
      for (const auto &m2 : m.second) {
        cout << "  " << m.first << ":";
        cout << m2.first << ", " << "count = " << m2.second 
        << ", log-likelihood = ";
        double altCount2 = m2.second;
        cout << std::log(altCount2 / labelsCount.at(m.first)) << endl;

      }
    }
  }

};


int main(int argc, char *argv[]) {

  cout.precision(3);
  string filename = argv[1];
  csvstream csvin(filename); // opens the csvstream

  Classifier classifier;

  cout << "training data:\n";

  classifier.train(csvin);
  
  cout << "trained on " << classifier.get_rowCount() << " examples\n";
  cout << "vocabulary size = " << classifier.get_words_size() << endl;

  cout << endl << "classes:\n";
  classifier.print_classes();

  cout << "classifier parameters:\n";

  classifier.print_classifier_parameters();
  cout << endl;



}