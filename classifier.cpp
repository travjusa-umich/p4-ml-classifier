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

// A Naive Bayes classifier that learns from labeled posts and
// predicts the most likely label for new posts using a
// bag-of-words model.
class Classifier {

private:
  int rowCount = 0;  // total number of training posts
  set<string> words;
  map<string, int> labelsCount;
  map<string, int> word_counts;
  map<string, map<string, int>> labels_word_counts;

public:

  // REQUIRES: csvin is open and has "tag" and "content" columns
  // MODIFIES: this classifier's counts, cout (if print_data is true)
  // EFFECTS:  Reads every post from csvin and records the counts
  //           needed for prediction. If print_data is true, prints
  //           each post's label and content.
  void train(csvstream &csvin, bool print_data) {

    map<string, string> row;

    while (csvin >> row) {
      if (print_data) {
          cout << "  label = " << row["tag"] 
          << ", content = " << row["content"] << "\n";
      }
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

  // REQUIRES: label was seen during training
  // EFFECTS:  Returns the log-likelihood of word given label.
  //           Uses the overall word frequency if word never appeared
  //           with label, and 1 / total posts if word never appeared
  //           in training at all.
  double log_likelihood(const string &label, 
                              const string &word) const 
  {
    const map<string, int> &label_words = labels_word_counts.at(label);

    if (label_words.find(word) != label_words.end()) {

      double wordCount = labels_word_counts.at(label).at(word);
      return std::log(wordCount / labelsCount.at(label));

    } else if (word_counts.find(word) != word_counts.end()) {

        double overallWordCount = word_counts.at(word);
        return std::log(overallWordCount / rowCount);

    } else {

        return std::log(1.0 / rowCount);
    }
  }

  // REQUIRES: label was seen during training
  // EFFECTS:  Returns ln(posts with label / total posts).
  double log_prior(const string& label) const {
    double dblRowCount = rowCount;
    return std::log(labelsCount.at(label) / dblRowCount);
  }

  // EFFECTS: Returns the number of unique words in the training data.
  int get_words_size() const {
    return words.size();
  }

  // EFFECTS: Returns the number of training posts.
  int get_rowCount() const {
    return rowCount;
  }

  // MODIFIES: cout
  // EFFECTS:  Prints each label with its post count and log-prior,
  //           in alphabetical order.
  void print_classes() const {
    for (const auto &label_map : labelsCount) {
      cout << "  " << label_map.first << ", " 
      << label_map.second << " examples, " 
      << "log-prior = ";
      double dblRowCount = rowCount;
      cout << std::log(label_map.second / dblRowCount) << endl;
    }
  }

  // MODIFIES: cout
  // EFFECTS:  Prints each label and word pair with its count and
  //           log-likelihood, in alphabetical order.
  void print_classifier_parameters() const {
    for (const auto &label : labels_word_counts) {
      for (const auto &word : label.second) {
        cout << "  " << label.first << ":";
        cout << word.first << ", " << "count = " << word.second 
        << ", log-likelihood = ";
        cout << log_likelihood(label.first, word.first) << endl;

      }
    }
  }

  // REQUIRES: train() has been called
  // EFFECTS:  Returns the label with the highest log-probability
  //           score for content, along with that score. Ties go to
  //           the alphabetically first label.
  pair<string, double> predict(const string& content) const {

    set<string> words = unique_words(content);
    string best_label = "";
    double best_score = 0;

    for (const auto &label_entry : labelsCount) {
      double score = log_prior(label_entry.first);
      for (const string &word : words) {
        score += log_likelihood(label_entry.first, word);
      }
      if (best_label == "" || score > best_score) {
        best_label = label_entry.first;
        best_score = score;
      }
    }
    return {best_label, best_score};
  }

  // MODIFIES: cout
  // EFFECTS:  Prints the vocabulary size, classes, and classifier
  //           parameters for train-only mode.
  void print_train_info() const {
    cout << "vocabulary size = " << get_words_size() << "\n\n";
    cout << "classes:\n";
    print_classes();
    cout << "classifier parameters:\n";
    print_classifier_parameters();
    cout << "\n";
  }

};


// REQUIRES: classifier has been trained, csvin_test is open
// MODIFIES: cout
// EFFECTS:  Predicts a label for each test post, then prints the
//           results and overall performance.
void run_tests(const Classifier &classifier, csvstream &csvin_test) {
  cout << "\ntest data:\n";
  map<string, string> row;
  int performance_counter = 0;
  int test_counter = 0;
  while (csvin_test >> row) {
    pair<string, double> prediction = classifier.predict(row["content"]);
    cout << "  correct = " << row["tag"] << ", predicted = " << prediction.first
         << ", log-probability score = " << prediction.second << "\n"
         << "  content = " << row["content"] << "\n\n";
    ++test_counter;
    if (row["tag"] == prediction.first) {
      ++performance_counter;
    }
  }
  cout << "performance: " << performance_counter << " / "
       << test_counter << " posts predicted correctly\n";
}

int main(int argc, char *argv[]) {
  cout.precision(3);

  if (argc != 2 && argc != 3) {
    cout << "Usage: classifier.exe TRAIN_FILE [TEST_FILE]\n";
    return 1;
  }

  bool train_only = (argc == 2);
  string train_file = argv[1];

  // Check the test file before printing anything
  if (!train_only) {
    try {
      csvstream test_check(argv[2]);
    } catch (const csvstream_exception &e) {
      cout << "Error opening file: " << argv[2] << "\n";
      return 1;
    }
  }

  Classifier classifier;
  try {
    csvstream csvin(train_file);
    if (train_only) {
      cout << "training data:\n";
    }
    classifier.train(csvin, train_only);
  } catch (const csvstream_exception &e) {
    cout << "Error opening file: " << train_file << "\n";
    return 1;
  }

  cout << "trained on " << classifier.get_rowCount() << " examples\n";

  if (train_only) {
    classifier.print_train_info();
    return 0;
  }

  string test_file = argv[2];
  try {
    csvstream csvin_test(test_file);
    run_tests(classifier, csvin_test);
  } catch (const csvstream_exception &e) {
    cout << "Error opening file: " << test_file << "\n";
    return 1;
  }
  return 0;
}