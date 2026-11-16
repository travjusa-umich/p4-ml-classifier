#include <iostream>
#include <string>
#include "csvstream.hpp"
#include <map>
#include <set>
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


int main(int argc, char *argv[]) {

    string filename = argv[1];
    csvstream csvin(filename);

    map<string, string> row;

    cout << "training data: \n";

    set<string> words;
    int rowCount = 0;
    while (csvin >> row) {
        cout << "  label = " << row["tag"] 
        << ", content = " << row["content"] << "\n";
        rowCount++;

        set<string> currentWords = unique_words(row["content"]);

        for (const string &word : currentWords) {
            words.insert(word);
        }
    }
    cout << "trained on " << rowCount << " examples \n";
    cout << "vocabulary size = " << words.size() << endl;

}