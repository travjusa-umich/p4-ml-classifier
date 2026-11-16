#include <iostream>
#include <string>
#include "csvstream.hpp"
#include <map>
using namespace std;


int main(int argc, char *argv[]) {

    string filename = argv[1];
    csvstream csvin(filename);

    map<string, string> row;

    cout << "training data: \n";

    int rowCount = 0;
    while (csvin >> row) {
        cout << "  label = " << row["tag"] 
        << ", content = " << row["content"] << "\n";
        count++;
    }
    cout << "trained on " << count << " examples \n";
    cout << "vocabulary size = " << <<;

}