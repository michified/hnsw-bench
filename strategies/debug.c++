#include <bits/stdc++.h>
#pragma GCC optimize("Ofast,unroll-loops")
using namespace std;

int dist(vector<int>& a, vector<int>& b) {
    int tot = 0;
    for (int i = 0; i < a.size(); ++i) {
        tot += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return tot;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n = 60000, q = 10000, d = 784;
    vector<vector<int>> data(n, vector<int>(d));
    ifstream cin("data/dataset.txt");
    for (int i = 0; i < n; ++i) { 
        for (int j = 0; j < d; ++j) {
            cin >> data[i][j];
        }
    }
    vector<vector<int>> queries(q, vector<int>(d));
    ifstream cin2("data/queries.txt");
    for (int i = 0; i < q; ++i) {
        for (int j = 0; j < d; ++j) {
            cin2 >> queries[i][j];
        }
    }
    vector<vector<int>> groundTruths(q, vector<int>(100));
    ifstream cin3("data/topk.txt");
    for (int i = 0; i < q; ++i) {
        for (int j = 0; j < 100; ++j) {
            cin3 >> groundTruths[i][j];
        }
    }
    cout << dist(queries[0], data[21394]) << endl;
    cout << dist(queries[1], data[17401]) << endl;
    cout << dist(queries[2], data[10865]) << endl;
    cout << dist(queries[3], data[1379]) << endl;
    cout << dist(queries[4], data[51074]) << endl;
    return 0;
}