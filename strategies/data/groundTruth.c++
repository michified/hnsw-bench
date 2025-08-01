#include <bits/stdc++.h>
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
    ifstream cin("dataset.txt");
    for (int i = 0; i < n; ++i) { 
        for (int j = 0; j < d; ++j) {
            cin >> data[i][j];
        }
    }
    vector<vector<int>> queries(q, vector<int>(d));
    ifstream cin2("queries.txt");
    for (int i = 0; i < q; ++i) {
        for (int j = 0; j < d; ++j) {
            cin2 >> queries[i][j];
        }
    }
    priority_queue<pair<int, int>> pq;
    ofstream cout("topk.txt");
    for (int i = 0; i < q; ++i) {
        for (int j = 0; j < n; ++j) {
            pq.push({dist(queries[i], data[j]), j});
            if (pq.size() > 100) {
                pq.pop();
            }
        }
        vector<int> topk;
        for (int k = 0; k < 100; k++) {
            topk.push_back(pq.top().second);
            pq.pop();
        }
        for (int k = 99; k >= 0; k--) {
            cout << topk[k] << " ";
        }
        if (i % 100 == 0) cout << endl;
        else cout << "\n";
    }
    return 0;
}