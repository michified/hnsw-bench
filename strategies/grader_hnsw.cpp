#include "hnsw_concurrent.cpp"
#include <bits/stdc++.h>
#pragma GCC optimize("Ofast,unroll-loops")
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    srand(time(0));
    int n = 60000, d = 784, q = 10000, k = 100;
    vector<vector<int>> vals(n, vector<int>(d)), queries(q, vector<int>(d)), groundTruths(q, vector<int>(100));
    ifstream cin("data/dataset.txt");
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < d; j++) {
            cin >> vals[i][j];
        }
    }
    cin.close();
    ifstream cin2("data/queries.txt");
    for (int i = 0; i < q; ++i) {
        for (int j = 0; j < d; j++) {
            cin2 >> queries[i][j];
        }
    }
    cin2.close();
    ifstream cin3("data/topk.txt");
    for (int i = 0; i < q; ++i) {
        for (int j = 0; j < 100; j++) {
            cin3 >> groundTruths[i][j];
        }
    }
    cin3.close();
    auto start = chrono::high_resolution_clock::now();
    HNSW hnsw(vals, (int) floor(log2(n)) / 2, (int) floor(log2(n)) / 2);
    auto build_end = chrono::high_resolution_clock::now();
    chrono::duration<double> build_elapsed = build_end - start;
    double total_query_time = 0.0;
    double total_recall = 0;
    for (int i = 0; i < q; ++i) {
        auto query_start = chrono::high_resolution_clock::now();
        auto res = hnsw.findClosestTo(queries[i], 0, k)[0];
        auto query_end = chrono::high_resolution_clock::now();
        chrono::duration<double, micro> query_elapsed = query_end - query_start;
        total_query_time += query_elapsed.count();
        set<int> groundTruthsK;
        for (int j = 0; j < k; j++) {
            groundTruthsK.insert(groundTruths[i][j]);
        }
        int overlapping = 0;
        for (auto& [dist, idx] : res) {
            if (groundTruthsK.count(idx)) {
                overlapping++;
            }
        }
        total_recall += (overlapping / (double) k) * 100;
    }
    cout << "Build time: " << build_elapsed.count() << " seconds" << endl;
    cout << "Average query time: " << (total_query_time / q) << " microseconds" << endl;
    cout << "Queries per second: " << (q / (total_query_time / 1e6)) << endl;
    cout << "Average recall: " << (total_recall / q) << "%" << endl;
    return 0;
}