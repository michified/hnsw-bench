#include <vector>
#include <unordered_map>
#include <stack>
#include <set>
#include <shared_mutex>
#include <utility>
#include <algorithm>
#include <thread>
#include <cstdlib>  // for rand()
using namespace std;

struct HNSWNode {
    int height, index;
    vector<int> val;
    vector<unordered_map<int, HNSWNode*>> neighbors;
    HNSWNode(vector<int> val, int height, int index) : val(val), height(height), index(index) {
        neighbors.resize(height + 1);
    }
};

class HNSW {
public:
    vector<HNSWNode*> nodes;
    int maxHeight, k;
    mutable shared_mutex nodes_mutex; // Add a shared_mutex for nodes

    HNSW(vector<vector<int>>& vals, int maxHeight, int k) : maxHeight(maxHeight), k(k) {
        int threads = thread::hardware_concurrency();
        HNSWNode* firstNode = new HNSWNode(vals[0], maxHeight, 0);
        nodes.push_back(firstNode);
        for (int i = 1; i < vals.size(); i += threads) {
            int start = i, end = min(i + threads, (int) vals.size());
            addBatchNodes(vals, start, end);
        }
    }

    int randomHeight() {
        int height = 0;
        while (rand() % 2 == 0 and height < maxHeight) {
            height++;
        }
        return height;
    }

    int dist(HNSWNode* a, HNSWNode* b) {
        int sum = 0.0;
        for (size_t i = 0; i < a->val.size(); ++i) {
            sum += (a->val[i] - b->val[i]) * (a->val[i] - b->val[i]);
        }
        return sum;
    }

    int dist(HNSWNode* node, vector<int>& val) {
        int sum = 0.0;
        for (size_t i = 0; i < node->val.size(); ++i) {
            sum += (node->val[i] - val[i]) * (node->val[i] - val[i]);
        }
        return sum;
    }

    void addBatchNodes(vector<vector<int>>& vals, int start, int end) {
        int threads = thread::hardware_concurrency();
        int stop = min(threads, (int) vals.size() - start);
        vector<vector<set<pair<int, int>>>> batchClosestK(stop);
        vector<thread> threadPool;
        vector<int> batchHeights(stop);
        for (int i = 0; i < stop; ++i) {
            batchHeights[i] = randomHeight();
            int j = start + i;
            threadPool.emplace_back([this, &vals, &batchClosestK, &batchHeights, i, j]() {
                batchClosestK[i] = this->findClosestTo(vals[j], batchHeights[i], k);
            });
        }
        for (int i = 0; i < stop; ++i) {
            threadPool[i].join();
        }
        for (int i = 0; i < stop; ++i) {
            auto& closestK = batchClosestK[i];
            int curHeight = batchHeights[i];
            HNSWNode* newNode = new HNSWNode(vals[start + i], curHeight, start + i);
            for (int h = 0; h <= curHeight; ++h) {
                for (auto& [d, idx] : closestK[h]) {
                    HNSWNode* neighbor = nodes[idx];
                    newNode->neighbors[h][neighbor->index] = neighbor;
                    if (h <= neighbor->neighbors.size()) neighbor->neighbors[h][newNode->index] = newNode;
                }
            }
            nodes.push_back(newNode);
        }
    }

    vector<set<pair<int, int>>> findClosestTo(vector<int>& val, int height, int topk) {
        shared_lock<shared_mutex> lock(nodes_mutex); // Lock nodes for reading
        vector<set<pair<int, int>>> ret(height + 1);
        set<pair<int, int>> result;
        stack<HNSWNode*> st;
        vector<int> dists(nodes.size(), -1);
        result.insert({dist(nodes[0], val), 0});
        for (int h = maxHeight; h >= 0; h--) {
            for (auto& [d, idx] : result) {
                st.push(nodes[idx]);
            }
            while (not st.empty()) {
                HNSWNode* cur = st.top();
                st.pop();
                if (h >= cur->neighbors.size()) continue;
                for (auto& [neighborIdx, neighbor] : cur->neighbors[h]) {
                    if (dists[neighborIdx] == -1) {
                        dists[neighborIdx] = dist(neighbor, val);
                    }
                    int neighborDist = dists[neighborIdx];
                    if ((result.size() < topk or neighborDist < result.rbegin()->first) and result.find({neighborDist, neighborIdx}) == result.end()) {
                        result.insert({neighborDist, neighborIdx});
                        st.push(neighbor);
                        if (result.size() > topk) {
                            result.erase(prev(result.end()));
                        }
                    }
                }
            }
            if (h <= height) ret[h] = result;
        }
        return ret;
    }
};