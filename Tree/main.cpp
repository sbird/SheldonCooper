//
//  main.cpp
//  2024fall_ComAstro
//
//  Created by honghui on 2024/10/14.
//

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <chrono>

const double const_G = 1.0;  // 引力常数

class Node {
public:
    double mass;
    std::vector<double> mposition;
    std::vector<double> lowerbound, upperbound;
    int depth;
    int index;
    Node* left_node;
    Node* right_node;
    Node* last_node;

    Node(double mass, std::vector<double> mposition, std::vector<double> lowerbound, std::vector<double> upperbound)
        : mass(mass), mposition(mposition), lowerbound(lowerbound), upperbound(upperbound), depth(0), index(-1), left_node(nullptr), right_node(nullptr), last_node(nullptr) {}
};

class Particle {
public:
    double mass;
    std::vector<double> position;
    std::vector<double> vel, acc;
    int seq;
    Node* parent;

    Particle(double mass, std::vector<double> position, std::vector<double> vel, std::vector<double> acc, int seq)
        : mass(mass), position(position), vel(vel), acc(acc), seq(seq), parent(nullptr) {}

    void update_time(double new_time) {
        // 更新粒子的时间信息
    }

    void cal_gforce(std::vector<Particle>& p_array) {
        acc = std::vector<double>(3, 0.0);  // 初始化加速度为 0
        for (size_t i = 0; i < p_array.size(); ++i) {
            if (i != seq) {
                std::vector<double> del_pos(3);
                for (int j = 0; j < 3; ++j) {
                    del_pos[j] = p_array[i].position[j] - position[j];
                }
                double rr = std::sqrt(del_pos[0] * del_pos[0] + del_pos[1] * del_pos[1] + del_pos[2] * del_pos[2]);
                for (int j = 0; j < 3; ++j) {
                    acc[j] += p_array[i].mass * del_pos[j] / std::pow(rr, 3);
                }
            }
        }
        for (int j = 0; j < 3; ++j) {
            acc[j] *= const_G;
        }
    }
};

void insert_Node_tree(Particle& par, std::vector<Particle>& list_par, Node* Node_tree) {
    int remainder_dep = Node_tree->depth % 3;
    Node_tree->mass += par.mass;
    for (int i = 0; i < 3; ++i) {
        Node_tree->mposition[i] += par.mass * par.position[i];
    }

    if (Node_tree->left_node == nullptr && Node_tree->right_node == nullptr) {
        if (Node_tree->index == -1) {
            Node_tree->index = par.seq;
            par.parent = Node_tree;
        } else {
            std::vector<double> new_mposition(3, 0.0);
            Node* node1 = new Node(0, new_mposition, Node_tree->lowerbound, Node_tree->upperbound);
            Node* node2 = new Node(0, new_mposition, Node_tree->lowerbound, Node_tree->upperbound);
            double cut = (Node_tree->upperbound[remainder_dep] + Node_tree->lowerbound[remainder_dep]) / 2.0;
            node1->upperbound[remainder_dep] = cut;
            node2->lowerbound[remainder_dep] = cut;
            node1->last_node = Node_tree;
            node2->last_node = Node_tree;
            node1->depth = Node_tree->depth + 1;
            node2->depth = Node_tree->depth + 1;
            Node_tree->left_node = node1;
            Node_tree->right_node = node2;

            if (list_par[Node_tree->index].position[remainder_dep] <= cut) {
                insert_Node_tree(list_par[Node_tree->index], list_par, Node_tree->left_node);
            } else {
                insert_Node_tree(list_par[Node_tree->index], list_par, Node_tree->right_node);
            }

            if (par.position[remainder_dep] <= cut) {
                insert_Node_tree(par, list_par, Node_tree->left_node);
            } else {
                insert_Node_tree(par, list_par, Node_tree->right_node);
            }
            Node_tree->index = -1;
        }
    } else {
        double cut = (Node_tree->upperbound[remainder_dep] + Node_tree->lowerbound[remainder_dep]) / 2.0;
        if (par.position[remainder_dep] <= cut) {
            insert_Node_tree(par, list_par, Node_tree->left_node);
        } else {
            insert_Node_tree(par, list_par, Node_tree->right_node);
        }
    }
}

std::vector<double> cal_force(Particle& par, Node* Node_tree) {
    std::vector<double> force(3, 0.0);
    if (Node_tree == nullptr || Node_tree->mass == 0 || Node_tree->index == par.seq) {
        return force;
    }

    std::vector<double> del_pos(3, 0.0);
    for (int i = 0; i < 3; ++i) {
        del_pos[i] = (Node_tree->mposition[i] / Node_tree->mass) - par.position[i];
    }
    double rr = std::sqrt(del_pos[0] * del_pos[0] + del_pos[1] * del_pos[1] + del_pos[2] * del_pos[2]);

    if (std::sqrt(std::pow(Node_tree->upperbound[0] - Node_tree->lowerbound[0], 2) +
                  std::pow(Node_tree->upperbound[1] - Node_tree->lowerbound[1], 2) +
                  std::pow(Node_tree->upperbound[2] - Node_tree->lowerbound[2], 2)) / rr < 1 || Node_tree->index != -1) {
        for (int i = 0; i < 3; ++i) {
            force[i] = const_G * Node_tree->mass * del_pos[i] / std::pow(rr, 3);
        }
        return force;
    } else {
        std::vector<double> force_left = cal_force(par, Node_tree->left_node);
        std::vector<double> force_right = cal_force(par, Node_tree->right_node);
        for (int i = 0; i < 3; ++i) {
            force[i] = force_left[i] + force_right[i];
        }
        return force;
    }
}

int main() {
    int num=1e4;
    int boxsize=10;
    std::vector<double> box_lowerbound(3, 0.0);
    std::vector<double> box_upperbound(3, std::pow(2, boxsize));
    std::vector<Particle> list_par;
    Node* root = new Node(0, std::vector<double>(3, 0.0), box_lowerbound, box_upperbound);

    for (int i = 0; i < num; ++i) {
        std::vector<double> position = {static_cast<double>(rand() % (int)(std::pow(2, boxsize) + 1)),
                                        static_cast<double>(rand() % (int)(std::pow(2, boxsize) + 1)),
                                        static_cast<double>(rand() % (int)(std::pow(2, boxsize) + 1))};
        Particle pp(1, position, std::vector<double>(3, 0.0), std::vector<double>(3, 0.0), i);
        list_par.push_back(pp);
    }

    for (int i = 0; i < num; ++i) {
        insert_Node_tree(list_par[i], list_par, root);
    }

    auto stime = std::chrono::high_resolution_clock::now();
    for(int i = 0; i< num; ++i)
        std::vector<double> force = cal_force(list_par[0], root);
    auto etime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = etime - stime;

//std::cout << "Force: [" << force[0] << ", " << force[1] << ", " << force[2] << "]" << std::endl;
    std::cout << "Time = " << elapsed_time.count() << " seconds" << std::endl;

    stime = std::chrono::high_resolution_clock::now();
    for(int i=0;i< num;++i)
        list_par[0].cal_gforce(list_par);
    etime = std::chrono::high_resolution_clock::now();
    elapsed_time = etime - stime;

//0std::cout << "Acceleration: [" << list_par[0].acc[0] << ", " << list_par[0].acc[1] << ", " << list_par[0].acc[2] << "]" << std::endl;
    std::cout << "Time = " << elapsed_time.count() << " seconds" << std::endl;

    return 0;
}
