#pragma once
#ifndef DHIF_INDEXDATA_HPP
#define DHIF_INDEXDATA_HPP 1

namespace DHIF {

// Faces of a cube
enum Face {TOP=0, BOTTOM, RIGHT, LEFT, FRONT, BACK};

class IndexData {
public:
    IndexData() {}
    ~IndexData() {
	global_inds_.clear();
	redundant_inds_.clear();
	skeleton_inds_.clear();
    }

    // Indices into the global matrix of size N^3 x N^3
    std::vector<int>& global_inds() { return global_inds_; }

    // If global_inds_ is of size n, then the redundant and skeleton
    // indices are disjoint index subsets of {0, ..., n-1} that correspond
    // to the degrees of freedom being eliminated and their interactions.
    std::vector<int>& redundant_inds() { return redundant_inds_; }
    std::vector<int>& skeleton_inds() { return skeleton_inds_; }

    void Print() {
	std::cout << "Global indices: " << std::endl;
	for (size_t i = 0; i < global_inds_.size(); ++i) {
	    std::cout << global_inds_[i] << std::endl;
	}
	std::cout << "Redundant indices: " << std::endl;
	for (size_t i = 0; i < redundant_inds_.size(); ++i) {
	    std::cout << redundant_inds_[i] << std::endl;
	}
	std::cout << "Skeleton indices: " << std::endl;
	for (size_t i = 0; i < skeleton_inds_.size(); ++i) {
	    std::cout << skeleton_inds_[i] << std::endl;
	}
    }

    void PrintGlobal() {
	std::cout << "Redundant (global): " << std::endl;
	for (size_t i = 0; i < redundant_inds_.size(); ++i) {
	    std::cout << global_inds_[redundant_inds_[i]] << std::endl;
	}
	std::cout << "Skeleton (global): " << std::endl;
	for (size_t i = 0; i < redundant_inds_.size(); ++i) {
	    std::cout << global_inds_[skeleton_inds_[i]] << std::endl;
	}
    }

private:
    std::vector<int> global_inds_;         // indices into N^3 x N^3 system
    std::vector<int> redundant_inds_;      // indices of global_inds_ corresponding
                                           // to what is being eliminated
    std::vector<int> skeleton_inds_;       // indices of global_inds_ corresponding
                                           // to non-zero entries of the matrix below
                                           // global_inds_(redundant_inds_).
};

}
#endif  // ifndef DHIF_INDEXDATA_HPP
