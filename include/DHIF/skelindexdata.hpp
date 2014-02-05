#pragma once
#ifndef DHIF_SKELINDEXDATA_HPP
#define DHIF_SKELINDEXDATA_HPP 1

namespace DHIF {

class SkelIndexData {
public:
    SkelIndexData() {}
    ~SkelIndexData() {
	global_rows_.clear();
	global_cols_.clear();
    }

    // Indices into the global matrix of size N^3 x N^3
    std::vector<int>& global_rows() { return global_rows_; }
    std::vector<int>& global_cols() { return global_cols_; }

private:
    std::vector<int> global_rows_;
    std::vector<int> global_cols_;
};

}
#endif  // ifndef DHIF_SKELINDEXDATA_HPP
