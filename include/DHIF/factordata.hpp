#pragma once
#ifndef DHIF_FACTORDATA_HPP
#define DHIF_FACTORDATA_HPP 1

namespace DHIF {

// Faces of a cube
enum Face {TOP=0, BOTTOM, RIGHT, LEFT, FRONT, BACK};

// Structure for storing the Schur complement data needed for application
// to a vector.
template <typename Scalar>
class FactorData {
public:
    Dense<Scalar>& A_22() { return A_22_; }
    Dense<Scalar>& A_22_inv() { return A_22_inv_; }
    Dense<Scalar>& X_mat() { return X_mat_; }
    Dense<Scalar>& Schur_comp() { return Schur_comp_; }
    Dense<Scalar>& W_mat() { return W_mat_; }

    int NumDOFsEliminated() { return ind_data_.redundant_inds().size(); }

    IndexData& ind_data() { return ind_data_; }
    void set_face(Face face) { face_ = face; }
    Face face() { return face_; }

private:
    IndexData ind_data_;
    Dense<Scalar> A_22_;        // matrix restricted to interactions
    Dense<Scalar> A_22_inv_;    // explicit inverse of A_22
    Dense<Scalar> X_mat_;       // A_22_inv * A_21
    Dense<Scalar> Schur_comp_;  // -A_12 * X_mat
    Dense<Scalar> W_mat_;       // Interpolative factor (only for Skel)
    Face face_;                 // To which face this data corresponds
                                // (only for Skel)
};

}
#endif  // ifndef DHIF_FACTORDATA_HPP
