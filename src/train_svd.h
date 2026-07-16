// train_svd.h - self-contained top-r SVD for LoRA-XS base initialization.
//
// LoRA-XS freezes U:[out,r] and V:[in,r] as the top-r left/right singular bases of the
// base weight W0 and trains only the r x r core M_xs. The official implementation gets
// these from torch.linalg.svd (models/lora/model.py). ggml has no SVD op and an exact
// full SVD on ~1500-dim weights across ~250 targets is far too slow, so we compute the
// top-r subspace with randomized SVD (power iterations) + a small dense Jacobi eigensolve.
// Bases are not bit-identical to torch's SVD, but they span the same dominant subspace,
// which is all LoRA-XS requires (M_xs is trained on top). Deterministic given a seed.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace sa3 {

// Row-major [rows][cols] dense matrix helpers (element(r,c) = data[r*cols + c]).

// C[ar][bc] = A[ar][ac] * B[br][bc], requires ac == br.
inline void svd_mat_mul(const float* A, int ar, int ac, const float* B, int bc, std::vector<float>& C) {
    C.assign((size_t)ar * bc, 0.0f);
    for (int i = 0; i < ar; ++i) {
        const float* Ai = A + (size_t)i * ac;
        float* Ci = C.data() + (size_t)i * bc;
        for (int k = 0; k < ac; ++k) {
            const float a = Ai[k];
            if (a == 0.0f) continue;
            const float* Bk = B + (size_t)k * bc;
            for (int j = 0; j < bc; ++j) Ci[j] += a * Bk[j];
        }
    }
}

// C[ac][bc] = A^T * B where A is [k][ac], B is [k][bc]; contracts the shared leading dim k.
inline void svd_matT_mul(const float* A, int k, int ac, const float* B, int bc, std::vector<float>& C) {
    C.assign((size_t)ac * bc, 0.0f);
    for (int p = 0; p < k; ++p) {
        const float* Ap = A + (size_t)p * ac;
        const float* Bp = B + (size_t)p * bc;
        for (int i = 0; i < ac; ++i) {
            const float a = Ap[i];
            if (a == 0.0f) continue;
            float* Ci = C.data() + (size_t)i * bc;
            for (int j = 0; j < bc; ++j) Ci[j] += a * Bp[j];
        }
    }
}

// Modified Gram-Schmidt: orthonormalize the `cols` columns of Q[rows][cols] in place.
// Columns that are (numerically) linearly dependent on earlier ones are zeroed rather than
// normalized, so rank-deficient inputs (common with oversampling) produce clean orthonormal
// bases instead of unit-length numerical noise that would pollute the downstream SVD.
inline void svd_orthonormalize(std::vector<float>& Q, int rows, int cols) {
    double ref = 0.0;
    for (int c = 0; c < cols; ++c) {
        double n = 0.0;
        for (int r = 0; r < rows; ++r) { const double v = Q[(size_t)r * cols + c]; n += v * v; }
        ref = std::max(ref, n);
    }
    const double tol = std::sqrt(ref) * 1e-6;
    for (int c = 0; c < cols; ++c) {
        for (int p = 0; p < c; ++p) {
            double dot = 0.0;
            for (int r = 0; r < rows; ++r) dot += (double)Q[(size_t)r * cols + c] * Q[(size_t)r * cols + p];
            const float d = (float)dot;
            for (int r = 0; r < rows; ++r) Q[(size_t)r * cols + c] -= d * Q[(size_t)r * cols + p];
        }
        double nrm = 0.0;
        for (int r = 0; r < rows; ++r) { const double v = Q[(size_t)r * cols + c]; nrm += v * v; }
        nrm = std::sqrt(nrm);
        const float inv = nrm > tol ? (float)(1.0 / nrm) : 0.0f;   // zero dependent columns
        for (int r = 0; r < rows; ++r) Q[(size_t)r * cols + c] *= inv;
    }
}

// Cyclic Jacobi eigensolver for a symmetric l x l matrix (row-major). On return `evec`
// holds eigenvectors as columns ([l][l]) and `eval` the eigenvalues (unsorted).
inline void svd_jacobi_eig(std::vector<float> A, int l, std::vector<float>& eval, std::vector<float>& evec) {
    std::vector<double> a((size_t)l * l);
    for (int i = 0; i < l * l; ++i) a[i] = A[i];
    std::vector<double> v((size_t)l * l, 0.0);
    for (int i = 0; i < l; ++i) v[(size_t)i * l + i] = 1.0;
    for (int sweep = 0; sweep < 100; ++sweep) {
        double off = 0.0;
        for (int p = 0; p < l; ++p) for (int q = p + 1; q < l; ++q) off += a[(size_t)p * l + q] * a[(size_t)p * l + q];
        if (off < 1e-24) break;
        for (int p = 0; p < l; ++p) {
            for (int q = p + 1; q < l; ++q) {
                const double apq = a[(size_t)p * l + q];
                if (std::fabs(apq) < 1e-300) continue;
                const double app = a[(size_t)p * l + p], aqq = a[(size_t)q * l + q];
                const double phi = 0.5 * std::atan2(2.0 * apq, aqq - app);
                const double c = std::cos(phi), s = std::sin(phi);
                for (int k = 0; k < l; ++k) {
                    const double akp = a[(size_t)k * l + p], akq = a[(size_t)k * l + q];
                    a[(size_t)k * l + p] = c * akp - s * akq;
                    a[(size_t)k * l + q] = s * akp + c * akq;
                }
                for (int k = 0; k < l; ++k) {
                    const double apk = a[(size_t)p * l + k], aqk = a[(size_t)q * l + k];
                    a[(size_t)p * l + k] = c * apk - s * aqk;
                    a[(size_t)q * l + k] = s * apk + c * aqk;
                }
                for (int k = 0; k < l; ++k) {
                    const double vkp = v[(size_t)k * l + p], vkq = v[(size_t)k * l + q];
                    v[(size_t)k * l + p] = c * vkp - s * vkq;
                    v[(size_t)k * l + q] = s * vkp + c * vkq;
                }
            }
        }
    }
    eval.assign(l, 0.0f);
    evec.assign((size_t)l * l, 0.0f);
    for (int i = 0; i < l; ++i) eval[i] = (float)a[(size_t)i * l + i];
    for (int i = 0; i < l * l; ++i) evec[i] = (float)v[i];
}

// Randomized top-r SVD of the [m x n] matrix M (row-major, M[o*n + i] = W[o,i]).
// Writes U ([m*r], row-major [out][r]) and V ([n*r], row-major [in][r]) with the
// per-column sign convention that the largest-magnitude entry of each U column is positive.
inline void randomized_svd_topr(const std::vector<float>& M, int m, int n, int rank,
                                unsigned long long seed, std::vector<float>& U, std::vector<float>& V) {
    const int r = std::min(rank, std::min(m, n));
    const int l = std::min(std::min(m, n), r + 8);
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> gauss(0.0f, 1.0f);

    std::vector<float> Omega((size_t)n * l);
    for (float& x : Omega) x = gauss(rng);
    std::vector<float> Y;
    svd_mat_mul(M.data(), m, n, Omega.data(), l, Y);      // Y[m][l] = M * Omega
    for (int q = 0; q < 2; ++q) {                         // power iterations sharpen the subspace
        svd_orthonormalize(Y, m, l);
        std::vector<float> Z;
        svd_matT_mul(M.data(), m, n, Y.data(), l, Z);     // Z[n][l] = M^T * Y
        svd_orthonormalize(Z, n, l);
        svd_mat_mul(M.data(), m, n, Z.data(), l, Y);      // Y[m][l] = M * Z
    }
    svd_orthonormalize(Y, m, l);                          // Q[m][l]

    std::vector<float> B;
    svd_matT_mul(Y.data(), m, l, M.data(), n, B);         // B[l][n] = Q^T * M
    std::vector<float> C((size_t)l * l, 0.0f);            // C[l][l] = B * B^T
    for (int i = 0; i < l; ++i) for (int j = 0; j < l; ++j) {
        double s = 0.0;
        for (int k = 0; k < n; ++k) s += (double)B[(size_t)i * n + k] * B[(size_t)j * n + k];
        C[(size_t)i * l + j] = (float)s;
    }
    std::vector<float> eval, evec;
    svd_jacobi_eig(C, l, eval, evec);                     // C = evec diag(eval) evec^T

    std::vector<int> order(l);
    for (int i = 0; i < l; ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) { return eval[a] > eval[b]; });

    U.assign((size_t)m * r, 0.0f);
    V.assign((size_t)n * r, 0.0f);
    for (int jj = 0; jj < r; ++jj) {
        const int e = order[jj];
        const float sigma = std::sqrt(std::max(eval[e], 0.0f));
        // left singular vector u = Q * evec[:,e]  ([m])
        std::vector<float> u((size_t)m, 0.0f);
        for (int rr = 0; rr < m; ++rr) {
            double s = 0.0;
            for (int k = 0; k < l; ++k) s += (double)Y[(size_t)rr * l + k] * evec[(size_t)k * l + e];
            u[rr] = (float)s;
        }
        // right singular vector v = B^T * evec[:,e] / sigma  ([n])
        std::vector<float> vv((size_t)n, 0.0f);
        if (sigma > 1e-12f) {
            const float inv = 1.0f / sigma;
            for (int i = 0; i < n; ++i) {
                double s = 0.0;
                for (int k = 0; k < l; ++k) s += (double)B[(size_t)k * n + i] * evec[(size_t)k * l + e];
                vv[i] = (float)s * inv;
            }
        }
        // sign convention: largest-|.| entry of the u column is positive
        int amax = 0; float vmax = 0.0f;
        for (int rr = 0; rr < m; ++rr) { const float av = std::fabs(u[rr]); if (av > vmax) { vmax = av; amax = rr; } }
        const float sgn = u[amax] < 0.0f ? -1.0f : 1.0f;
        for (int rr = 0; rr < m; ++rr) U[(size_t)rr * r + jj] = u[rr] * sgn;
        for (int i = 0; i < n; ++i)   V[(size_t)i * r + jj] = vv[i] * sgn;
    }
}

} // namespace sa3
