// CMA-ES (mu/mu_w, lambda) with rank-one and rank-mu covariance updates and cumulative step-size adaptation (CSA).
// Written from the open description: N. Hansen, "The CMA Evolution Strategy: A Tutorial", arXiv:1604.00772 (2016), the
// default strategy parameters of its Table 1 (eq. 48-58 numbering of the tutorial). No external library.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace cma {

// Jacobi eigen decomposition of a symmetric matrix a (n x n, row-major): a = V diag(d) V^T, V columns = eigenvectors.
inline void Eigen(const std::vector<double>& a_in, int n, std::vector<double>* d, std::vector<double>* V) {
  std::vector<double> a = a_in;
  V->assign(n * n, 0.0);
  for (int i = 0; i < n; ++i) (*V)[i * n + i] = 1.0;
  for (int sweep = 0; sweep < 60; ++sweep) {
    double off = 0;
    for (int i = 0; i < n; ++i)
      for (int j = i + 1; j < n; ++j) off += a[i * n + j] * a[i * n + j];
    if (off < 1e-24) break;
    for (int p = 0; p < n; ++p)
      for (int q = p + 1; q < n; ++q) {
        const double apq = a[p * n + q];
        if (std::fabs(apq) < 1e-300) continue;
        const double theta = (a[q * n + q] - a[p * n + p]) / (2.0 * apq);
        const double t = (theta >= 0 ? 1.0 : -1.0) / (std::fabs(theta) + std::sqrt(theta * theta + 1.0));
        const double c = 1.0 / std::sqrt(t * t + 1.0), s = t * c;
        for (int k = 0; k < n; ++k) {  // rotate columns p, q
          const double akp = a[k * n + p], akq = a[k * n + q];
          a[k * n + p] = c * akp - s * akq;
          a[k * n + q] = s * akp + c * akq;
        }
        for (int k = 0; k < n; ++k) {  // rotate rows p, q
          const double apk = a[p * n + k], aqk = a[q * n + k];
          a[p * n + k] = c * apk - s * aqk;
          a[q * n + k] = s * apk + c * aqk;
        }
        for (int k = 0; k < n; ++k) {
          const double vkp = (*V)[k * n + p], vkq = (*V)[k * n + q];
          (*V)[k * n + p] = c * vkp - s * vkq;
          (*V)[k * n + q] = s * vkp + c * vkq;
        }
      }
  }
  d->assign(n, 0.0);
  for (int i = 0; i < n; ++i) (*d)[i] = a[i * n + i];
}

struct Strategy {
  int n = 0, lambda = 0, mu = 0, gen = 0;
  double sigma = 0.3, mueff = 0, cc = 0, cs = 0, c1 = 0, cmu = 0, damps = 0, chiN = 0;
  std::vector<double> w, mean, C, B, D, pc, ps;  // B: eigenvectors (columns), D: sqrt eigenvalues

  void Init(int n_, int lambda_, const std::vector<double>& x0, double sigma0) {
    n = n_;
    lambda = lambda_ > 0 ? lambda_ : 4 + (int)(3 * std::log((double)n));
    mu = lambda / 2;
    w.resize(mu);
    double sw = 0, sw2 = 0;
    for (int i = 0; i < mu; ++i) { w[i] = std::log(mu + 0.5) - std::log(i + 1.0); sw += w[i]; }
    for (int i = 0; i < mu; ++i) { w[i] /= sw; sw2 += w[i] * w[i]; }
    mueff = 1.0 / sw2;
    cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n);
    cs = (mueff + 2) / (n + mueff + 5);
    c1 = 2 / ((n + 1.3) * (n + 1.3) + mueff);
    cmu = std::min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) * (n + 2) + mueff));
    damps = 1 + 2 * std::max(0.0, std::sqrt((mueff - 1) / (n + 1)) - 1) + cs;
    chiN = std::sqrt((double)n) * (1 - 1.0 / (4 * n) + 1.0 / (21.0 * n * n));
    mean = x0;
    sigma = sigma0;
    C.assign(n * n, 0.0);
    B.assign(n * n, 0.0);
    D.assign(n, 1.0);
    for (int i = 0; i < n; ++i) { C[i * n + i] = 1; B[i * n + i] = 1; }
    pc.assign(n, 0.0);
    ps.assign(n, 0.0);
    gen = 0;
  }

  // lambda candidates x_k = mean + sigma B D z_k (raw, unbounded); y_k = B D z_k is kept for the update.
  void Ask(std::mt19937_64& rng, std::vector<std::vector<double>>* xs) const {
    std::normal_distribution<double> nd(0.0, 1.0);
    xs->assign(lambda, std::vector<double>(n));
    for (int k = 0; k < lambda; ++k) {
      std::vector<double> z(n);
      for (int i = 0; i < n; ++i) z[i] = D[i] * nd(rng);
      for (int i = 0; i < n; ++i) {
        double s = 0;
        for (int j = 0; j < n; ++j) s += B[i * n + j] * z[j];
        (*xs)[k][i] = mean[i] + sigma * s;
      }
    }
  }

  // xs sorted by cost, best first (only the first mu are used).
  void Tell(const std::vector<std::vector<double>>& xs_sorted) {
    std::vector<double> old = mean, ymean(n, 0.0);
    for (int i = 0; i < n; ++i) {
      double m = 0;
      for (int k = 0; k < mu; ++k) m += w[k] * xs_sorted[k][i];
      mean[i] = m;
      ymean[i] = (m - old[i]) / sigma;
    }
    // C^(-1/2) ymean = B D^-1 B^T ymean
    std::vector<double> t(n, 0.0), inv(n, 0.0);
    for (int j = 0; j < n; ++j) {
      double s = 0;
      for (int i = 0; i < n; ++i) s += B[i * n + j] * ymean[i];
      t[j] = s / D[j];
    }
    for (int i = 0; i < n; ++i) {
      double s = 0;
      for (int j = 0; j < n; ++j) s += B[i * n + j] * t[j];
      inv[i] = s;
    }
    const double csn = std::sqrt(cs * (2 - cs) * mueff);
    double psn = 0;
    for (int i = 0; i < n; ++i) { ps[i] = (1 - cs) * ps[i] + csn * inv[i]; psn += ps[i] * ps[i]; }
    psn = std::sqrt(psn);
    const double hs = psn / std::sqrt(1 - std::pow(1 - cs, 2.0 * (gen + 1))) / chiN < 1.4 + 2.0 / (n + 1) ? 1.0 : 0.0;
    const double ccn = std::sqrt(cc * (2 - cc) * mueff);
    for (int i = 0; i < n; ++i) pc[i] = (1 - cc) * pc[i] + hs * ccn * ymean[i];
    const double dh = (1 - hs) * cc * (2 - cc);
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j) {
        double rmu = 0;
        for (int k = 0; k < mu; ++k) rmu += w[k] * ((xs_sorted[k][i] - old[i]) / sigma) * ((xs_sorted[k][j] - old[j]) / sigma);
        C[i * n + j] = (1 - c1 - cmu + c1 * dh) * C[i * n + j] + c1 * pc[i] * pc[j] + cmu * rmu;
      }
    sigma *= std::exp((cs / damps) * (psn / chiN - 1));
    sigma = std::min(sigma, 1.0);
    ++gen;
    // symmetrise and decompose
    for (int i = 0; i < n; ++i)
      for (int j = i + 1; j < n; ++j) C[i * n + j] = C[j * n + i] = 0.5 * (C[i * n + j] + C[j * n + i]);
    std::vector<double> d;
    Eigen(C, n, &d, &B);
    D.assign(n, 1.0);
    for (int i = 0; i < n; ++i) D[i] = std::sqrt(std::max(d[i], 1e-20));
  }

  bool Save(const char* path) const {
    FILE* f = fopen(path, "w");
    if (!f) return false;
    fprintf(f, "%d %d %d %.17g\n", n, lambda, gen, sigma);
    auto dump = [&](const std::vector<double>& v) { for (double x : v) fprintf(f, "%.17g ", x); fprintf(f, "\n"); };
    dump(mean); dump(C); dump(pc); dump(ps);
    fclose(f);
    return true;
  }

  bool Load(const char* path, int n_expect, int lambda_expect) {
    FILE* f = fopen(path, "r");
    if (!f) return false;
    int nn = 0, lam = 0, g = 0;
    double sg = 0;
    if (fscanf(f, "%d %d %d %lf", &nn, &lam, &g, &sg) != 4 || nn != n_expect) { fclose(f); return false; }
    std::vector<double> x0(nn, 0.0);
    Init(nn, lambda_expect > 0 ? lambda_expect : lam, x0, sg);
    auto load = [&](std::vector<double>* v) { for (double& x : *v) if (fscanf(f, "%lf", &x) != 1) return false; return true; };
    bool ok = load(&mean) && load(&C) && load(&pc) && load(&ps);
    fclose(f);
    if (!ok) return false;
    gen = g;
    std::vector<double> d;
    Eigen(C, n, &d, &B);
    for (int i = 0; i < n; ++i) D[i] = std::sqrt(std::max(d[i], 1e-20));
    return true;
  }
};

}  // namespace cma
