#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <Eigen/Dense>
using namespace std;

// ======= Utility: standard normal CDF =======
static double stdnorm_cdf(double x){
    static const double invSqrt2=0.7071067811865475;
    return 0.5 * std::erfc(-x * invSqrt2);
}

// =========== Regression function enumeration ===========
enum FuncNum {
    POLY_REG_FILTER=1,
    POLY_REG_NOFILTER=2,
    POLY_REG_CHOOSE_DEG_FILTER=3,
    POLY_REG_CHOOSE_DEG_NOFILTER=4,
    LAGUERRE_REG_FILTER=5,
    LAGUERRE_REG_NOFILTER=6
};

// =========== Helper: compute the mean of a vector ===========
static double meanVec(const vector<double>& vec){
    if(vec.empty()) return 0.0;
    double s=0.0;
    for(auto &v: vec) s += v;
    return s / static_cast<double>(vec.size());
}

// =========== Black-Scholes European option pricing (used for control variate) ===========
static double bs_euro(char type, double S, double K,
                      double r, double div, double sigma,
                      double t)
{
    if(t <= 0.0) return 0.0;
    double d1= ( std::log(S / K) + (r - div + 0.5 * sigma * sigma) * t ) / ( sigma * std::sqrt(t) );
    double d2= d1 - sigma * std::sqrt(t);
    if(type=='C' || type=='c'){
        return stdnorm_cdf(d1) * S - stdnorm_cdf(d2) * K * std::exp(-(r - div) * t);
    } else {
        return stdnorm_cdf(-d2) * K * std::exp(-(r - div) * t) - stdnorm_cdf(-d1) * S;
    }
}

// =========== Core class: American LSMC + partial Greeks (Delta, Gamma, Vega) ===========
class AmericanLSMC_CV {
public:
    AmericanLSMC_CV(string type, double S, double K, double T,
                    int M, double r, double div, double sigma,
                    int func_num, int n_poly, int sims, int seed)
    : type_(type), S_(S), K_(K), T_(T), M_(M), r_(r), div_(div), sigma_(sigma),
      func_num_(func_num), n_poly_(n_poly), sims_(sims), seed_(seed)
    {
        dt_ = T_ / static_cast<double>(M_);
        discount_ = std::exp(-(r_ - div_) * dt_);
        V0_ = 0.;
        std_ = 0.;
    }

    // Main pricing function
    void price(){
        // 1) Generate paths
        buildPaths();

        // 2) Calculate payoff
        payoff_.resize((M_ + 1) * sims_);
#pragma omp parallel for
        for(int t=0; t <= M_; t++){
            for(int i=0; i < sims_; i++){
                double s = paths_[static_cast<long long>(t) * sims_ + i];
                double p = 0.;
                if(type_=="C" || type_=="c"){
                    p = std::max(s - K_, 0.);
                } else {
                    p = std::max(K_ - s, 0.);
                }
                payoff_[static_cast<long long>(t) * sims_ + i] = p;
            }
        }

        // 3) Backward induction in LSMC
        V_.resize(sims_);
        Y_.resize(sims_);

        // Initialization: at t = M_ => payoff(M_)
#pragma omp parallel for
        for(int i=0; i < sims_; i++){
            V_[i] = payoff_[static_cast<long long>(M_) * sims_ + i];
            Y_[i] = payoff_[static_cast<long long>(M_) * sims_ + i];
        }

        // Time steps from M_-1 down to 1
        for(int i = M_ - 1; i >= 1; i--){
            // Compute BS_price for remaining time
            vector<double> BS_price(sims_);
#pragma omp parallel for
            for(int sidx = 0; sidx < sims_; sidx++){
                double s = paths_[static_cast<long long>(i) * sims_ + sidx];
                double remainT = T_ * (1.0 - static_cast<double>(i) / (M_ * T_));
                BS_price[sidx] = bs_euro(type_[0], s, K_, r_, div_, sigma_, remainT);
            }

            // Data needed for regression => WY=V*Y, Y, Y^2, V
            vector<double> dataWY(sims_), dataY(sims_), dataY2(sims_), dataV(sims_), dataX(sims_);
#pragma omp parallel for
            for(int sidx=0; sidx < sims_; sidx++){
                dataWY[sidx] = V_[sidx] * Y_[sidx];
                dataY[sidx]  = Y_[sidx];
                dataY2[sidx] = Y_[sidx] * Y_[sidx];
                dataV[sidx]  = V_[sidx];
                dataX[sidx]  = paths_[static_cast<long long>(i) * sims_ + sidx];
            }

            // Perform regression => WY_apx, Y_apx, Y2_apx, cont
            vector<double> WY_apx, Y_apx, Y2_apx, cont;
            vector<int> condIdx;
            regr(dataX, dataWY, WY_apx, condIdx);
            regr(dataX, dataY,  Y_apx, condIdx);
            regr(dataX, dataY2, Y2_apx, condIdx);
            regr(dataX, dataV,  cont, condIdx);

            // Compute coefficient b
            double cMean  = meanVec(cont);
            double wyMean = meanVec(WY_apx);
            double yMean  = meanVec(Y_apx);
            double y2Mean = meanVec(Y2_apx);
            double denom  = (y2Mean - yMean * yMean);
            double b = 0.;
            if(fabs(denom) > 1e-14){
                double num = (cMean * yMean - wyMean);
                b = num / denom;
            }

            // Update V_ and Y_ based on whether we are filtering or not
            if(func_num_ % 2 == 1){
                // Filter
#pragma omp parallel for
                for(int sidx=0; sidx < sims_; sidx++){
                    V_[sidx] *= discount_;
                }
                for(int k=0; k < static_cast<int>(condIdx.size()); k++){
                    int idx = condIdx[k];
                    double cv  = cont[k] + b * (Y_apx[k] - BS_price[idx]);
                    double pay = payoff_[static_cast<long long>(i) * sims_ + idx];
                    if(cv > pay){
                        // keep discounted V_[idx]
                    } else {
                        V_[idx] = pay;
                    }
                }
#pragma omp parallel for
                for(int k=0; k < static_cast<int>(condIdx.size()); k++){
                    int idx = condIdx[k];
                    double c   = cont[k];
                    double pay = payoff_[static_cast<long long>(i) * sims_ + idx];
                    if(c > pay && pay < 0){
                        Y_[idx] *= discount_;
                    } else {
                        Y_[idx] = BS_price[idx];
                    }
                }
            } else {
                // No filter
                vector<double> CV_cont(sims_);
#pragma omp parallel for
                for(int sidx=0; sidx < sims_; sidx++){
                    double cv = cont[sidx] + b * (Y_apx[sidx] - BS_price[sidx]);
                    CV_cont[sidx] = cv;
                }
#pragma omp parallel for
                for(int sidx=0; sidx < sims_; sidx++){
                    double pay = payoff_[static_cast<long long>(i) * sims_ + sidx];
                    if(CV_cont[sidx] > pay){
                        V_[sidx] = discount_ * V_[sidx];
                    } else {
                        V_[sidx] = pay;
                    }
                }
#pragma omp parallel for
                for(int sidx=0; sidx < sims_; sidx++){
                    double pay = payoff_[static_cast<long long>(i) * sims_ + sidx];
                    double c   = cont[sidx];
                    if(c > pay && pay < 0){
                        Y_[sidx] *= discount_;
                    } else {
                        Y_[sidx] = BS_price[sidx];
                    }
                }
#pragma omp parallel for
                for(int sidx=0; sidx < sims_; sidx++){
                    V_[sidx] *= discount_;
                }
            }
        }

        // Final option value: discount * mean(V)
        double sumV = 0.;
#pragma omp parallel for reduction(+:sumV)
        for(int i=0; i < sims_; i++){
            sumV += V_[i];
        }
        double meanV = sumV / static_cast<double>(sims_);
        double V0 = discount_ * meanV;

        // Discounted mean(Y)
        double sumY = 0.;
#pragma omp parallel for reduction(+:sumY)
        for(int i=0; i < sims_; i++){
            sumY += Y_[i];
        }
        double meanY = sumY / static_cast<double>(sims_);
        double Y0 = discount_ * meanY;

        // Discounted mean(V * Y)
        double sumVY = 0.;
#pragma omp parallel for reduction(+:sumVY)
        for(int i=0; i < sims_; i++){
            sumVY += V_[i] * Y_[i];
        }
        double meanVY = sumVY / static_cast<double>(sims_);
        double FY0 = discount_ * meanVY;

        // Discounted mean(Y^2)
        double sumY2 = 0.;
#pragma omp parallel for reduction(+:sumY2)
        for(int i=0; i < sims_; i++){
            sumY2 += Y_[i] * Y_[i];
        }
        double meanY2 = sumY2 / static_cast<double>(sims_);
        double Y2_0 = discount_ * meanY2;

        double bCV = 0.;
        {
            double denom = (Y2_0 - Y0 * Y0);
            if(fabs(denom) > 1e-14){
                double num = V0 * Y0 - FY0;
                bCV = num / denom;
            }
        }

        // Final control variate value: V + b*(Y - bs_euro(S,T))
        vector<double> V_CV(sims_);
        double bs0 = bs_euro(type_[0], S_, K_, r_, div_, sigma_, T_);
#pragma omp parallel for
        for(int i=0; i < sims_; i++){
            V_CV[i] = V_[i] + bCV * (Y_[i] - bs0);
        }

        double V0_CV = V0 + bCV * (Y0 - bs0);

        // Immediate exercise payoff at t=0
        double payoff0 = 0.;
        if(type_=="C" || type_=="c"){
            payoff0 = std::max(S_ - K_, 0.);
        } else {
            payoff0 = std::max(K_ - S_, 0.);
        }

        // Final price is max of immediate exercise payoff and CV estimation
        V0_ = std::max(V0_CV, payoff0);

        // Standard deviation => discount * std(V_CV)
        double sumVCV = 0., sumVCV2 = 0.;
#pragma omp parallel for reduction(+:sumVCV,sumVCV2)
        for(int i=0; i < sims_; i++){
            sumVCV  += V_CV[i];
            sumVCV2 += V_CV[i] * V_CV[i];
        }
        double meanVCV = sumVCV / static_cast<double>(sims_);
        double varVCV  = sumVCV2 / static_cast<double>(sims_) - meanVCV * meanVCV;
        if(varVCV < 0.) varVCV = 0.;
        double stdVCV = std::sqrt(varVCV);
        std_ = discount_ * stdVCV;
    }

    // Accessors for price and standard deviation
    double getV0() const { return V0_; }
    double getStd() const{ return std_; }

    // Compute Delta via finite difference
    double computeDelta(double frac=0.01){
        double eps = S_ * frac;
        if(eps < 1e-8) eps = 1e-4;
        AmericanLSMC_CV up(type_, S_ + eps, K_, T_, M_, r_, div_, sigma_,
                           func_num_, n_poly_, sims_, seed_);
        up.price();
        double Vu = up.getV0();

        AmericanLSMC_CV dn(type_, S_ - eps, K_, T_, M_, r_, div_, sigma_,
                           func_num_, n_poly_, sims_, seed_);
        dn.price();
        double Vd = dn.getV0();

        return (Vu - Vd) / (2. * eps);
    }

    // Compute Gamma via finite difference
    double computeGamma(double frac=0.01){
        double base = getV0();
        double eps  = S_ * frac;
        if(eps < 1e-8) eps = 1e-4;

        AmericanLSMC_CV up(type_, S_ + eps, K_, T_, M_, r_, div_, sigma_,
                           func_num_, n_poly_, sims_, seed_);
        up.price();
        double Vu = up.getV0();

        AmericanLSMC_CV dn(type_, S_ - eps, K_, T_, M_, r_, div_, sigma_,
                           func_num_, n_poly_, sims_, seed_);
        dn.price();
        double Vd = dn.getV0();

        return (Vu + Vd - 2. * base) / (eps * eps);
    }

    // Compute Vega via finite difference
    double computeVega(double frac=0.01){
        double eps = sigma_ * frac;
        if(eps < 1e-8) eps = 1e-4;

        // up
        AmericanLSMC_CV solverUp(type_, S_, K_, T_, M_, r_, div_, sigma_ + eps,
                                 func_num_, n_poly_, sims_, seed_);
        solverUp.price();
        double Vu = solverUp.getV0();

        // down
        AmericanLSMC_CV solverDn(type_, S_, K_, T_, M_, r_, div_, sigma_ - eps,
                                 func_num_, n_poly_, sims_, seed_);
        solverDn.price();
        double Vd = solverDn.getV0();

        return (Vu - Vd) / (2. * eps);
    }

private:
    // Generate paths for the underlying asset
    void buildPaths(){
        paths_.resize((M_ + 1) * sims_);

#pragma omp parallel for
        for(int i = 0; i < sims_; i++){
            paths_[i] = S_;
        }
        for(int j = 1; j <= M_; j++){
            mt19937_64 rng(seed_ + j);
            int half = sims_ / 2;
            vector<double> zhalf(half);
            for(int i = 0; i < half; i++){
                zhalf[i] = std::normal_distribution<double>(0.0, 1.0)(rng);
            }
            vector<double> zall(sims_);
#pragma omp parallel for
            for(int i = 0; i < half; i++){
                zall[i]       = zhalf[i];
                zall[i+half]  = -zhalf[i];
            }
            double drift = (r_ - div_ - 0.5 * sigma_ * sigma_) * dt_;
            double vol   = sigma_ * std::sqrt(dt_);

#pragma omp parallel for
            for(int i = 0; i < sims_; i++){
                double prev = paths_[(j - 1) * sims_ + i];
                double now  = prev * std::exp(drift + vol * zall[i]);
                paths_[j * sims_ + i] = now;
            }
        }
    }

    // Regression dispatcher (deciding between filtering vs no filtering)
    void regr(const vector<double>& X,
              const vector<double>& Y,
              vector<double>& outCont,
              vector<int>& outCond)
    {
        bool doFilter = (func_num_ % 2 == 1);
        outCont.clear();
        outCond.clear();

        if(doFilter){
            poly_reg_filter(X, Y, outCont, outCond);
        } else {
            poly_reg_no_filter(X, Y, outCont);
        }
    }

    // Polynomial regression with filtering
    void poly_reg_filter(const vector<double>& data_x,
                         const vector<double>& data_y,
                         vector<double>& outCont,
                         vector<int>& outCond)
    {
        outCond.clear();
        if(type_=="C" || type_=="c"){
            for(int i = 0; i < static_cast<int>(data_x.size()); i++){
                if(data_x[i] > K_) outCond.push_back(i);
            }
        } else {
            for(int i = 0; i < static_cast<int>(data_x.size()); i++){
                if(data_x[i] < K_) outCond.push_back(i);
            }
        }
        outCont.resize(outCond.size());
        if(outCond.size() < 2){
            for(int i=0; i<static_cast<int>(outCond.size()); i++){
                outCont[i] = 0.;
            }
            return;
        }
        vector<double> Xf(outCond.size()), Yf(outCond.size());
        for(size_t idx=0; idx < outCond.size(); idx++){
            int ii = outCond[idx];
            Xf[idx] = data_x[ii];
            Yf[idx] = data_y[ii] * discount_;
        }
        int nrows = static_cast<int>(Xf.size());
        int ncols = n_poly_;
        Eigen::MatrixXd A(nrows, ncols);
        Eigen::VectorXd b(nrows);

        for(int i=0; i<nrows; i++){
            double xx = Xf[i];
            for(int p=1; p <= ncols; p++){
                A(i, p-1) = std::pow(xx, p);
            }
            b(i) = Yf[i];
        }
        Eigen::MatrixXd ATA = A.transpose() * A;
        Eigen::VectorXd ATb = A.transpose() * b;
        Eigen::VectorXd sol = ATA.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(ATb);

        for(int i=0; i<nrows; i++){
            double xx = Xf[i];
            double val = 0.;
            for(int p=1; p <= ncols; p++){
                val += sol(p-1) * std::pow(xx, p);
            }
            outCont[i] = val;
        }
    }

    // Polynomial regression without filtering
    void poly_reg_no_filter(const vector<double>& data_x,
                            const vector<double>& data_y,
                            vector<double>& outCont) const {
        int N = static_cast<int>(data_x.size());
        outCont.resize(N);

        Eigen::MatrixXd A(N, n_poly_ + 1);
        Eigen::VectorXd bVec(N);
        for(int i=0; i<N; i++){
            double xx = data_x[i];
            for(int p=0; p <= n_poly_; p++){
                A(i,p) = std::pow(xx, p);
            }
            bVec(i) = data_y[i] * discount_;
        }
        Eigen::MatrixXd ATA = A.transpose() * A;
        Eigen::VectorXd ATb = A.transpose() * bVec;
        Eigen::VectorXd sol = ATA.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(ATb);

        for(int i=0; i<N; i++){
            double xx = data_x[i];
            double val = 0.;
            for(int p=0; p <= n_poly_; p++){
                val += sol[p] * std::pow(xx, p);
            }
            outCont[i] = val;
        }
    }

private:
    string type_;
    double S_, K_, T_;
    int M_;
    double r_, div_, sigma_;
    int func_num_, n_poly_;
    int sims_, seed_;
    double dt_, discount_;

    vector<double> paths_;   // (M_+1) * sims_
    vector<double> payoff_;  // (M_+1) * sims_
    vector<double> V_, Y_;   // size = sims_

    double V0_, std_;
};

int main(){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    // Example parameters
    string type = "P";
    double S = 36.;
    double K = 40.;
    double T = 1.;
    int M = 40;
    double r = 0.05;
    double div = 0.;
    double sigma = 0.3;
    int n_poly = 2;
    int n_sim = 100000;
    int seed = 321;
    int func_num = 5;

    auto start= std::chrono::high_resolution_clock::now();
    AmericanLSMC_CV solver(type, S, K, T, M, r, div, sigma,
                           func_num, n_poly, n_sim, seed);
    solver.price();

    double price = solver.getV0();
    double stdev = solver.getStd();

    // Only compute Delta, Gamma, Vega
    double delta = solver.computeDelta();
    double gamma = solver.computeGamma();
    double vega  = solver.computeVega();

    auto end= std::chrono::high_resolution_clock::now();
    double secs = std::chrono::duration<double>(end - start).count();

    cout << fixed << setprecision(8);
    cout << "Price:      " << price << "\n";
    cout << "Delta:      " << delta << "\n";
    cout << "Gamma:      " << gamma << "\n";
    cout << "Vega:       " << vega  << "\n";
    cout << "std:        " << stdev << "\n";
    cout << "Time used:  " << secs  << " s\n";

    return 0;
}
