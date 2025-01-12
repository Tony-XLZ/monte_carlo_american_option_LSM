#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <chrono>

// --------------------------- //
//    Simple Quadratic Polynomial Regression Function    //
// --------------------------- //
bool solve3x3(const double A[9], const double B[3], double X[3]) {
    // A stored in row-major order: [a11, a12, a13, a21, a22, a23, a31, a32, a33]
    double mat[9];
    double vec[3];
    for(int i = 0; i < 9; i++) mat[i] = A[i];
    for(int i = 0; i < 3; i++) vec[i] = B[i];

    // Gaussian Elimination
    for(int i = 0; i < 3; ++i) {
        double pivot = mat[i*3 + i];
        if(std::fabs(pivot) < 1e-14) return false;

        // Normalize
        for(int col = i; col < 3; col++){
            mat[i*3 + col] /= pivot;
        }
        vec[i] /= pivot;

        // Eliminate
        for(int row = i+1; row < 3; row++){
            double factor = mat[row*3 + i];
            for(int col = i; col < 3; col++){
                mat[row*3 + col] -= factor * mat[i*3 + col];
            }
            vec[row] -= factor * vec[i];
        }
    }
    // Back Substitution
    for(int i = 2; i >= 0; i--){
        double sum = vec[i];
        for(int col = i+1; col < 3; col++){
            sum -= mat[i*3 + col]*X[col];
        }
        X[i] = sum;
    }
    return true;
}

bool polyfit2(const std::vector<double>& X,
              const std::vector<double>& Y,
              double& a, double& b, double& c)
{
    size_t n = X.size();
    if(n < 3) return false;

    double sumX=0.0, sumX2=0.0, sumX3=0.0, sumX4=0.0;
    double sumY=0.0, sumXY=0.0, sumX2Y=0.0;

#pragma omp parallel for reduction(+:sumX,sumX2,sumX3,sumX4,sumY,sumXY,sumX2Y)
    for(int i = 0; i < (int)n; i++){
        double x  = X[i];
        double y  = Y[i];
        double x2 = x*x;
        sumX  += x;
        sumX2 += x2;
        sumX3 += x2*x;
        sumX4 += x2*x2;
        sumY  += y;
        sumXY += x*y;
        sumX2Y+= x2*y;
    }

    double A[9] = {
        (double)n, sumX,   sumX2,
        sumX,      sumX2,  sumX3,
        sumX2,     sumX3,  sumX4
    };
    double B[3] = { sumY, sumXY, sumX2Y };
    double sol[3];
    if(!solve3x3(A, B, sol)) return false;

    a = sol[0]; b = sol[1]; c = sol[2];
    return true;
}

int main(){
    // ---------------- Parameters ----------------
    const double S0    = 60.0;  // Initial price
    const double K     = 60.0;  // Strike price
    const double T     = 0.25;  // Time to maturity (years)
    const double r     = 0.1;   // Risk-free rate
    const double sigma = 0.45;  // Volatility
    const int    M     = 30;    // Number of time steps
    const int    N     = 100000; // Number of simulation paths

    const double dt   = T / M; // Time step
    const double disc = std::exp(-r*dt); // Discount factor

    // Record the start time
    auto start = std::chrono::high_resolution_clock::now();

    // -------------- Path Generation --------------
    // Use float to save memory bandwidth
    std::vector<float> S;
    S.resize((size_t)N * (M+1));

    // Parallel initialization
#pragma omp parallel for
    for(int i = 0; i < N; i++){
        S[(size_t)i*(M+1)] = (float)S0;
    }

    std::mt19937_64 rng(123456789ULL);
    std::normal_distribution<double> normDist(0.0, 1.0);

    for(int t = 1; t <= M; t++){
#pragma omp parallel for
        for(int i = 0; i < N; i++){
            double z = normDist(rng);
            float prevS = S[(size_t)i*(M+1) + (t-1)];
            // Stock evolution
            float St = prevS * std::exp( (r - 0.5f*sigma*sigma)*dt
                                         + sigma*std::sqrt(dt)*z );
            S[(size_t)i*(M+1) + t] = St;
        }
    }

    // ----------------- Terminal Payoff -----------------
    // Use double to store option value V
    std::vector<double> V(N);
#pragma omp parallel for
    for(int i = 0; i < N; i++){
        float ST = S[(size_t)i*(M+1) + M];
        V[i] = std::max<double>(K - ST, 0.0);
    }

    // -------------- Longstaff–Schwartz Backward Induction --------------
    for(int t = M-1; t >= 1; t--){
        // 1) Collect in-the-money paths in parallel
        std::vector<int> idxITM(N); // Temporary storage for indices
        int idxCount = 0; // Count of valid indices

#pragma omp parallel
        {
            std::vector<int> localIdx;
            localIdx.reserve(10000); // Reserve memory to reduce reallocations

#pragma omp for nowait
            for(int i=0; i<N; i++){
                float St = S[(size_t)i*(M+1)+t];
                if(St < K){
                    localIdx.push_back(i);
                }
            }
            // Merge into global idxITM using atomic operations
#pragma omp critical
            {
                int oldPos = idxCount;
                idxCount += (int)localIdx.size();
                for(size_t k=0; k<localIdx.size(); k++){
                    idxITM[oldPos + (int)k] = localIdx[k];
                }
            }
        }

        if(idxCount == 0){
            // No in-the-money paths
#pragma omp parallel for
            for(int i=0; i<N; i++){
                V[i] *= disc;
            }
            continue;
        }

        // 2) Build (X, Y) for regression in parallel
        std::vector<double> X, Y;
        X.resize(idxCount);
        Y.resize(idxCount);

#pragma omp parallel for
        for(int k=0; k<idxCount; k++){
            int i = idxITM[k];
            float St = S[(size_t)i*(M+1)+t];
            double contV = V[i] * disc; // Discounted continuation value
            X[k] = (double)St;
            Y[k] = contV;
        }

        double a=0.0, b=0.0, c=0.0;
        bool fitted = polyfit2(X, Y, a, b, c);

        // 3) Update option values in parallel
#pragma omp parallel for
        for(int k=0; k<idxCount; k++){
            int i = idxITM[k];
            float St = S[(size_t)i*(M+1)+t];
            double exercise = K - (double)St;
            double contVal  = fitted ? (a + b*St + c*St*St) : 0.0;

            if(exercise > contVal){
                V[i] = exercise; // Immediate exercise
            } else {
                V[i] = Y[k]/disc; // Continuation
            }
        }

        // 4) Apply discount to all option values
#pragma omp parallel for
        for(int i=0; i<N; i++){
            V[i] *= disc;
        }
    }

    // ------------- Calculate Initial Option Price -------------
    double sumV = 0.0;
#pragma omp parallel for reduction(+:sumV)
    for(int i = 0; i < N; i++){
        sumV += V[i];
    }
    double optionPrice = sumV / N;

    // -------------- Time Statistics --------------
    auto end = std::chrono::high_resolution_clock::now();
    auto ms  = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "American Put Option Price: " << optionPrice << std::endl;
    if (ms < 1000) {
        std::cout << "Program Execution Time: " << ms << " ms\n";
    } else {
        int sec = (int)(ms / 1000);
        int msec= (int)(ms % 1000);
        std::cout << "Program Execution Time: " << sec << " seconds " << msec << " ms\n";
    }

    return 0;
}
