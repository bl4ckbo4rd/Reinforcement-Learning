#include "problems.h"


void grid_problem(int D1, int D2, int M, int k, int seed_Jab, int seed_dyn, int NI, double g, double n, double L, double F, int CLUSTER){
    
    int NS = D1 * D2;
    int NA = 5;
    
    vector < vector < double > > r(NS, vector <double>(NA, 0.));
    
    r[1][0] = 100;
    r[1][1] = -25;
    r[1][2] = -25;
    r[1][3] = -25;
    r[1][4] = -25;
    
    r[0][2] = 100;
    r[2][1] = 100;
    r[6][3] = 100;
    
    r[11][2] = -100;
    r[12][0] = -100;
    r[13][1] = -100;
    r[7][4]  = -100;
    
    
    BM rbm(NS, NA, M, r, D1);
    rbm.output_handling(k, CLUSTER);
    rbm.init(seed_Jab);
    rbm.Qlearning(NI, seed_dyn, g, n, L, F);
    rbm.printQ();
    rbm.optimal_policy();
    rbm.printLearnedParameters();
    
}

