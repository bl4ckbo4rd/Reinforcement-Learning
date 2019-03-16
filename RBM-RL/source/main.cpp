#include "problems.h"

int main (int argc, char *argv[]){
    
    //M         : # of hidden units
    //n         : learning rate
    //g         : discount rate
    //L         : regularizer
    //F         : momentum
    //seed_Jab  : seed for the initialization of J (couplings)
    //seed_dyn  : seed for the initialization of the dynamics
    //CLUSTER   : set to 1 to run on cluster, 0 otherwise
    //k         : label used to index the example studied
    
    
    double n;
    double L, F;
    int k, seed_Jab, seed_dyn, NI, CLUSTER;
    double g;
    
    if(argc==10){
        int i = 1;
        n           = atof(argv[i++]);
        g           = atof(argv[i++]);
        L           = atof(argv[i++]);
        F           = atof(argv[i++]);
        seed_Jab    = atoi(argv[i++]);
        seed_dyn    = atoi(argv[i++]);
        NI          = atoi(argv[i++]);
        CLUSTER     = atoi(argv[i++]);
        k           = atoi(argv[i++]);
    }
    else{
        cout << "argument: n, g, L, F, seed_Jab, seed_dyn, NI, CLUSTER, k" << endl;
        return 0;
    }
    
    int M = 16;
    int D1 = 5;
    int D2 = 3;
    
    grid_problem(D1, D2, M, k, seed_Jab, seed_dyn, NI, g, n, L, F, CLUSTER);
    
    return 1;
}
