#include "rbm.h"


//----------------------------------------------------------------------------------------------------------------------------------------------------------------------BM constructor//


BM::BM(int p_NS, int p_NA, int p_M, vector < vector < double > >& p_r, int p_D1){
    
    NS = p_NS;
    NA = p_NA;
    N  = NS + NA;
    M  = p_M;
    
    D1 = p_D1;
    
    v.resize(N,0);
    
    h.resize(M,0);
    
    r.resize(NS, vector < double> (NA, 0.));
    
    r = p_r;
    
}



//------------------------------------------------------------------------------------------------------------------------------------------------------------------------BM init weight and Q functions //

void BM::init(int seed_Jab){
    
    std::mt19937 gen(seed_Jab);
    std::normal_distribution<double> d(0.,0.1);
    
    J.resize(N);
    for (int i = 0; i < N; i ++){
        J[i].resize(M);
        for (int j = 0; j < M; j ++){
            J[i][j] = d(gen);
        }
    }
    
    D_J.resize(N);
    for (int i = 0; i < N; i ++){
        D_J[i].resize(M,0.);
    }
    
    D_J_p.resize(N);
    for (int i = 0; i < N; i ++){
        D_J_p[i].resize(M,0.);
    }
    
    mean_h.resize(M,0.);
    
    Q.resize(NS, vector <double>(NA,0.));
    
    vector <int> v(NS + NA, 0);
    
    for (int s = 0; s < NS; s++){
        for (int a = 0; a < NA; a++){
            v[s] = 1;
            v[NS+a] = 1;
            
            Q[s][a] = - freeEnergy_RBM(v);
            
            v[s] = 0;
            v[NS+a] = 0;
            
        }
    }
    
    
    
}


int BM::get_new_state(int s, int a){
    
    int c = 1;
    
    vector <int> v;
    
    v.push_back(s);
    v.push_back(s - 1);
    v.push_back(s + 1);
    v.push_back(s - D1);
    v.push_back(s + D1);
    
    return v[a];
    
}


int BM::boltzmann_action_sampler(int s){
    
    int c = 1;
    
    vector <int> v;
    
    v.push_back(0);
    
    if (s % D1){
        v.push_back(1);
        c++;
    }
    if ( (s + 1) % D1){
        v.push_back(2);
        c ++;
    }
    if (s - D1 >= 0){
        v.push_back(3);
        c ++;
    }
    if (s + D1 < NS){
        v.push_back(4);
        c ++;
    }
    
    double Z = 0.;
    vector <double> P(v.size(), 0.);
    vector <double> C(v.size(), 0.);
 
    for (int i = 0; i < v.size(); i++){
        int a = v[i];
        Z += exp(Q[s][a]);
    }

    for (int i = 0; i < v.size(); i++){
        int a = v[i];
        P[i] = exp(Q[s][a]) / Z;
    }
    
    C[0] = P[0];
    for (int i = 1; i < C.size(); i++)
        C[i] = C[i-1] + P[i];
    
    double r = (double)rand() / RAND_MAX;
    int a;
    
    if (r < C[0])
        a = 0;
    else{
        for (int i = 0; i < v.size()-1; i++){
            if ( r > C[i] && r < C[i+1] ){
                a = v[i+1];
            }
        }
    }
    
    return a;
    
}


void BM::Qlearning(int NI, int seed_dyn, double g, double n, double L, double F){

    int min_s = 0;
    int max_s = NS - 1;
    
    int signJ;
    
    srand(seed_dyn);
    
    //Boltzmann dynamics
    
    for (int t = 0; t < NI; t++){
        
        //sample randomly a state
        int s  = rand() % (max_s - min_s + 1) + min_s;
        
        //sample action with Boltzmann
        int a  = boltzmann_action_sampler(s);
        
        //get the corresponding state
        int ns = get_new_state(s,a);
        //maximize over possible actions
        int na = opt_action_finder(Q[ns],ns);

        //Q(s,a) = - F(s,a);
        vector <int> v(NS + NA, 0);
        v[s] = 1;
        v[NS+a] = 1;
        Q[s][a] = - freeEnergy_RBM(v);
        
        //Q(ns,na) = - F(ns,na);
        vector <int> nv(NS + NA, 0);
        nv[ns] = 1;
        nv[NS+na] = 1;
        Q[ns][na] = - freeEnergy_RBM(nv);
        
        double AQ = r[s][a] + g * Q[ns][na] - Q[s][a];
        
        RBM_sampler(v);

        cout << s << " " << a << " " << ns << " " << na << " --------------------------> " << r[s][a] << " " << Q[s][a] << " " << Q[ns][na] << " " << AQ  << endl;

        for (int j = 0; j < M; j ++){
            
            if (J[s][j] > 0)
                signJ = 1;
            else
                signJ = -1;
            
            D_J[s][j]  = n * ( AQ * v[s] * mean_h[j]  - L * signJ );
            
            J[s][j]    = J[s][j] + D_J[s][j] + F * D_J_p[s][j];
            
            
            if (J[NS+a][j] > 0)
                signJ = 1;
            else
                signJ = -1;
            
            D_J[NS+a][j]  = n * ( AQ * v[NS+a] * mean_h[j]  - L * signJ );
            
            J[NS+a][j]    = J[NS+a][j] + D_J[NS+a][j] + F * D_J_p[NS+a][j];
            
        }
        
        
        D_J_p = D_J;
        
        
        //Q(s,a) = - F(s,a);
        vector <int> v1(NS + NA, 0);
        v1[s] = 1;
        v1[NS+a] = 1;
        Q[s][a] = - freeEnergy_RBM(v1);
        
        //Q(ns,na) = - F(ns,na);
        vector <int> nv1(NS + NA, 0);
        nv1[ns] = 1;
        nv1[NS+na] = 1;
        Q[ns][na] = - freeEnergy_RBM(nv1);
        
        cout << "dopo aver aggiornato i pesi" << " ------> " << Q[s][a] << " " << Q[ns][na] << endl;
        cout << endl;
        
        if (t % NS)
            compute_F_duringQL();

        
    }
    
}


void BM::optimal_policy(){
    
    pi.resize(NS);
    
    for (int s = 0; s < NS; s++){
        pi[s] = opt_action_finder(Q[s],s);
        cout << s << " " << pi[s] << endl;
    }

}

int BM::opt_action_finder(vector <double>& Qs, int s){
    
    double eps = 0.01;
    
    int c = 1;
    
    vector <int> v;
    
    v.push_back(0);
    
    if (s % D1){
        v.push_back(1);
        c++;
    }
    if ( (s + 1) % D1){
        v.push_back(2);
        c++;
    }
    if (s - D1 >= 0){
        v.push_back(3);
        c++;
    }
    if (s + D1 < NS){
        v.push_back(4);
        c++;
    }
    
    double max = Qs[v[0]];
    double na  = v[0];
    
    for (int a = 0; a < c; a++){
        if (Qs[v[a]] > max + eps){
            max = Qs[v[a]];
            na = v[a];
        }
        if ( abs(Qs[v[a]] - max) < eps ){
            if (0.5 > (double)rand()/RAND_MAX)
                na = v[a];
        }
    }
    
    return na;
    
}


void BM::RBM_sampler(vector <int>& sa){
   
    double h_field;
    
    for (int i = 0; i < N; i ++)
        v[i] = sa[i];
    
    
    //update h
    for (int j = 0; j < M; j ++){
        
        h_field = 0.;
        for (int i = 0; i < N; i ++){
            h_field += J[i][j] * v[i];
        }
        
        mean_h[j] = 1. / ( 1 + exp ( - h_field ) );
        
    }
    
    SVD_J();
    
}




void BM::SVD_J(){
    
    //compute the svd of the weight matrix:
    
    //A is a N * M matrix equal to J.
    //This matrix is decomposed into a matrix U_J, which is N * N, a matrix S_J, which is N * M and a matrix V_J that is M * M.
    //Since the matrix S has only M diagonal elements different from 0, only the first M column vectors of U_J are important: they are contained in sU_J
    
    mat A(N,M);
    
    mat U_J;
    vec S_J;
    mat V_J;
    
    for (int i = 0; i < N; i ++){
        for (int j = 0; j < M; j ++){
            A(i,j) = J[i][j];
        }
    }
    
    svd(U_J,S_J,V_J,A);
    
    for (int i = 0; i < M; i++)
         file_sJ << S_J(i,0) << " ";
    
    file_sJ << endl;

}


void BM::output_handling(int k, int CLUSTER){
    
    string path1;
    
    if (CLUSTER)
        path1 = "/home/jrocchi/DATA/LEARNING/REINF-LEARNING";
    else
        path1 = "/Users/jrocchi/DATA/LEARNING/REINF-LEARNING";
    
    string data_sJ = path1 + "/file_sJ_" + to_string(k);
    file_sJ.open(data_sJ);
    
    if (!file_sJ)
    {
        cerr << "Uh oh, file singular values couplings could not be open for writing!" << endl;
        exit(1);
    }
    
    string dataJ = path1 + "/file_J_" + to_string(k);
    fileJ.open(dataJ);
    
    if (!fileJ)
    {
        cerr << "Uh oh, fileJ could not be open for writing!" << endl;
        exit(1);
    }
    
    string dataF = path1 + "/file_FreeEnergy_" + to_string(k);
    file_FreeEnergy.open(dataF);
    
    if (!file_FreeEnergy)
    {
        cerr << "Uh oh, file Free Energy could not be open for writing!" << endl;
        exit(1);
    }
    
    
}

void BM::printLearnedParameters(){
    
    for (int i = 0; i < N; i ++){
        for (int j = 0; j < M; j ++){
            fileJ << J[i][j] << " ";
        }
        fileJ << endl;
    }
    
}

void BM::printQ(){
    
    for (int s = 0; s < NS; s ++){
        cout << "state " << s << " ---> Q: ";
        for (int a = 0; a < NA; a ++)
            cout << Q[s][a] << " ";
        cout << endl;
    }
    
}


double BM::freeEnergy_RBM(vector <int> v){
    
    int i, j;
    
    double mean_h, h_field, E, S, F;
    
    vector <double> p_h(M, 0.0);
    
    for (j = 0; j < M; j ++){
        h_field = 0.;
        for (i = 0; i < N; i ++){
            h_field += J[i][j] * v[i];
        }
        p_h[j] = 1./(1 + exp (- h_field));
    }
    
    E = 0.;
    S = 0.;
    
    for (j = 0; j < M; j ++){
        for (i = 0; i < N; i ++){
            mean_h = p_h[j];
            E -= J[i][j] * mean_h * v[i];
        }
    }
    
    for (j = 0; j < M; j ++){
        S -= ( p_h[j] * log(p_h[j]) + (1-p_h[j]) * log(1-p_h[j]) );
    }
    
    F = E - S;
    
    return F;
    
}


void BM::compute_F_duringQL(){
    
    vector <int> v(NS + NA, 0);
    
    for (int s = 0; s < NS; s++){
        for (int a = 0; a < NA; a++){
            v[s] = 1;
            v[NS+a] = 1;
            
            Q[s][a] = - freeEnergy_RBM(v);
            
            v[s] = 0;
            v[NS+a] = 0;
            
            file_FreeEnergy << Q[s][a] << " ";
            
        }
    }
    
}



