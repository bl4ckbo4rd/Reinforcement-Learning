#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <stdlib.h>
#include <random>
#include <iterator>
#include <armadillo>

using namespace std;
using namespace arma;

class BM{
public:
    
    int NS;                                                                                              //# of states
    int NA;                                                                                              //# of actions
    int N;                                                                                               //# of visible nodes = NS + NA
    int M;                                                                                               //# of hidden nodes
    
    int D1;                                                                                              //orizonthal side of the grid
    
    vector < vector <double> > r;                                                                        //reward function, problem dependent
    
    ofstream file_FreeEnergy;                                                                            //we store in this file the free energy of each
                                                                                                         //action during the Q-learning
    
    ofstream file_sJ;                                                                                    //file containing the singular values of the matrix J during Q-learning
    ofstream fileJ;                                                                                      //file containing the learned coupling matrix J
    
    BM(int p_NS, int p_NA, int p_M, vector < vector <double> >& p_r, int p_D1);
    
    vector <int> v;   		                                                                             //visible units, N = NS + NA
    vector <int> h;                                                      		                         //hidden units,  M
    
    vector <double> mean_h;                                                                              //<h> computed with P(h|v), with v given by {s,a}
    
    vector <int> pi;                                                                                     //optimal policy
    
    vector < vector <double> > Q;                                                                        //Q functions
    
    vector < vector <double> > J;                                                                        //J is a N times M matrix of couplings
    vector < vector <double> > D_J;                                                                      //D_J is a N times M matrix of difference of couplings
    vector < vector <double> > D_J_p;                                                                    //D_J_p is a N times M matrix of couplings at the previous iteratios step
    
    void init(int);                                                                                      //this method initializes the initial J with a N(0,0.1) distribution
                                                                                                         //and initializes the Q(s,a) with - F(s,a) computed with these weights
    
    void RBM_sampler(vector <int>& sa);                                                                  //this method computes <h> using a RBM
    
    void BM_sampler(vector <int>& sa);                                                                   //this method computes <h> using a BM

    int boltzmann_action_sampler(int s);                                                                 //this method samples a new action with a boltzmann distribution
                                                                                                         //per codificare le azioni: 0 resta nel posto dove stai
                                                                                                         //                          1 sx
                                                                                                         //                          2 dx
                                                                                                         //                          3 sopra
                                                                                                         //                          4 sotto
    
                                                                                                         //in the state s, ~ e^Q(s,a).

    int get_new_state(int s, int a);                                                                     //this method returns the a new state given the original state, and
                                                                                                         //the action chosen.
    
    void Qlearning(int NI, int seed_dyn, double g, double n, double L, double F);                        //this method performs the Q-learning
                                                                                                         //qui dovresti fare aggiungere all'inizio del learning una fase in cui esplora tutte le azioni possibili
    
    int opt_action_finder(vector <double>& Qs, int s);                                                   //this method finds the optimal action in state s
                                                                                                         //maximizing Q(s,a) over a
    
    void printLearnedParameters();                                                                       //print parameters at the end of the learning
    
    void SVD_J();                                                                                        //method that performs the SVD of the coupling matrix
    
    void output_handling(int k, int CLUSTER);                                                            //method that handles all the output files
    
    double freeEnergy_RBM(vector <int> v);                                                               //this method that computes the free energy of
                                                                                                         //a given vector v = {s,a}
    
    void compute_F_duringQL();                                                                           //this method computes the free energy of each action in each state
                                                                                                         //during Q-learning after each sweep (~after each NS steps)
    
    void printQ();                                                                                       //this method prints for each state s the values of Q(s,a)

    void optimal_policy();                                                                               //methods that prints the optimal action for each state s
    
};



