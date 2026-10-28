#include<iostream>
#include<vector>
#include<cmath>
#include<cstring>
#include<random>
#include<fstream>
#include<string>
#include<sstream>
#include<filesystem>
#include<iomanip>
#include<chrono>
#include<ctime>
#include "zetacal.h"
#include "evcal.h"
#include <mpi.h>
using namespace std;
const double pi = acos(-1);
std::string to_concise_string(double, int digit=-1);
class BD{
    public:
    //parameters
    static const int n_polymer = 100;
    static const int BeadNumber = 11;
    double dt = 5e-7; //5e-8; //2.75e-7; //(s)
    double epsilon_dot = 0; //rate of strain(1/s)
    double KuhnLength = 0.0014938; //um
    double KuhnStep; // decided by input argument
    double LongestRelaxationTime; // it is used when HIModel=0
    bool Hookean = false;
    int HIModel = 3;    // 3 for C2D2
    double c_cstar = 1000.0/3920.0; //c/2590 for PS16; c/3920 for PS7
    double hstar = 0.027; // for PS
    int EVModel = 0;    //soft Gaussina model: E(r) = z*exp(-(r/l/sigma)^2)
    double EV_v = 0; // um^3
    double EV_cutoff = 1e10;
    double Temperature = 298; // K
    double SolventViscosity = 0.059; // Pa*s
    double TotalViscosity = 0.245+0.003*(c_cstar/0.0137);
    double SurfaceTension = 0.031; // N/m
    double MoleculePerUnitVolume; //(1/m^3)
    int iofrq = 2e5;
    vector<vector<vector<double>>> r; //positions of beads; three dimensional, local r, where local means for the mpi process
    double d; // m
    double d1; // m    d & d1 will be set later
    double kBT = 1;
    double NormalStressDiff;
    double GradientV[3]={0};
    double DragCoefficient; //(s/um^2)
    double DragCoefficient_zimm;
    double DragCoefficient_avg;
    double alpha = 0; // a factor that change the drag coefficient to make it be deformation-dependant
    double L; //contour length
    double R_0; //equilibrium end to end distance
    int OutputRate;
    double HenckyStrain;
    double time = 0;
    double timeVEC = 0;
    int world_rank, world_size;
    int start_p, end_p;
    //functions
    void set_r_vector();
    void assign_L_R0();
    void initial();
    void iteration();
    double* F_string(int, int);
    double* rand_3d();
    double mean_Q2[BeadNumber-1] = {};
    double dist_square(double, double, double, double, double, double);
    double NormalStressDifference_polymer();
    double polynomial(double, vector<double>, bool derivative=false);
    void acquire_coefficients(vector<double> &, string);
    void copy_r_vector_to_array(double [n_polymer][3][BeadNumber]);

    //output xyz file
    void output(int, ofstream&, double [n_polymer][3][BeadNumber]);
    void initial_configuration_computation(double);

    string file_path;
    string initial_configuration_file = "/Users/chinaecheremanyanwu/Desktop/MyBD_Results/P=100/N=11/equilibrated_PS7configuration_v=0,FENE/";
};

int main(int argc, char *argv[]){
    // string Radius[5] = {"2mm", "3.5mm", "7mm", "10mm", "20mm"};
    // double MW_list[7] = {0.5e6, 1e6, 2e6, 4e6, 8e6, 12e6, 16e6};
    double MW_list[] = {6.848e6}; // for PS16, 14.96e6 for PS16, 6.848e6 for PS7
    double h0[] = {1.24e-3}; // {2e-3, 3.5e-3, 7e-3, 10e-3, 20e-3, 50e-3, 100e-3}; //SI, diameter
    BD bd;
    double mw = MW_list[stoi(argv[2])]; // g/mol
    bd.KuhnStep = mw/504.4; //mw/137;
    double c = bd.c_cstar/985 * (mw/1e3/(M_PI/6*pow(bd.KuhnLength*1e-6*sqrt(bd.KuhnStep), 3)*6.02e23)); // for PEO, fluid density of 1048; for PS, fluid density of 985
    bd.MoleculePerUnitVolume = c * 6.02e23 * 985 / (mw/1e3); // (1/m^3)
    
    double X = 1;
    // initialize MPI environment
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &bd.world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &bd.world_size);
    srand((stoi(argv[1])+123)*bd.world_rank*std::time(0));

    int polymers_per_process = bd.n_polymer / bd.world_size;
    int remainder = bd.n_polymer % bd.world_size;
    bd.start_p = bd.world_rank * polymers_per_process + std::min(bd.world_rank, remainder);
    bd.end_p = bd.start_p + polymers_per_process + (bd.world_rank < remainder ? 1 : 0);
    int local_polymer_count = bd.end_p - bd.start_p; // the number of polymer calculating in the process
    
    // Prepare counts and displacements for MPI_Gatherv
    // set up two arrays, counts and displs, for receiving r value from each process
    std::vector<int> counts(bd.world_size), displs(bd.world_size);
    for (int i = 0; i < bd.world_size; ++i) {
        int polymers_this_rank = (i < remainder) ? polymers_per_process + 1 : polymers_per_process;
        counts[i] = polymers_this_rank * 3 * bd.BeadNumber;  // 3 dimensions, BeadNumber per polymer
        displs[i] = (i == 0) ? 0 : displs[i-1] + counts[i-1];
    }
    
    
    printf("rank=%d, locl_count=%d, start:%d, end:%d\n", bd.world_rank, local_polymer_count, bd.start_p, bd.end_p);

    bd.LongestRelaxationTime = 0.015; // 0.047s, PS16; 0.015s, PS7

    bd.set_r_vector();
    bd.initial();
    if(bd.HIModel == 3){
        auto [zet, Gcorrn, zb, Nb] = zetacal(1, bd.c_cstar, bd.KuhnStep, bd.hstar, bd.HIModel); // zet when equilibrium, length=R0
        bd.DragCoefficient_zimm = bd.BeadNumber * bd.LongestRelaxationTime * 16*3*(bd.BeadNumber-1)/(2*bd.L*bd.KuhnLength)*pow(sin(pi/(2*bd.BeadNumber)), 2)/zet; // s/um^2
        bd.DragCoefficient_zimm /= bd.BeadNumber;

        ostringstream conc;
        conc << defaultfloat << setprecision(4) << bd.c_cstar;

        bd.file_path = bd.initial_configuration_file+"/../CaBER_visEq/MPI_hk="+to_concise_string(bd.hstar)+
                        "_v="+to_concise_string(bd.EV_v)+"_p="+to_concise_string(bd.n_polymer)+"_2R0="+
                        to_concise_string(h0[stoi(argv[1])]*1e3)+"mm_MW="+to_concise_string(mw/1e6)+"M_c_cstar="+conc.str()+"/";
    }
    else if(bd.HIModel == 0){
        // bd.DragCoefficient_zimm = bd.LongestRelaxationTime * 16*3*(bd.BeadNumber-1)/(2*bd.L*bd.KuhnLength)*pow(sin(pi/(2*bd.BeadNumber)), 2); //s/um^2
        bd.DragCoefficient_zimm = bd.BeadNumber * bd.LongestRelaxationTime * 16*3*(bd.BeadNumber-1)/(2*bd.L*bd.KuhnLength)*pow(sin(pi/(2*bd.BeadNumber)), 2); // s/um^2
        bd.DragCoefficient_zimm /= bd.BeadNumber;

        ostringstream conc;
        conc << defaultfloat << setprecision(4) << bd.c_cstar;

        bd.file_path = bd.initial_configuration_file+"/../CaBER_visEq/MPI_Hookean_FD="+to_concise_string(bd.DragCoefficient_zimm, 2)+"_v="+to_concise_string(bd.EV_v)
                        +"_p="+to_concise_string(bd.n_polymer)+"_2R0="+
                        to_concise_string(h0[stoi(argv[1])]*1e3)+"mm_MW="+to_concise_string(mw/1e6)+"M_c_cstar="+conc.str()+"/";
        
    }
    bd.dt *= bd.LongestRelaxationTime;    
    if(bd.world_rank==0) cout<<"tau_1="<<bd.LongestRelaxationTime<<endl;

    //bulid a new directory if there is no one exist
    if(!filesystem::exists(bd.file_path)){
        filesystem::create_directories(bd.file_path);
    }
    
    ofstream outp[bd.n_polymer];
    if(bd.world_rank == 0) for(int p=0;p<bd.n_polymer;++p) outp[p].open(bd.file_path+"polymer_"+to_string(p+1)+".xyz");
    

    //bd.initial_configuration_computation(bd.LongestRelaxationTime);
    bd.d = h0[stoi(argv[1])];
    bd.NormalStressDiff = bd.NormalStressDifference_polymer();

    double r_global_StaticArray[bd.n_polymer][3][bd.BeadNumber];    // static r global array, used to gather the r value from different processes
    double r_local_StaticArray[bd.n_polymer][3][bd.BeadNumber];

    // three steps to output the trajectory
    // Use MPI_Gatherv to gather the valid portions of r from all processes
    bd.copy_r_vector_to_array(r_local_StaticArray);
    MPI_Gatherv(&r_local_StaticArray[bd.start_p][0][0], local_polymer_count * 3 * bd.BeadNumber, MPI_DOUBLE,
                &r_global_StaticArray[0][0][0], counts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    // if(bd.world_rank==0){
    //     string eq_path = "/Users/chinaecheremanyanwu/Desktop/MyBD_Results/P=100/N=11/equilibrated_PS7configuration_v=0,FENE/";
    //     filesystem::create_directories(eq_path);
    //     for(int p=0;p<bd.n_polymer;++p){
    //         ofstream eq(eq_path+"p_"+to_string(p+1)+".xyz");
    //         if(!eq) MPI_Abort(MPI_COMM_WORLD, 1);
    //         eq << setprecision(17); bd.output(p, eq, r_global_StaticArray);
    //     }
    // }


    // // FiSER start
    // int epoch = 0;
    // if(bd.world_rank==0) printf("FiSER step starts.\n");
    // while(bd.HenckyStrain < 2*log(2)){
    //     epoch ++;
    //     bd.HenckyStrain += bd.dt*bd.epsilon_dot;
    //     bd.iteration();

    //     if((epoch+1) % (bd.iofrq) == 0){
    //         // three steps to output the trajectory
    //         bd.copy_r_vector_to_array(r_local_StaticArray);
    //         MPI_Gatherv(&r_local_StaticArray[bd.start_p][0][0], local_polymer_count * 3 * bd.BeadNumber, MPI_DOUBLE,
    //                     &r_global_StaticArray[0][0][0], counts.data(), displs.data(), MPI_DOUBLE,
    //                     0, MPI_COMM_WORLD);
    //         if(bd.world_rank==0) for(int p=0;p<bd.n_polymer;++p) bd.output(p, outp[p], r_global_StaticArray);
    //     }
    // }
    // if(bd.world_rank==0) printf("FiSER step finished.\n");
    

    // // Visco capillary regime
    // double t0 = h0[stoi(argv[1])]/2 / (2*0.0709*bd.SurfaceTension/bd.TotalViscosity);
    // int epoch = 0;
    // while(true){
    //     epoch ++;
    //     bd.epsilon_dot = 2.0/(t0-bd.time);
    //     bd.time += bd.dt;
    //     bd.HenckyStrain += bd.dt * bd.epsilon_dot;
    //     bd.iteration();

    //     //examine Anna force balance equation
    //     bd.d = 0.0709*2 * bd.SurfaceTension/bd.TotalViscosity*(t0-bd.time);
    //     bd.NormalStressDiff = bd.NormalStressDifference_polymer();
    //     if((3*bd.SolventViscosity*bd.epsilon_dot) > (2*X-1)*2 * bd.SurfaceTension / bd.d - bd.NormalStressDiff){
    //         // three steps to output the trajectory
    //         bd.copy_r_vector_to_array(r_local_StaticArray);
    //         MPI_Gatherv(&r_local_StaticArray[bd.start_p][0][0], local_polymer_count * 3 * bd.BeadNumber, MPI_DOUBLE,
    //                     &r_global_StaticArray[0][0][0], counts.data(), displs.data(), MPI_DOUBLE,
    //                     0, MPI_COMM_WORLD);
    //         if(bd.world_rank==0){
    //             for(int p=0;p<bd.n_polymer;++p) bd.output(p, outp[p], r_global_StaticArray);
    //             printf("epoch = %d, Hencky=%f, time=%f, e=%f, s=%f, drag=%f\n",  epoch, bd.HenckyStrain, bd.time, bd.epsilon_dot, bd.NormalStressDiff, bd.DragCoefficient_avg);
    //             printf("force balance is reached\n");
    //         }
    //         break;
    //     }
        
    //     if((epoch+1) % bd.iofrq == 0){
    //         // three steps to output the trajectory
    //         bd.copy_r_vector_to_array(r_local_StaticArray);
    //         MPI_Gatherv(&r_local_StaticArray[bd.start_p][0][0], local_polymer_count * 3 * bd.BeadNumber, MPI_DOUBLE,
    //                     &r_global_StaticArray[0][0][0], counts.data(), displs.data(), MPI_DOUBLE,
    //                     0, MPI_COMM_WORLD);
    //         if(bd.world_rank==0) for(int p=0;p<bd.n_polymer;++p) bd.output(p, outp[p], r_global_StaticArray);
    //     }
    // }

    // if(bd.world_rank==0) ("VC regime finished.\n");



    //force balance regime
    int epoch = 0;
    while(true){
        epoch ++;
        bd.NormalStressDiff = bd.NormalStressDifference_polymer();
        // bd.epsilon_dot = (1.0/(3*bd.SolventViscosity)) * ((2*X-1)*2 * bd.SurfaceTension/bd.d - bd.NormalStressDiff);
        double A = (985.0*144.0*8.0)*bd.d*bd.d/32.0;
        double B = 3*bd.SolventViscosity;
        double S = (2*X-1)*2*bd.SurfaceTension/bd.d-bd.NormalStressDiff;

        // if(!std::isfinite(B*B+4*A*S) || B*B+4*A*S < 0) MPI_Abort(MPI_COMM_WORLD, 1);
        double discriminant = B*B+4*A*S;
        if(!std::isfinite(discriminant) || discriminant < 0){
            if(bd.world_rank==0){
                cerr << setprecision(17) << "FAILED: time=" << bd.timeVEC
                    << " d=" << bd.d << " stress=" << bd.NormalStressDiff
                    << " capillary=" << (2*X-1)*2*bd.SurfaceTension/bd.d
                    << " discriminant=" << discriminant << endl;
            }
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }


        bd.epsilon_dot = 2*S/(B+sqrt(B*B+4*A*S));

        bd.HenckyStrain += bd.dt*bd.epsilon_dot;
        bd.timeVEC += bd.dt;
        bd.iteration();

        bd.d = bd.d * (1 - bd.epsilon_dot/2*bd.dt); //m(SI)

        // if(bd.d < 1e-6){
        if(bd.d/h0[stoi(argv[1])] <= 1e-3){
            // three steps to output the trajectory
            bd.copy_r_vector_to_array(r_local_StaticArray);
            MPI_Gatherv(&r_local_StaticArray[bd.start_p][0][0], local_polymer_count * 3 * bd.BeadNumber, MPI_DOUBLE,
                        &r_global_StaticArray[0][0][0], counts.data(), displs.data(), MPI_DOUBLE,
                        0, MPI_COMM_WORLD);
            if(bd.world_rank==0){
                for(int p=0;p<bd.n_polymer;++p) bd.output(p, outp[p], r_global_StaticArray);
                printf("epoch = %d, Hencky=%f, time=%f, e=%f, d=%e\n",  epoch, bd.HenckyStrain, bd.timeVEC, bd.epsilon_dot, bd.d);
            }
            break;
        }

        if(epoch % bd.iofrq == 0){
            // three steps to output the trajectory
            bd.copy_r_vector_to_array(r_local_StaticArray);
            MPI_Gatherv(&r_local_StaticArray[bd.start_p][0][0], local_polymer_count * 3 * bd.BeadNumber, MPI_DOUBLE,
                        &r_global_StaticArray[0][0][0], counts.data(), displs.data(), MPI_DOUBLE,
                        0, MPI_COMM_WORLD);
            if(bd.world_rank==0){
                for(int p=0;p<bd.n_polymer;++p) bd.output(p, outp[p], r_global_StaticArray);
                printf("epoch = %d, Hencky=%f, time=%f, e=%f, d=%e\n",  epoch, bd.HenckyStrain, bd.timeVEC, bd.epsilon_dot, bd.d);
            }
        }
    }

    if(bd.world_rank==0) for(int p=0;p<bd.n_polymer;++p) outp[p].close();

    
    MPI_Finalize();

    return 0;    
}

void BD::initial(){
    assign_L_R0();
    HenckyStrain = 0;
    time = 0;
    for(int p=start_p;p<end_p;++p){
        ifstream inf_initial_configuration(initial_configuration_file+"p_"+to_string(p+1)+".xyz");
        string line;
        int beadindex;

        for(int i=0;i<2;++i) getline(inf_initial_configuration, line);
        for(int i=0;i<BeadNumber;++i){
            inf_initial_configuration>>beadindex>>r[p][0][i]>>r[p][1][i]>>r[p][2][i];
        }
        getline(inf_initial_configuration, line); //there is a invisible line of blank info

        inf_initial_configuration.close();
    }
    // // compute end to end distance at equilibrium
    // double E2Edist = 0;
    // for(int p=0;p<n_polymer;++p){
    //     E2Edist += pow(dist_square(r[p][0][0], r[p][1][0], r[p][2][0], r[p][0][BeadNumber-1], r[p][1][BeadNumber-1], r[p][2][BeadNumber-1]), 0.5);
    // }
    // R_0 = E2Edist / n_polymer;
}
void BD::iteration(){
    double local_Q2[BeadNumber-1] = {};
    for(int p=start_p; p<end_p; ++p){
        for(int i=0; i<BeadNumber-1; ++i){
            local_Q2[i] += dist_square(r[p][0][i], r[p][1][i], r[p][2][i], r[p][0][i+1], r[p][1][i+1], r[p][2][i+1]);
        }
    }

    MPI_Allreduce(local_Q2, mean_Q2, BeadNumber-1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    for(int i=0; i<BeadNumber-1; ++i){mean_Q2[i] /= n_polymer;}

    GradientV[0] = -epsilon_dot/2;
    GradientV[1] = -epsilon_dot/2;
    GradientV[2] = epsilon_dot;
    double r_new[n_polymer][3][BeadNumber];
    // memcpy(&r_old, r, BeadNumber*3);
    double* fs; //string force
    double* random_vector; //for brownian motion
    double force[3][BeadNumber];
    double local_DragCoefficient_avg = 0;
    for(int p=start_p;p<end_p;++p){
        // deformation-dependant drag coefficient
        double end_to_end = pow(dist_square(r[p][0][0], r[p][1][0], r[p][2][0], r[p][0][BeadNumber-1], r[p][1][BeadNumber-1], r[p][2][BeadNumber-1]), 0.5);
        auto [zet, Gcorrn, zb, Nb] = zetacal(end_to_end/(sqrt(KuhnStep)*KuhnLength), c_cstar, KuhnStep, hstar, HIModel);
        DragCoefficient = DragCoefficient_zimm * zet;
        local_DragCoefficient_avg += DragCoefficient;
        vector<vector<double>> evforce = evcal(r[p], KuhnLength, KuhnStep/(BeadNumber-1), EV_v, EVModel, EV_cutoff);
        for(int i=0;i<BeadNumber;++i){
            fs = F_string(p, i);
            random_vector = rand_3d();
            for(int j=0;j<3;++j){
                force[j][i] = fs[j];
                r_new[p][j][i] = r[p][j][i] + r[p][j][i]*GradientV[j]*dt + ((fs[j]+evforce[j][i])/DragCoefficient/kBT)*dt + pow(6*dt/DragCoefficient, 0.5)*random_vector[j];
            }
        }
    }
    
    // center of mass at the origin
    double Rcm[3] = {0};
    for(int p=start_p;p<end_p;++p){
        for(int i=0;i<3;++i) Rcm[i] = 0;
        for(int j=0;j<3;++j){
            for(int i=0;i<BeadNumber;++i){
                Rcm[j] += r_new[p][j][i];
            }
            Rcm[j] /= BeadNumber;
        }
        for(int i=0;i<BeadNumber;++i){
            for(int j=0;j<3;++j){
                r[p][j][i] = r_new[p][j][i] - Rcm[j];
            }
        }
    }
    MPI_Allreduce(&local_DragCoefficient_avg, &DragCoefficient_avg, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    DragCoefficient_avg /= n_polymer;
}
double* BD::F_string(int p, int i){
    static double f[3];
    // FENE model is currently active
    double Q0 = L/(BeadNumber-1);
    double H = 3*kBT/(Q0*KuhnLength);
    if(Hookean==false){
        for(int j=0;j<3;++j){
            // if(i==0){
            //     double abs_R0 = pow(dist_square(r[p][0][i], r[p][1][i], r[p][2][i], r[p][0][i+1], r[p][1][i+1], r[p][2][i+1]), 0.5);
            //     f[j] = kBT/KuhnLength * (abs_R0/Q0)* (3-pow(abs_R0/Q0, 2))/(1-pow(abs_R0/Q0, 2)) * (r[p][j][i+1]-r[p][j][i])/abs_R0;

            //     // printf("kun=%f, abs_R0=%f, Q0=%f, r=%f, f=%f, de=%f\n", KuhnLength, abs_R0, Q0, r[p][j][i+1]-r[p][j][i], f[j], (3-pow(abs_R0/Q0, 2)));
            // }
            // else if(i==BeadNumber-1){
            //     double abs_R1 = pow(dist_square(r[p][0][i], r[p][1][i], r[p][2][i], r[p][0][i-1], r[p][1][i-1], r[p][2][i-1]), 0.5);
            //     f[j] = kBT/KuhnLength * (abs_R1/Q0)* (3-pow(abs_R1/Q0, 2))/(1-pow(abs_R1/Q0, 2)) * (r[p][j][i-1]-r[p][j][i])/abs_R1;
            // }
            // else{
            //     double abs_R0 = pow(dist_square(r[p][0][i], r[p][1][i], r[p][2][i], r[p][0][i+1], r[p][1][i+1], r[p][2][i+1]), 0.5);
            //     double abs_R1 = pow(dist_square(r[p][0][i], r[p][1][i], r[p][2][i], r[p][0][i-1], r[p][1][i-1], r[p][2][i-1]), 0.5);
            //     f[j] = kBT/KuhnLength * (abs_R0/Q0)* (3-pow(abs_R0/Q0, 2))/(1-pow(abs_R0/Q0, 2)) * (r[p][j][i+1]-r[p][j][i])/abs_R0;
            //     f[j] += kBT/KuhnLength * (abs_R1/Q0)* (3-pow(abs_R1/Q0, 2))/(1-pow(abs_R1/Q0, 2)) * (r[p][j][i-1]-r[p][j][i])/abs_R1;
            // }
            if(i==0){
                double abs_R0 = pow(dist_square(r[p][0][i], r[p][1][i], r[p][2][i], r[p][0][i+1], r[p][1][i+1], r[p][2][i+1]), 0.5);
                f[j] = H/(1-pow(abs_R0/Q0, 2)) * (r[p][j][i+1]-r[p][j][i]);

                // printf("kun=%f, abs_R0=%f, Q0=%f, r=%f, f=%f, de=%f\n", KuhnLength, abs_R0, Q0, r[p][j][i+1]-r[p][j][i], f[j], (3-pow(abs_R0/Q0, 2)));
            }
            else if(i==BeadNumber-1){
                double abs_R1 = pow(dist_square(r[p][0][i], r[p][1][i], r[p][2][i], r[p][0][i-1], r[p][1][i-1], r[p][2][i-1]), 0.5);
                f[j] = H/(1-pow(abs_R1/Q0, 2)) * (r[p][j][i-1]-r[p][j][i]);
            }
            else{
                double abs_R0 = pow(dist_square(r[p][0][i], r[p][1][i], r[p][2][i], r[p][0][i+1], r[p][1][i+1], r[p][2][i+1]), 0.5);
                double abs_R1 = pow(dist_square(r[p][0][i], r[p][1][i], r[p][2][i], r[p][0][i-1], r[p][1][i-1], r[p][2][i-1]), 0.5);
                f[j] = H/(1-pow(abs_R0/Q0, 2)) * (r[p][j][i+1]-r[p][j][i]);
                f[j] += H/(1-pow(abs_R1/Q0, 2)) * (r[p][j][i-1]-r[p][j][i]);
            }
            // if(i==0){
            //     f[j] = H/(1-mean_Q2[i]/(Q0*Q0)) * (r[p][j][i+1]-r[p][j][i]);

            //     // printf("kun=%f, abs_R0=%f, Q0=%f, r=%f, f=%f, de=%f\n", KuhnLength, abs_R0, Q0, r[p][j][i+1]-r[p][j][i], f[j], (3-pow(abs_R0/Q0, 2)));
            // }
            // else if(i==BeadNumber-1){
            //     f[j] = H/(1-mean_Q2[i-1]/(Q0*Q0)) * (r[p][j][i-1]-r[p][j][i]);
            // }
            // else{
            //     f[j] = H/(1-mean_Q2[i]/(Q0*Q0)) * (r[p][j][i+1]-r[p][j][i]);
            //     f[j] += H/(1-mean_Q2[i-1]/(Q0*Q0)) * (r[p][j][i-1]-r[p][j][i]);
            // }
        }
    }
    // Hookean spring
    else{
        for(int j=0;j<3;++j){
            if(i==0){
                double abs_R0 = pow(dist_square(r[p][0][i], r[p][1][i], r[p][2][i], r[p][0][i+1], r[p][1][i+1], r[p][2][i+1]), 0.5);
                f[j] = kBT/KuhnLength * (abs_R0/Q0)* (3)/(1) * (r[p][j][i+1]-r[p][j][i])/abs_R0;
            }
            else if(i==BeadNumber-1){
                double abs_R1 = pow(dist_square(r[p][0][i], r[p][1][i], r[p][2][i], r[p][0][i-1], r[p][1][i-1], r[p][2][i-1]), 0.5);
                f[j] = kBT/KuhnLength * (abs_R1/Q0)* (3)/(1) * (r[p][j][i-1]-r[p][j][i])/abs_R1;
            }
            else{
                double abs_R0 = pow(dist_square(r[p][0][i], r[p][1][i], r[p][2][i], r[p][0][i+1], r[p][1][i+1], r[p][2][i+1]), 0.5);
                double abs_R1 = pow(dist_square(r[p][0][i], r[p][1][i], r[p][2][i], r[p][0][i-1], r[p][1][i-1], r[p][2][i-1]), 0.5);
                f[j] = kBT/KuhnLength * (abs_R0/Q0)* (3)/(1) * (r[p][j][i+1]-r[p][j][i])/abs_R0;
                f[j] += kBT/KuhnLength * (abs_R1/Q0)* (3)/(1) * (r[p][j][i-1]-r[p][j][i])/abs_R1;
            }
        }
    }
    return f;
}
double BD::NormalStressDifference_polymer() {
    double local_Q2[BeadNumber-1] = {};

    for(int p=start_p; p<end_p; ++p){
        for(int i=0; i<BeadNumber-1; ++i){
            local_Q2[i] += dist_square(r[p][0][i], r[p][1][i], r[p][2][i], r[p][0][i+1], r[p][1][i+1], r[p][2][i+1]);
        }
    }

    MPI_Allreduce(local_Q2, mean_Q2, BeadNumber-1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    for(int i=0; i<BeadNumber-1; ++i){ mean_Q2[i] /= n_polymer; }

    double Q0 = L / (BeadNumber - 1);
    double H = 3*kBT/(Q0*KuhnLength);
    double f[3];
    double NSD_local = 0;

    for (int p = start_p; p < end_p; ++p) {
        double NSD_p = 0.0;
        for (int i = 0; i < BeadNumber - 1; ++i) {
            double abs_R0 = pow(dist_square(r[p][0][i], r[p][1][i], r[p][2][i], r[p][0][i + 1], r[p][1][i + 1], r[p][2][i + 1]), 0.5);
            for (int j = 1; j < 3; ++j) {
            //     f[j] = kBT / KuhnLength * (abs_R0 / Q0) * (3 - pow(abs_R0 / Q0, 2)) / (1 - pow(abs_R0 / Q0, 2)) * (r[p][j][i + 1] - r[p][j][i]) / abs_R0;
            // }
                // f[j] = H/(1-mean_Q2[i]/(Q0*Q0)) * (r[p][j][i+1]-r[p][j][i]);
                f[j] = H/(1-pow(abs_R0/Q0, 2)) * (r[p][j][i+1]-r[p][j][i]);
            }
            NSD_p += f[2] * (r[p][2][i + 1] - r[p][2][i]);
            NSD_p -= f[1] * (r[p][1][i + 1] - r[p][1][i]);
        }
        NSD_local += NSD_p;
    }

    // Sum NSD values from all processes
    double NSD_total = 0.0;
    MPI_Allreduce(&NSD_local, &NSD_total, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    // Average over all polymers
    NSD_total /= n_polymer;

    // Final result: multiply by constants
    double NSD_avg = MoleculePerUnitVolume * NSD_total * (1.38e-23 * Temperature);

    return NSD_avg;
}
double* BD::rand_3d(){
    static double rand_v[3];
    for(int i=0;i<3;++i){
        rand_v[i] = ((double) rand() / (RAND_MAX)) * 2 - 1;
    }
    return rand_v;
}
double BD::dist_square(double x1, double y1, double z1, double x2, double y2, double z2){
    return pow(x2-x1, 2) + pow(y2-y1, 2) + pow(z2-z1, 2);
}
void BD::output(int p,ofstream& outf, double r_global[n_polymer][3][BeadNumber]){
    outf<<BeadNumber<<endl;
    outf<<"comment line. Hencky="<<HenckyStrain<<" strain_rate="<<epsilon_dot<<" DragCoef="<<DragCoefficient_avg<<" tIC="<<time<<" tVEC="<<timeVEC<<" d="<<d
    <<" Stress="<<NormalStressDiff<<" dt="<<dt<<endl;
    for(int i=0;i<BeadNumber;++i){
        outf<<i<<" "<<r_global[p][0][i]<<" "<<r_global[p][1][i]<<" "<<r_global[p][2][i]<<endl;
    }
}
void BD::set_r_vector(){
    //initial positions
    for(int p=0;p<n_polymer;++p){
        vector<vector<double>> direction;
        r.push_back(direction);
    }
    for(int p=0;p<n_polymer;++p){
        for(int d=0;d<3;++d){
            vector<double> beads;
            r[p].push_back(beads);
        }
    }
    for(int p=0;p<n_polymer;++p){
        for(int d=0;d<3;++d){
            for(int b=0;b<BeadNumber;++b){
                r[p][d].push_back(0);
            }
        }
    }
}
void BD::assign_L_R0(){
    L = KuhnLength * KuhnStep;
    R_0 = sqrt(KuhnStep) * KuhnLength;
}
void BD::copy_r_vector_to_array(double array[n_polymer][3][BeadNumber]){
    for(int p=0;p<n_polymer;++p){
        for(int d=0;d<3;++d){
            for(int b=0;b<BeadNumber;++b){
                array[p][d][b] = r[p][d][b];
            }
        }
    }
}
void BD::initial_configuration_computation(double RelaxationTime){
    double eps_original = epsilon_dot;
    epsilon_dot = 0;
    int epoch = round(5*RelaxationTime/dt);
    for(int i=0;i<epoch;++i){
        if(i%(epoch/10)==0){
            if(world_rank==0) cout<<"initial equilibrium: "<<i<<"/"<<epoch<<endl;
        }
        iteration();
    }
    epsilon_dot = eps_original;
    printf("equilibrium reached!\n");
}
double BD::polynomial(double x, vector<double> coeffs, bool derivative){
    int order = coeffs.size() - 1;
    double y = 0;
    if(derivative == false){
        for(int i=0;i<coeffs.size();++i){
            y += coeffs[i] * pow(x, i);
        }
        return y;
    }
    else{
        for(int i=0;i<order;++i){
            y += coeffs[i+1] * (i+1) * pow(x, i);
        }
        return y;
    }
}
void BD::acquire_coefficients(vector<double> &coeffs, string path){
    ifstream inf;
    inf.open(path);
    while(true){
        double a=0;
        inf >> a;
        coeffs.push_back(a);
        if(inf.eof()) break;
    }
    if(coeffs.back() == 0) coeffs.pop_back();
}




std::string to_concise_string(double value, int digit) {
    std::ostringstream out;
    
    if (digit == -1) {
        out << std::setprecision(15) << value; // High precision to capture full accuracy
        std::string str = out.str();

        // Remove trailing zeros and decimal point if necessary
        if (str.find('.') != std::string::npos) {
            // Remove trailing zeros
            str.erase(str.find_last_not_of('0') + 1, std::string::npos);
            // Remove the decimal point if it's the last character
            if (str.back() == '.') {
                str.pop_back();
            }
        }
        return str;
    } else {
        // Use scientific notation for very small or very large numbers
        if (value != 0.0 && (std::abs(value) < 1e-4 || std::abs(value) >= 1e6)) {
            out << std::scientific << std::setprecision(digit) << value;
        } else {
            out << std::fixed << std::setprecision(digit) << value;
        }
        return out.str();
    }
}