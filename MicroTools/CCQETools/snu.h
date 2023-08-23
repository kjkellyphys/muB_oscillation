#ifndef __SNU_H
#define __SNU_H

#include <gsl/gsl_matrix.h>
#include <globes/globes.h>

// Use NuSQuIDS instead of GLoBES for probabilities
//#define NU_USE_NUSQUIDS
extern glb_probability_nusquids_function nu_hook_probability_matrix_nusquids;

// Options for Ivan's oscillation + decay code
//Flag removed Oct 2018  #define OSC_DECAY_PSEUDOSCALAR  false   // pseudoscalar or scalar interactions?
#define OSC_DECAY_MAJORANA      false   // Majorana or Dirac neutrinos? FIXME FIXME


// Arrangement of oscillation parameters in glb_params data structure:
//   th12,    th13,    th23,    deltaCP,
//   dm21,    dm31,    dm41,    dm51,    ...,
//   th14,    th24,    th34,    th15,    th25,    ...,
//   delta_1, delta_2, ...
//
//   |\eps|^s_{ee},   \phi^s_{ee},   ..., |\eps^s_{sn,e}|,  \phi^s_{sn,e},
//                ...                ...                 ...
//   |\eps|^s_{e,sn}, \phi^s_{e,sn}, ..., |\eps^s_{sn,sn}|, \phi^s_{sn,sn},
//
//   \eps^m_{ee},     |\eps^m_{e\mu}|, \phi^m_{e\mu},  ...,  |\eps^m_{e,sn}|,   \phi^m_{e,sn}
//                    \eps^m_{\mu\mu},                 ...,  |\eps^m_{\mu,sn}|, \phi^m_{\mu,sn}
//                                                     ...         ...
//                                                           \eps^m_{sn,sn}
//
//   |\eps|^d_{ee},   \phi^d_{ee},   ..., |\eps^d_{sn,e}|,  \phi^d_{sn,e},
//                ...                ...                 ...
//   |\eps|^d_{e,sn}, \phi^d_{e,sn}, ..., |\eps^d_{sn,sn}|, \phi^d_{sn,sn}

// Names of oscillation parameters
extern char snu_param_strings[][64];

// Maximum number of neutrino species and related quantities
#define SNU_MAX_FLAVORS   5
//#define SNU_MAX_PARAMS    (6*(SNU_MAX_FLAVORS)*(SNU_MAX_FLAVORS) - SNU_MAX_FLAVORS)
#define SNU_MAX_PARAMS    (6*(SNU_MAX_FLAVORS)*(SNU_MAX_FLAVORS) + 10)
            // this is deliberately generous to allow for redundant parameters
#define SNU_MAX_ANGLES    ((SNU_MAX_FLAVORS * (SNU_MAX_FLAVORS-1))/2)
#define SNU_MAX_PHASES    (((SNU_MAX_FLAVORS-1)*(SNU_MAX_FLAVORS-2))/2)

// A flag that switches on special treatment of MINOS ND
#define MINOS_ND_PROBABILITY 777


// Data structure for holding probability table
struct snu_probability_table
{
  double *probability_table;       // Buffer holding probabilities
  int n_p;                         // Number of parameters scanned over in table
  glb_params default_values;       // Input values for those parameters not scanned over
  char *params[SNU_MAX_PARAMS+1];  // Names of parameters
  double p_min[SNU_MAX_PARAMS+1];  // Lower edges of the scan range
  double p_max[SNU_MAX_PARAMS+1];  // Upper edges of the scan range
  int p_steps[SNU_MAX_PARAMS+1];   // Number of steps in the scan
  unsigned long p_flags[SNU_MAX_PARAMS+1]; // Flags for each parameter (e.g. DEG_LOGSCALE)
};


// Function declarations
// ---------------------

#ifdef __cplusplus
extern "C" {
#endif

// snu.c
int snu_init_probability_engine_3();
int snu_init_probability_engine(int _n_flavors, int _rotation_order[][2], int _phase_order[]);
int snu_free_probability_engine();
int snu_set_oscillation_parameters(glb_params p, void *user_data);
int snu_get_oscillation_parameters(glb_params p, void *user_data);
int snu_filtered_probability_matrix_cd(double P[SNU_MAX_FLAVORS][SNU_MAX_FLAVORS],
      double E, double L, double V, double sigma, int cp_sign, void *user_data);
int snu_probability_matrix(double _P[3][3], int cp_sign, double E,
      int psteps, const double *length, const double *density,
      double filter_sigma, void *user_data);
int snu_probability_matrix_all(double P[SNU_MAX_FLAVORS][SNU_MAX_FLAVORS], int cp_sign,
      double E, int psteps, const double *length, const double *density,
      double filter_sigma, void *user_data);
int snu_probability_matrix_m_to_f(double P[SNU_MAX_FLAVORS][SNU_MAX_FLAVORS],
      int cp_sign, double E, int psteps, const double *length, const double *density,
      double filter_sigma, void *user_data);
int snu_filtered_probability_matrix_m_to_f(double P[SNU_MAX_FLAVORS][SNU_MAX_FLAVORS],
      double E, double L, double V, double sigma, int cp_sign, void *user_data);
gsl_matrix_complex *snu_get_U();
int snu_print_gsl_matrix_complex(gsl_matrix_complex *A);

int snu_probability_matrix_nusquids(double P[][2][3],
      unsigned n_E, double *E, double ini_state_nu[][3], double ini_state_nubar[][3],
      int psteps, const double *length, const double *density, const double filter_value);
int snu_probability_matrix_osc_decay(double P[][2][3],
      unsigned n_E, double *E, double ini_state_nu[][3], double ini_state_nubar[][3],
      int psteps, const double *length, const double *density, const double filter_value);

struct snu_probability_table *snu_alloc_probability_table();
int snu_free_probability_table(struct snu_probability_table *p);
int snu_compute_probability_table(int experiment, struct snu_probability_table *p,
                                  const char *output_file);
int snu_load_probability_table(const char *input_file, struct snu_probability_table *p);
int snu_tabulated_probability_matrix(double _P[3][3], int cp_sign, double E,
    int psteps, const double *length, const double *density,
    double filter_sigma, void *user_data);

// nusquids.cc
int snu_probability_matrix_nusquids_internal(double P[][2][SNU_MAX_FLAVORS],
      unsigned n_E, double *E, double ini_state_nu[][3], double ini_state_nubar[][3],
      int psteps, const double *length, const double *density,
      unsigned n_flavors, double th[SNU_MAX_FLAVORS+1][SNU_MAX_FLAVORS+1],
      double delta[SNU_MAX_PHASES], double dmsq[SNU_MAX_FLAVORS-1],
      double M_A_prime, double g_prime);

int snu_set_oscillation_parameters_osc_decay_internal(int n_flavors, glb_params params);
int snu_get_oscillation_parameters_osc_decay_internal(int n_flavors, glb_params params);
double snu_get_m4Gamma_osc_decay();
int snu_probability_matrix_osc_decay_internal(double P[][2][SNU_MAX_FLAVORS],
      unsigned n_E, double *E, double ini_state_nu[][3], double ini_state_nubar[][3],
      int psteps, const double *length, const double *density,
      unsigned n_flavors, const double filter_value);

// prem.c
int LoadPREMProfile(const char *prem_file);
double GetPREMDensity(double t, double L);
double GetAvgPREMDensity(double L_tot, double L1, double L2);
int GetPREM3LayerApprox(double L, int *n_layers, double *lengths,
                        double *densities);

#ifdef __cplusplus
}
#endif

#endif

