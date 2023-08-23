#ifndef __NU_H
#define __NU_H

// Parallelization?
//#define NU_MPI
#define NU_PSEUDO_MPI

#ifdef NU_MPI
  #include <mpi.h>
#endif

#include <math.h>
#include <globes/globes.h>
#include "glb_smear.h"
#include "const.h"
#include "snu.h"

// Select/deselect parts of the code
// (can also be done from the Makefile)
//#define USE_ATM              // Michele's atmospherics code
//#define USE_SOLAR            // Michele's solar code
//#define NU_USE_MONTECUBES    // MonteCUBES support

// Macros
#define SQR(x)      ((x)*(x))                        // x^2
#define SQR_ABS(x)  (SQR(creal(x)) + SQR(cimag(x)))  // |x|^2
#define POW10(x)    (exp(M_LN10*(x)))                // 10^x
#define MIN(X,Y)    ( ((X) < (Y)) ? (X) : (Y) )
#define MAX(X,Y)    ( ((X) > (Y)) ? (X) : (Y) )
#define SIGN(a,b)   ( (b) > 0.0 ? (fabs(a)) : (-fabs(a)) )
#define SGN(a)      ( ((a) >= 0.0) ? (1) : (-1) )
#define KRONECKER(i,j)  ( (i)==(j) ? 1 : 0 )
#define ROUND(x)    ( (int)((x) + 0.5) )


/* Debug level */
extern int debug_level;

/* MPI-related global variables and definition of parallelized for loop */
#if defined NU_MPI || defined NU_PSEUDO_MPI
  extern int mpi_rank;
  extern int mpi_size;
  #define MPIFOR(ii,min,max)                                           \
      for (int ii=min + (mpi_rank*(max+1-min))/mpi_size;               \
               ii < min + ((mpi_rank+1)*(max+1-min))/mpi_size; ii++)
#else
  #define MPIFOR(ii,min,max) for (int ii=min; ii <= max; ii++)
#endif

/* Mass hierarchies */
#define HIERARCHY_NORMAL     1
#define HIERARCHY_INVERTED  -1

/* Definition of command line or environment variable argument */
typedef struct
{
  char *name;
  int id;
} env_param;

/* Additional dynamic parameters for wide band beam */
typedef struct
{
  int flags;
  double eff_1st_max_nu;
  double eff_2nd_max_nu;
  double eff_1st_max_nubar;
  double eff_2nd_max_nubar;
  double E_1st_min;
} wbb_params_type;

/* Parameters passed to the prior function */
typedef struct
{
  int ext_flags;                  /* Which external codes are used? */
  int n_scan_params;              /* Number of variable parameters in scan or MCMC */
  int scan_params[32];            /* IDs of these parameters */
  unsigned long scan_p_flags[32]; /* Flags for each parameter */
} prior_params;

/* Neutrino flavors */
#define NU_E    0
#define NU_MU   1
#define NU_TAU  2
#define NU_S1   3
#define NU_S2   4
#define NU_S3   5
#define NU_S    NU_S1

/* Different plot types */
enum { NU_ACTION_SPECTRUM,
       NU_ACTION_PARAM_SCAN,
       NU_ACTION_MCMC,
       NU_ACTION_EXPOSURE_SCAN,
       NU_ACTION_PROB_TABLE
       // NU_ACTION_CHECK_BF
     };

/* External analysis routines */
enum { EXT_MB               = 0x000001,
       EXT_MB_300           = 0x000002,
       EXT_MBANTI           = 0x000004,
       EXT_MBANTI_200       = 0x000008,
       EXT_KARMEN           = 0x000010,
       EXT_LSND             = 0x000020,
       EXT_REACTORS         = 0x000040,
       EXT_NOMAD            = 0x000080,
       EXT_CDHS             = 0x000100,
       EXT_ATM_TABLE        = 0x000200,
       EXT_ATM_COMP         = 0x000400,
       EXT_DEEPCORE         = 0x000800,
       EXT_SOLAR            = 0x001000,
       EXT_MINOS2016        = 0x002000,
       EXT_MINOS2017        = 0x004000,
       EXT_LSND_IVAN        = 0x008000,
       EXT_KARMEN_IVAN      = 0x010000,
       EXT_DECAY_KINEMATICS = 0x020000,
       EXT_FREE_STREAMING   = 0x040000,
       EXT_MB_JK            = 0x080000,
       EXT_MUBOONE          = 0x100000
};

/* Experiment and rule numbers */
extern int EXP_BEAM_NEAR;
extern int EXP_BEAM_FAR;
extern int EXP_REACTOR_NEAR;
extern int EXP_REACTOR_FAR;


#define KAMLAND_N_REACT      16  // Number of reactors in KamLAND experiments

#define RULE_T2K_NUMU         0
#define RULE_T2K_NUE          1
#define RULE_T2K_NUMU_NOSC    1

#define RULE_NOVA_NUE         0
#define RULE_NOVA_NUMU        1
#define RULE_NOVA_NUE_BAR     2
#define RULE_NOVA_NUMU_BAR    3

#define RULE_WBB_WC_NUE       0
#define RULE_WBB_WC_NUMU      1

#define RULE_WBB_LAR_E_CC     0
#define RULE_WBB_LAR_MU_CC    1
#define RULE_WBB_LAR_EBAR_CC  2
#define RULE_WBB_LAR_MUBAR_CC 3
#define RULE_WBB_LAR_E_QE     4
#define RULE_WBB_LAR_MU_QE    5
#define RULE_WBB_LAR_EBAR_QE  6
#define RULE_WBB_LAR_MUBAR_QE 7

/* MINOS 2017 flags */
#define MINOS_2017_PRINT_RATES  0x01  /* Output event rates */

/* Options for degfinder */
#define DEG_NO_NH      0x01   /* Omit normal hierarchy fits                    */
#define DEG_NO_IH      0x02   /* Omit inverted hierarchy fits                  */
#define DEG_NO_SYS     0x04   /* Switch off systematics                        */
#define DEG_NO_CORR    0x10   /* Switch off correlations                       */
#define DEG_NO_DEG     0x20   /* Switch off all degeneracies                   */
#define DEG_ONLY_STD_DEG 0x40   /* Switch off degeneracies in all but the std. osc. params */
#define DEG_MCMC       0x80   /* Instead of ordinary minimization, run Markov Chain MC */

#define DEG_LOGSCALE   0x01   /* Use logarithmic scale for a parameter         */
#define DEG_PLUS_MINUS 0x02   /* Consider positive and negative values for a   */
                              /* param (only in combination with DEG_LOGSCALE) */
#define DEG_S22        0x04   /* Scan over sin^2 2*param rather than param     */

/* Flags for WBB analysis */ 
#define WBB_NO_1ST_MAX          0x01 
#define WBB_NO_2ND_MAX          0x02 
#define WBB_1ST_MAX_TOTAL_RATES 0x04 
#define WBB_2ND_MAX_TOTAL_RATES 0x08

/* Flags identifying NC and CC events, respectively, in MINOS */
#define MINOS_NC            1
#define MINOS_CC            2


/* Starting values for systematics minimization */
#define MAX_SYS             200
extern double sys_startval_beam_plus[MAX_SYS];
extern double sys_startval_beam_minus[MAX_SYS];
extern double sys_startval_reactor[MAX_SYS];


/* Function declarations */
/* --------------------- */

#ifdef __cplusplus
extern "C" {
#endif

/* util.c */
int LoadNd(const char *filename, double **buffer, const int n_columns, int *n_rows);
int LoadNdAlloc(const char *filename, double **buffer, const int n_columns, int *n_rows);

/* degfinder.c */
double ChiNPWrapper(glb_params base_values, int hierarchy, glb_params fit_values);
int degfinder(const glb_params base_values, const int n_prescan_params,
      const int *prescan_params, const double *prescan_min,
      const double *prescan_max, const int *prescan_steps,
      const glb_projection prescan_proj, const glb_projection fit_proj,
      int *n_deg, glb_params *deg_pos, double *deg_chi2, const unsigned long flags,
      const unsigned long *prescan_flags, const char *output_file);

/* sys.c */
double chiNOvA(int exp, int rule, int n_params, double *x, double *errors,
              void *user_data);
double chiWBB_WC(int exp, int rule, int n_params, double *x, double *errors,
              void *user_data);
double chiWBB_WCfast(int exp, int rule, int n_params, double *x, double *errors,
              void *user_data);
double chiWBB_LAr(int exp, int rule, int n_params, double *x, double *errors,
              void *user_data);
double chiDCNorm(int exp, int rule, int n_params, double *x, double *errors,
              void *user_data);
double chiKamLAND(int exp, int rule, int n_params, double *x, double *errors,
              void *user_data);
double chiLSNDspectrum(int exp, int rule, int n_params, double *x, double *errors,
              void *user_data);
double chiT2K(int exp, int rule, int n_params, double *x, double *errors,
              void *user_data);
double chiT2K_FDonly(int exp, int rule, int n_params, double *x, double *errors,
              void *user_data);


/* minos.c */
int minos_smear(glb_smear *s, double **matrix, int *lower, int *upper);
double chiMINOS_2016(int exp, int rule, int n_params, double *x, double *errors,
              void *user_data);
double chiMINOS_2011(int exp, int rule, int n_params, double *x, double *errors,
              void *user_data);
double chiMINOS_2010(int exp, int rule, int n_params, double *x, double *errors,
              void *user_data);
int MINOS_2016_init();
double MINOS_2016_prior(glb_params params);
int MINOS_2017_init();
double MINOS_2017_prior(glb_params params, unsigned flags);

/* mb-2018.cc */
int chiMB_init(int threshold);
int chiMB_clear();
double chiMB(int exper, int rule, int n_params, double *x, double *errors,
              void *user_data);
int getMBspectrum(const char *fname);

/* mb-jk-2018.cc */
int chiMB_jk_init(const char *bg_tune);
int chiMB_jk_clear();
double chiMB_jk(int print_spectrum);

/* muboone.cc */
int chiMuBooNE_init(const char *bg_tune, int __use_feldman_cousins);
int chiMuBooNE_clear();
double chiMuBooNE(int print_spectrum);

/* e776.c */
double chi_E776(int exp, int rule, int n_params, double *x,
              double *errors, void *user_data);
double chi_E776_rates(int exp, int rule, int n_params, double *x,
              double *errors, void *user_data);

/* c12.c */
double chi_lsnd_c12(int exp, int rule, int n_params,
              double *x, double *errors, void *user_data);
double chi_karmen_c12_JR(int exp, int rule, int n_params,
              double *x, double *errors, void *user_data);
double chi_karmen_c12(int exp, int rule, int n_params,
              double *x, double *errors, void *user_data);
double chi_nue_carbon(int exp, int rule, int n_params,
              double *x, double *errors, void *user_data);
double chi_nue_carbon_spectrum(int exp, int rule, int n_params,
              double *x, double *errors, void *user_data);
void init_nue_carbon(int KM_spectrum);

/* icarus-2012.c */
int init_icarus_2012();
double chi_ICARUS_2012(int exp, int rule, int n_params, double *x, double *errors,
                       void *user_data);

/* icarus-2014.c */
int init_icarus_2014();
double chi_ICARUS_2014(int exp, int rule, int n_params, double *x, double *errors,
                       void *user_data);

/* opera-2013.c */
int init_OPERA();
double chi_OPERA(int exp, int rule, int n_params, double *x, double *errors,
                 void *user_data);

/* sensitivities.c */
double sample(double min, double max, int steps, int i);
typedef int (*sens_func)(const char *, double, double, int, double, double, int);
int RestrictWBBEnergyWindow(wbb_params_type *wbb_params);
int print_rates(const long ext_flags);
int my_print_params(glb_params p);
int param_scan(const char *key_string, int n_p, char *params[], double p_min[], double p_max[],
       int p_steps[], unsigned long p_flags[], int n_min_params, char *min_params[],
       int prescan_n_p, char *prescan_params[], double prescan_p_min[],
       double prescan_p_max[], int prescan_p_steps[], unsigned long prescan_p_flags[]);
//int mcmc(const char *key_string, int n_p, char *params[],
//       double p_min[], double p_max[], unsigned long p_flags[]);
int mcmc_deg(const char *output_file, int n_p, char *params[],
       double p_min[], double p_max[], unsigned long p_flags[],
       int prescan_n_p, char *prescan_params[], double prescan_p_min[],
       double prescan_p_max[], int prescan_p_steps[], unsigned long prescan_p_flags[]);

/* prem.c */
int LoadPREMProfile(const char *prem_file);
double GetPREMDensity(double t, double L);
double GetAvgPREMDensity(double L_tot, double L1, double L2);
int GetPREM3LayerApprox(double L, int *n_layers, double *lengths,
                        double *densities);

/* prior.cc */
int ext_init(int ext_flags, int use_feldman_cousins);
double my_prior(const glb_params in, void* user_data);

/* iface.c */
int checkBF(int n_flavors);

/* nu.cc */
int load_exps(const int n_exps, char **exps);


/* Inline functions */
/* ---------------- */

/* Square of real number */
static inline double square(double x)
{
  return x*x;
} 

/* Gauss likelihood (this is sufficient for reactor experiments due to the large event
   numbers; for other setups, one should use Poisson statistics) */
static inline double gauss_likelihood(double true_rate, double fit_rate, double sqr_sigma)
{
  if (sqr_sigma > 0)
    return square(true_rate - fit_rate) / sqr_sigma;
  else
    return 0.0;
}

/* Poisson likelihood */
static inline double poisson_likelihood(double true_rate, double fit_rate)
{
  double res;
  res = fit_rate - true_rate;
  if (true_rate > 0)
  {
    if (fit_rate <= 0.0)
      res = 1e100;
    else
      res += true_rate * log(true_rate/fit_rate);
  }
  else
    res = fabs(res);

  return 2.0 * res;
}

#ifdef __cplusplus
}
#endif

#endif

