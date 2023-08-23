/***************************************************************************
 * Functions for MicroBooNE fit in the 1e1p CCQE channel                   *
 * based heavily on my MiniBooNE fit                                       *
 * and of course on https://arxiv.org/abs/2110.14080                       *
 ***************************************************************************
 * Author: Joachim Kopp, CERN (jkopp@cern.ch)                              *
 * Date:   December 2021                                                   *
 ***************************************************************************/
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_interp.h>
#include <gsl/gsl_roots.h>
#include <gsl/gsl_multimin.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_sf_gamma.h>
#include <globes/globes.h>
#include "nu.h"

// Flags that affect chi^2 calculation
//   OSC_NORM     : Take into account impact of \nu_\mu disapp. on normalization
//   OSC_BG       : Include oscillations of \nu_e backgrounds
//   OSC_NUMU     : Allow also the \nu_\mu sample to oscillate
//   OSC_NO_NUBAR : Omit anti-neutrino data from the fit
//   OSC_NO_NUMU  : Omit muon neutrinos and anti-neutrinos from the fit
//                  (I'm not sure this is consistent - use for debugging only)
// TAG_DEFS - This signals ./compile-and-run where to insert new #define's
//#define OSC_NORM     // TAG_DEF - tells ./compile-and-run which #define's to replace
//#define OSC_BG       // TAG_DEF
//#define OSC_NUMU     // TAG_DEF
#define OSC_NO_NUBAR // TAG_DEF // FIXME
#define OSC_NO_NUMU  // TAG_DEF // FIXME
//#define MUBOONE_SENSITIVITY  // TAG_DEF - set data=BG prediction to compute sensitivity
                             //           rather than doing a real fit

#define MUBOONE_CHI2_POISSON // use Poissonian chi^2 + nuisance parameters
                             // instead of assuming Gaussian chi^2
//#define MUBOONE_CHI2_CNP     // use the combined Neyman-Pearson chi^2 that
//                             // MicroBooNE have been using

#if (defined MUBOONE_CHI2_POISSON) && (defined MUBOONE_CHI2_CNP)
  #error "ERROR: Both MUBOONE_CHI2_POISSON and MUBOONE_CHI2_CNP defined in muboone.cc"
#endif

#define MUBOONE_DATA_DIR          "glb/muboone/"
#define MUBOONE_COV_MATRIX_FILE   "glb/muboone/1e1pCCQE-cov-matrix-total.h"
#define MB_MIGRATION_MATRIX_FILE  "glb/muboone/mb-migration-matrix-11bins.h"
#define MUB_MIGRATION_MATRIX_FILE "glb/muboone/mub-migration-matrix-ccqe.h"

static const double MB_baseline = 0.520;   // [km] - MiniBooNE baseline
static const double SB_baseline = 0.100;   // [km] - SciBooNE baseline

static const double MB_exposure  = 12.84e20; // MiniBooNE exposure
static const double muB_exposure =  6.67e20; // muBooNE exposure (CCQE)

static const double MB_mass      = 818.;     // [tons] MiniBooNE target mass
static const double muB_mass     =  85.;     // [tons] muBooNE target mass

// MiniBooNE low-pass filter
//static double muboone_lowpass_width = 5.0;   // [MeV]

// Energy binning
static const int E_true_bins   =  680;       // binning in E_true
static const double E_true_min =  120.0;     // [MeV]
static const double E_true_max = 3520.0;     // [MeV]

#define NE (E_reco_bins_e)
static const int E_reco_bins_e = 10;         // fig. 16 of 2110.14080
static const double E_reco_bin_edges_e[] = {
                             200.0, 300.0,  400.0,  500.0,  600.0,  700.0,
                             800.0, 900.0, 1000.0, 1100.0, 1200.0 }; // [MeV]
static const double E_reco_min_e = E_reco_bin_edges_e[0];
static const double E_reco_max_e = E_reco_bin_edges_e[E_reco_bins_e];

#define NMU (E_reco_bins_mu)
static const int E_reco_bins_mu = 19;        // fig. 8 of 2110.14080
static const double E_reco_bin_edges_mu[] = { 250., 300., 350., 400., 450., 500., 550.,
                                              600., 650., 700., 750., 800., 850., 900.,
                                              950., 1000., 1050., 1100., 1150., 1200. };
static const double E_reco_min_mu = E_reco_bin_edges_mu[0];
static const double E_reco_max_mu = E_reco_bin_edges_mu[E_reco_bins_mu];

// data and predicted backgrounds
static double data_e[]      = { 4, 1, 1, 2, 5, 3, 8, 0, 1, 0 }; // HepData
static double data_mu[]     = {  25, 192, 277, 401, 389, 463, 438, 481, 395, 353,
                                      303, 232, 240, 177, 118, 109, 109,  85,  57 };

static const double pred_mu[] =                   // total \nu_mu prediction
  { 13.18681,  74.72527, 149.45055, 220.87912, 247.25275, 320.87912, 349.45055,
   316.48352, 298.90110, 269.23077, 242.85714, 204.39560, 168.13187, 139.56044,
   105.49451,  82.41758,  60.43956,  58.24176,  51.64835 };
static const double pred_mu_bg[] =                // total BG in the \nu_\mu channel
  { 16.4835, 105.495, 114.286, 134.066,  84.6154, 76.9231, 81.3187, 81.3187,
    61.5385,  57.1429, 53.8462, 46.1539, 46.1538, 50.5495, 34.0659, 26.3736,
    21.978,   12.0879, 10.989 };

static const double bg_e_pre_constraint[] =       // pre-constraint BG, HepData
  { 1.7210158742468715, 3.2197328392334565, 2.916224217932062, 2.8238164171104554,
    3.1481503816715652, 2.9776962942387017, 2.8726438026244123, 2.577059859454922,
    2.558364855753583,  2.5773475987551016 };

static const double bg_e_CC_nue[] =               // CC \nu_e BG
  { 0.91082, 2.96015, 3.03605, 2.99810, 3.26376,
    2.88425, 2.84630, 2.58065, 2.42884, 2.04934 };
static const double bg_e_other[] =                // non-\nu_e BG (mostly \nu_\mu)
  { 0.91081, 0.68311, 0.30361, 0.11385, 0.11385,
    0.0759,  0.1518,  0.18975, 0.26566, 0.3795 };
static const double sig_e_LEE[] =                 // template for the low-E excess
  { 2.72837, 4.81974, 2.08734, 0.96505, 0.49739,
    0.23985, 0.1009,  0.0246,  0.,      0. };
static const double sig_plus_bge_LEE[] =          // template for the low-E excess + BG
  { 4.550, 8.463, 5.427, 4.077, 3.875, 3.200, 3.099, 2.795, 2.694, 2.391 };

// MiniBooNE's low-energy data and prediction
static const int mb_E_reco_bins_e = 11;
static const double mb_E_reco_bin_edges_e[] =
                                  {  200.0, 300.0,  375.0,  475.0,  550.0,  675.0,
                                     800.0, 950.0, 1100.0, 1250.0, 1500.0, 3000.0 };
static const double mb_data_e[] = { 497.0,      283.0,      313.0,      167.0,
                                    184.0,      163.0,      140.0,      115.0,
                                     97.0,       98.0,      130.0 };
static const double mb_bg_e[]   = { 361.002334, 216.002142, 239.436776, 127.517957, // total BG
                                    179.035344, 133.901816, 139.020389, 113.446978,
                                     81.204519,  98.603919, 137.953204 };
static const double mb_bg_e_cc[] = { 39.839,     47.4743,    83.291,     63.1215,   // \nu_e BG
                                    103.11,      98.5888,   105.197,     88.962,
                                     63.9345,    86.81,      70.185 };

// Binning for unfolding of MiniBooNE detector response
static const int mb_E_true_bins_e = 13;
static const double mb_E_true_bin_edges_e[] = // binning for d'Agostini's unfolding
                                 { 200, 250,  300,  350,  400,  450,  500,
                                   600, 800, 1000, 1500, 2000, 2500, 3000 };
static const double mb_true_nue[] = // MB true \nu_e spectrum, fig.3 in MICROBOONE-NOTE-1043-PUB
                                 { 0.782, 1.448, 1.973, 2.688, 3.292, 3.757, 4.502,
                                   5.314, 5.717, 5.131, 3.653, 2.419, 1.466 };
static double sm[E_reco_bins_e][mb_E_true_bins_e];  // muBooNE energy smearing matrix
//static double eff[E_reco_bins_e];                   // extra efficiency factors
#include MB_MIGRATION_MATRIX_FILE
#include MUB_MIGRATION_MATRIX_FILE

// microBooNE efficiencies
// from Ivan, based on comparing Fig. 1 in arXiv:2110.13978 with MC binned in true energy;
// multiply by the ratio in nu_e bkg between MiniBooNE and MicroBooNE. This takes into
// account the different cross-sections
static const double mub_eff[] =  { 0.0228416, 0.0446695, 0.0544701, 0.0595202, 0.0599082,
                                   0.0552498, 0.0511008, 0.0506197, 0.0385511, 0.0216909,
                                   0.00497669, 0.00113876, 0.000117409 };

// covariance matrix
#define NCOV (n_cov)
static const int n_cov = E_reco_bins_e + E_reco_bins_mu;
#include MUBOONE_COV_MATRIX_FILE
static gsl_matrix *Mcov = NULL;     // block convariance matrix
static gsl_matrix *Mcovinv = NULL;  // inverse of covariance matrix
static gsl_permutation *perm = NULL;

// Global data structures
static double R_nu[E_true_bins][mb_E_reco_bins_e]; // E_true -> E_reco mapping for MB based on MC
static double R_nu_bar[E_true_bins][mb_E_reco_bins_e];

static int use_feldman_cousins = 0;

// data structure for passing data to Feldman-Cousins callback function
struct MuBooNE_fc_data {
  glb_params test_params;
  double this_data_e[E_reco_bins_e];
  double this_data_mu[E_reco_bins_mu];
};

// data structure for passing data to chi^2 function when minimizing over
// nuisance parameters
struct MuBooNE_poisson_data {
  int n_nuis;
  double data[NE+NMU];
  double pred[NE+NMU];
  double *Mcov_inv;
};

// declarations from glb_min_sup.h
extern "C" {
  typedef double (*glb_minimize_func)(double *, int, void *);
  int glb_hybrid_minimizer(double *P, int n, int n_osc, double ftol, int *iter,
                           double *fret, glb_minimize_func f, void *user_data);
}


/***************************************************************************
 * d'Agostini's method for unfolding the MiniBooNE detector response,      *
 * followed by folding in our model of the muBooNE detector response       *
 * see microboone.fnal.gov/wp-content/uploads/MICROBOONE-NOTE-1043-PUB.pdf *
 ***************************************************************************
 * Parameters:                                                             *
 *   d: predicted MiniBooNE spectrum (signal + intrinsic \nu_e background) *
 *      must have length mb_E_reco_bins_e                                  *
 *   rates_muboone: an array of length E_reco_bins_e that will be filled   *
 *      with the predicted signal rate in microBooNE                       *
 ***************************************************************************/
int dAgostini_unfolding(const double *d, double *rates_muboone)
{
  double u[mb_E_true_bins_e];  // initial guess for the true spectrum
  for (int it=0; it < mb_E_true_bins_e; it++)
    u[it] = mb_true_nue[it] * (mb_E_true_bin_edges_e[it+1] - mb_E_true_bin_edges_e[it]);

  double epsilon[mb_E_true_bins_e]; // efficiencies
  for (int it=0; it < mb_E_true_bins_e; it++)
  {
    epsilon[it] = 0.;
    for (int ir=0; ir < mb_E_reco_bins_e; ir++)
      epsilon[it] += mb_migration_matrix[ir][it];
  }

  // iterate d'Agostini's unfolding method
  for (int n=0; n < 3; n++)
  {
    double A_dot_u[mb_E_reco_bins_e];
    for (int ir=0; ir < mb_E_reco_bins_e; ir++)
    {
      A_dot_u[ir] = 0.;
      for (int it=0; it < mb_E_true_bins_e; it++)
        A_dot_u[ir] += mb_migration_matrix[ir][it] * u[it];
    }

    for (int it=0; it < mb_E_true_bins_e; it++)
    {
      double x = 0.;
      for (int ir=0; ir < mb_E_reco_bins_e; ir++)
        x += mb_migration_matrix[ir][it] / A_dot_u[ir] * d[ir];
      u[it] = x * u[it] / epsilon[it];
    }
  }

  for (int it=0; it < mb_E_true_bins_e; it++)
  {
    // subtract predicted MiniBooNE \nu_e background
//    u[it] = (u[it] - 12.84/6.67 * mb_true_nue[it] * (mb_E_true_bin_edges_e[it+1]
//                                                   - mb_E_true_bin_edges_e[it]));
//                                                // factor 12.84/6.46 = ratio of POT
    if (u[it] < 0.)
    {
      u[it] = 0.;
      continue;
    }

    // set to 0 everything above 800 MeV (see Fig. 2 in https://arxiv.org/abs/2110.14054)
//    if (it >= 8) //FIXME?
//    {
//      u[it] = 0.;
//      continue;
//    }

    // rescale to MicroBooNE expsure
    u[it] *= muB_exposure / MB_exposure;

    // account for detector size
    u[it] *= muB_mass / MB_mass;

    // multiply by muBooNE efficiency from https://arxiv.org/abs/2110.14054
    u[it] *= mub_eff[it];
  }

//    for (int it=0; it < mb_E_true_bins_e; it++)//FIXME
//      printf("%g\n", u[it]);//FIXME
//    getchar();//FIXME

  // Fold with smearing matrix
  for (int ir=0; ir < E_reco_bins_e; ir++)
  {
    rates_muboone[ir] = 0.;
    for (int it=0; it < mb_E_true_bins_e; it++)
      rates_muboone[ir] += mub_migration_matrix[ir][it] * u[it];
//      rates_muboone[ir] += sm[ir][it] * u[it];
  }

  return 0;
}


/***************************************************************************
 * Initialize data structures required for MiniBooNE chi^2 calculation *
 ***************************************************************************/
int chiMuBooNE_init(const char *bg_tune, int __use_feldman_cousins)
{
  use_feldman_cousins = __use_feldman_cousins;

  printf("# Flags in MicroBooNE code: ");
  #ifdef OSC_BG
    printf("OSC_BG ");
  #endif
  #ifdef OSC_NORM
    printf("OSC_NORM ");
  #endif
  #ifdef OSC_NUMU
    printf("OSC_NUMU ");
  #endif
  #ifdef OSC_NO_NUBAR
    printf("OSC_NO_NUBAR ");
  #endif
  #ifdef OSC_NO_NUMU
    printf("OSC_NO_NUMU ");
  #endif
  #ifdef MUBOONE_SENSITIVITY
    printf("MUBOONE_SENSITIVITY ");
  #endif
  if (bg_tune && strlen(bg_tune) > 0)
    printf("bg_tune=%s", bg_tune);
  printf("\n");

  FILE *f;

  // Sort MC events into MiniBooNE detector response matrices
  memset(R_nu, 0, E_true_bins*mb_E_reco_bins_e*sizeof(R_nu[0][0]));
  f = fopen(MUBOONE_DATA_DIR "miniboone_numunuefullosc_ntuple.txt", "r");
  if (!f)
  {
    fprintf(stderr, "chiMuBooNE_init: MC n-tuple file for neutrino mode not found.\n");
    return -1;
  }
  int n_nu = 0;    // Event counter
  while (!feof(f))
  {
    volatile int status, i_true, i_reco;
      // Here, I ran into a GCC bug: with -O2, the code sometimes claims
      // that i_reco>=0 even though i_reco=-1 (indicating an energy outside
      // the simulated range). I assume this is because i_reco is relocated
      // into a processor register and then treated as unsigned int.
      // Including "volatile" is a workaround for this.
    double E_true, mb_E_reco, L, w;
    status = fscanf(f, "%lg %lg %lg %lg", &E_true, &mb_E_reco, &L, &w);
    if (status == EOF)
      break;
    i_true = E_true_bins * (E_true - E_true_min) / (E_true_max   - E_true_min);
    i_reco = mb_E_reco_bins_e;
    while (mb_E_reco < mb_E_reco_bin_edges_e[i_reco])
        i_reco--;
    if (i_true >= 0  &&  i_true < E_true_bins  &&
        i_reco >= 0  &&  i_reco < mb_E_reco_bins_e)
      R_nu[i_true][i_reco] += w;
    n_nu++;
  }
  if (f) fclose(f);

  memset(R_nu_bar, 0, E_true_bins*mb_E_reco_bins_e*sizeof(R_nu_bar[0][0]));
  f = fopen(MUBOONE_DATA_DIR "miniboone_nubarfullosc_ntuple.txt", "r");
  if (!f)
  {
    fprintf(stderr, "chiMuBooNE_init: MC n-tuple file for antineutrino mode not found.\n");
    return -2;
  }
  int n_nu_bar = 0;
  while (!feof(f))
  {
    volatile int status, i_true, i_reco;
    double E_true, mb_E_reco, L, w;
    status = fscanf(f, "%lg %lg %lg %lg", &E_true, &mb_E_reco, &L, &w);
    if (status == EOF)
      break;
    i_true = E_true_bins * (E_true - E_true_min) / (E_true_max   - E_true_min);
    i_reco = mb_E_reco_bins_e;
    while (mb_E_reco < mb_E_reco_bin_edges_e[i_reco])
        i_reco--;
    if (i_true >= 0  &&  i_true < E_true_bins  &&
        i_reco >= 0  &&  i_reco < mb_E_reco_bins_e)
      R_nu_bar[i_true][i_reco] += w;
    n_nu_bar++;
  }
  if (f) fclose(f);

  // Normalize detector response matrices (see MB instructions)
  for (int it=0; it < E_true_bins; it++)
    for (int ir=0; ir < mb_E_reco_bins_e; ir++)
    {
      R_nu[it][ir]     /= n_nu;
      R_nu_bar[it][ir] /= n_nu_bar;
    }

  // construct Gaussian smeaaring matrix for muBooNE
  const double sigma = 0.165; // see https://arxiv.org/abs/2110.14054
  for (int ir=0; ir < E_reco_bins_e; ir++)
    for (int it=0; it < mb_E_true_bins_e; it++)
    {
      double E_true = 0.5 * (mb_E_true_bin_edges_e[it+1] + mb_E_true_bin_edges_e[it]);
      sm[ir][it] = 0.5 * ( erf( (E_reco_bin_edges_e[ir+1] - E_true) / (sigma*E_true*M_SQRT2) )
                         - erf( (E_reco_bin_edges_e[ir]   - E_true) / (sigma*E_true*M_SQRT2) ) );
    }

  // determine extra muBooNE efficiency factors (deviations from flat efficiency)
  // by applying d'Agostini's method to the observed MiniBooNE excess and comparing
  // the result to muBooNE's eLEE template
////  double d[] = { 174.648704,   112.55166925, 152.975656, 100.406957, 103.62124375,
////                 126.55393375, 103.5530225,   88.392172,  82.187173,  85.783805, 111.19639 };
//  double d[mb_E_reco_bins_e];
//  double rates_e[E_reco_bins_e];
//  for (int ir=0; ir < mb_E_reco_bins_e; ir++)
////    d[ir] = mb_data_e[ir] - (mb_bg_e[ir] - mb_bg_e_cc[ir]);
//    d[ir] = mb_data_e[ir] - mb_bg_e[ir];
//  dAgostini_unfolding(d, rates_e);
//  for (int ir=0; ir < E_reco_bins_e; ir++)
//    printf("%10g %10g\n", sig_e_LEE[ir], rates_e[ir]); //FIXME
//  getchar();

//  for (int ir=0; ir < E_reco_bins_e; ir++)
//    if (ir <= 4)
//      eff[ir] = sig_e_LEE[ir] / rates_e[ir];
//    else
//      eff[ir] = 1.0;

  // Initialize data structures for covariance matrix
  Mcov    = gsl_matrix_alloc(NCOV, NCOV);
  Mcovinv = gsl_matrix_alloc(NCOV, NCOV);
  perm    = gsl_permutation_alloc(NCOV);


  // when calculating sensitivities rather than fits to actual data,
  // replace the data by the BG prediction
  #ifdef MUBOONE_SENSITIVITY
    #ifdef OSC_NO_NUMU
      for (int ir=0; ir < E_reco_bins_e; ir++)
        data_e[ir] = bg_e_CC_nue[ir] + bg_e_other[ir];;
    #else
      for (int ir=0; ir < E_reco_bins_e; ir++)
        data_e[ir] = bg_e_pre_constraint[ir];
    #endif

    for (int ir=0; ir < E_reco_bins_mu; ir++)
      data_mu[ir] = pred_mu[ir] + pred_mu_bg[ir];
  #endif

  return 0;
}


/***************************************************************************
 * Cleanup GSL data structures required for MicroBooNE chi^2 calculation   *
 ***************************************************************************/
int chiMuBooNE_clear()
{
  if (perm)    { gsl_permutation_free(perm); perm    = NULL; }
  if (Mcov)    { gsl_matrix_free(Mcov);      Mcov    = NULL; }
  if (Mcovinv) { gsl_matrix_free(Mcovinv);   Mcovinv = NULL; }
  return 0;
}


/***************************************************************************
 * Calculate MicroBooNE event rates for the current set of oscillation     *
 * parameters                                                              *
 ***************************************************************************
 * Parameters:                                                             *
 *   rates_XXX: arrays to be filled with the computed rates                *
 ***************************************************************************/
int MuBooNE_rates(
  double rates_e[E_reco_bins_e],          // Oscillated flux: \nu_e sig (nu mode)
  double rates_e_bar[E_reco_bins_e],      //                  \nu_e sig (anti-nu mode)
  double rates_e_bg[E_reco_bins_e],       //                  \nu_e BG (nu mode)
  double rates_e_bar_bg[E_reco_bins_e],   //                  \nu_e BG (anti-nu mode)
  double rates_mu[E_reco_bins_mu],        //                  \nu_\mu sig (nu mode)
  double rates_mu_bar[E_reco_bins_mu],    //                  \nu_\mu sig (anti-nu mode)
  double rates_mu_bg[E_reco_bins_mu],     //                  \nu_\mu BG (nu mode)
  double rates_mu_bar_bg[E_reco_bins_mu], //                  \nu_\mu BG (anti-nu mode)
  double rates_e_mb[mb_E_reco_bins_e],    //                  \nu_e sig (nu mode, MB binning)
  double rates_e_bar_mb[mb_E_reco_bins_e])//                  \nu_e sig (anti-nu mode, MB binning)    
{
  // Evolve initial spectrum though Ivan's code
  memset(rates_e,         0, E_reco_bins_e * sizeof(rates_e[0]));
  memset(rates_e_bar,     0, E_reco_bins_e * sizeof(rates_e_bar[0]));
  memset(rates_e_bg,      0, E_reco_bins_e * sizeof(rates_e_bg[0]));
  memset(rates_e_bar_bg,  0, E_reco_bins_e * sizeof(rates_e_bar_bg[0]));
  memset(rates_e_mb,      0, mb_E_reco_bins_e * sizeof(rates_e_mb[0]));
  memset(rates_e_bar_mb,  0, mb_E_reco_bins_e * sizeof(rates_e_bar_mb[0]));

  // compute rates in MiniBooNE ...
  for (int it=0; it < E_true_bins; it++)
  {
    double E_true = E_true_min + (it+0.5) * (E_true_max - E_true_min) / E_true_bins;
    double Pe[3][3];
//    double Pbare[3][3];
    double rho = 0.0;
    double filter_sigma = 0.3 * (E_true_max - E_true_min) / E_true_bins;
    snu_probability_matrix(Pe,   +1,0.001*E_true,1,&MB_baseline,&rho,0.001*filter_sigma,NULL);
//    snu_probability_matrix(Pbare,-1,0.001*E_true,1,&MB_baseline,&rho,0.001*filter_sigma,NULL);
    for (int imb=0; imb < mb_E_reco_bins_e; imb++)
    {
      #ifdef OSC_NORM
        rates_e_mb[imb]     += R_nu[it][imb] * Pe[NU_MU][NU_E];//FIXME FIXME/ Pe[NU_MU][NU_MU];
//        rates_e_bar_mb[imb] += R_nu_bar[it][imb] * Pbare[NU_MU][NU_E] / Pbare[NU_MU][NU_MU];
                  /* Pme/Pmm - see discussion with W Louis:
                   * Flux is normalized to \nu_\mu rate, i.e. if there is \nu_\mu
                   * disappearance, the flux is underestimated by 1 / Pmm */
      #else
        rates_e_mb[imb]     += R_nu[it][imb] * Pe[NU_MU][NU_E];
//        rates_e_bar_mb[imb] += R_nu_bar[it][imb] * Pbare[NU_MU][NU_E];
      #endif
    } // for (imb)
  } // for (it)

  // ... then convert them to MicroBooNE rates using d'Agostini's method,  eq. (11) in
  //   https://microboone.fnal.gov/wp-content/uploads/MICROBOONE-NOTE-1043-PUB.pdf
  double d[mb_E_reco_bins_e];  // MB signal plus intrinsic \nu_e background
  for (int ir=0; ir < mb_E_reco_bins_e; ir++)
//    d[ir] = rates_e_mb[ir] + mb_bg_e_cc[ir];
    d[ir] = rates_e_mb[ir];

  dAgostini_unfolding(d, rates_e);

//    for (int ir=0; ir < E_reco_bins_e; ir++)
//      rates_e[ir] *= eff[ir];


  // Backgrounds
  #ifdef OSC_BG
    for (int ir=0; ir < E_reco_bins_e; ir++)
    {
      const int n = 20;
      double dE = (E_reco_bin_edges_e[ir+1] - E_reco_bin_edges_e[ir]) / n;
      double P_ee=0., P_mumu=0., P_bar_ee=0., P_bar_mumu=0.;
      double P[3][3], P_bar[3][3];
      double rho = 0.0;
      double filter_sigma = 0.2 * dE;
      for (int j=0; j < n; j++)
      {
        double E = E_reco_bin_edges_e[ir] + (j+0.5) * dE;
        snu_probability_matrix(P,+1,0.001*E,1,&MB_baseline,&rho,0.001*filter_sigma,NULL);
//        snu_probability_matrix(P_bar,-1,0.001*E,1,&MB_baseline,&rho,0.001*filter_sigma,NULL);
        P_ee       += P[NU_E][NU_E]       / n;
        P_mumu     += P[NU_MU][NU_MU]     / n;
//        P_bar_ee   += P_bar[NU_E][NU_E]   / n;
//        P_bar_mumu += P_bar[NU_MU][NU_MU] / n;
      }
      #ifdef OSC_NO_NUMU
        rates_e_bg[ir]     = (bg_e_other[ir] + bg_e_CC_nue[ir]*P_ee/P_mumu);
      #else
        rates_e_bg[ir]     = (bg_e_other[ir] + bg_e_CC_nue[ir]*P_ee/P_mumu) *
                                bg_e_pre_constraint[ir] / (bg_e_other[ir] + bg_e_CC_nue[ir]);
          // use pre-constraint BG when we're implementing the constraint from \nu_\mu explicitly
      #endif
      rates_e_bar_bg[ir] = 0.;
      rates_e[ir]       += rates_e_bg[ir];
      rates_e_bar[ir]   += rates_e_bar_bg[ir];
    }
  #else
    for (int ir=0; ir < E_reco_bins_e; ir++)
    {
      #ifdef OSC_NO_NUMU
        rates_e_bg[ir]     = bg_e_other[ir] + bg_e_CC_nue[ir];
      #else
        rates_e_bg[ir]     = bg_e_other[ir] + bg_e_CC_nue[ir] *
                                bg_e_pre_constraint[ir] / (bg_e_other[ir] + bg_e_CC_nue[ir]);
      #endif
      rates_e_bar_bg[ir] = 0.;
      rates_e[ir]       += rates_e_bg[ir];
      rates_e_bar[ir]   += rates_e_bar_bg[ir];
    }
  #endif // ifdef(OSC_BG)


  // \nu_\mu control sample
  memset(rates_mu,        0, E_reco_bins_e * sizeof(rates_mu[0]));
  memset(rates_mu_bar,    0, E_reco_bins_e * sizeof(rates_mu_bar[0]));
  memset(rates_mu_bg,     0, E_reco_bins_e * sizeof(rates_mu_bg[0]));
  memset(rates_mu_bar_bg, 0, E_reco_bins_e * sizeof(rates_mu_bar_bg[0]));
  #ifdef OSC_NUMU  // Oscillate muon neutrino rates
    for (int ir=0; ir < E_reco_bins_mu; ir++)
    {
      const int n = 20;
      double P=0., Pbar=0.;
      double dE = (E_reco_bin_edges_mu[ir+1] - E_reco_bin_edges_mu[ir]) / n;
      double Pmu[3][3], Pbarmu[3][3];
      double rho = 0.0;
      double filter_sigma = 0.2 * dE;
      for (int j=0; j < n; j++)
      {
        double E = E_reco_bin_edges_mu[ir] + (j+0.5) * dE;
        snu_probability_matrix(Pmu,   +1,0.001*E,1,&MB_baseline,&rho,0.001*filter_sigma,NULL);
//        snu_probability_matrix(Pbarmu,-1,0.001*E,1,&MB_baseline,&rho,0.001*filter_sigma,NULL);
        P    += Pmu[NU_MU][NU_MU]    / n;
//        Pbar += Pbarmu[NU_MU][NU_MU] / n;
      }
      rates_mu_bg[ir]     = pred_mu_bg[ir];
      rates_mu_bar_bg[ir] = 0.;
      rates_mu[ir]        = pred_mu[ir] * P + rates_mu_bg[ir];
      rates_mu_bar[ir]    = 0.;
    }
  #else
    for (int ir=0; ir < E_reco_bins_mu; ir++)
    {
      rates_mu_bg[ir]     = pred_mu_bg[ir];
      rates_mu_bar_bg[ir] = 0.;
      rates_mu[ir]        = pred_mu[ir] + rates_mu_bg[ir];
      rates_mu_bar[ir]    = 0.;
    }
  #endif // ifdef OSC_NUMU

  return 0;
}


/***************************************************************************
 * Callback function for minimization over nuisance parameters             *
 ***************************************************************************/
double chiMuBooNE_poisson_callback(double *x, int new_rates_flag, void *params)
{
  struct MuBooNE_poisson_data *d = (struct MuBooNE_poisson_data *) params;
  int n = d->n_nuis;
  double chi2 = 0.;
  double (*_Mcovinv)[NCOV] = (double (*)[NCOV]) d->Mcov_inv;

  // statistical part
  for (int i=0; i < n; i++)
    chi2 += poisson_likelihood(d->data[i], MAX(0., (1. + x[i]) * d->pred[i]));

  // correlations among nuisance parameter
  for (int i=0; i < n; i++)
    for (int j=0; j < n; j++)
      chi2 += x[i]*d->pred[i] * _Mcovinv[i][j] * x[j]*d->pred[j];

  return chi2;
}


/***************************************************************************
 * Calculate chi^2 for the MicroBooNE analysis, for a given data vector    *
 * see https://www-boone.fnal.gov/for_physicists/data_release/nuebar2010/  *
 * for the instructions which we loosely follow                            *
 * This function is called by the wrapper chiMuBooNE()                     *
 ***************************************************************************
 * Parameters:                                                             *
 *   data_e:         an array holding the \nu_e-like data to analyze       *
 *   data_mu:        an array holding the \nu_\mu-like data to analyze     *
 *   print_spectrum: 0: no extra output                                    *
 *                   1: output signal and total BG rates                   *
 ***************************************************************************/
double chiMuBooNE_internal(double data_e[E_reco_bins_e],
                           double data_mu[E_reco_bins_mu], int print_spectrum)
{
  double rates_e[E_reco_bins_e];          // Oscillated flux: \nu_e sig (nu mode)
  double rates_e_bar[E_reco_bins_e];      //                  \nu_e sig (anti-nu mode)
  double rates_e_bg[E_reco_bins_e];       //                  \nu_e BG (nu mode)
  double rates_e_bar_bg[E_reco_bins_e];   //                  \nu_e BG (anti-nu mode)
  double rates_mu[E_reco_bins_mu];        //                  \nu_\mu sig (nu mode)
  double rates_mu_bar[E_reco_bins_mu];    //                  \nu_\mu sig (anti-nu mode)
  double rates_mu_bg[E_reco_bins_mu];     //                  \nu_\mu BG (nu mode)
  double rates_mu_bar_bg[E_reco_bins_mu]; //                  \nu_\mu BG (anti-nu mode)
  double rates_e_mb[mb_E_reco_bins_e];    //                  \nu_e sig (nu mode - MB binning)
  double rates_e_bar_mb[mb_E_reco_bins_e];//                  \nu_e sig (anti-nu mode - MB binning)

  // compute event rates
  MuBooNE_rates(rates_e, rates_e_bar, rates_e_bg, rates_e_bar_bg, rates_mu, rates_mu_bar,
                rates_mu_bg, rates_mu_bar_bg, rates_e_mb, rates_e_bar_mb);

  // Covariance matrix
  double (*_Mcov)[NCOV]    = (double (*)[NCOV]) gsl_matrix_ptr(Mcov, 0, 0);
  double (*_Mcovinv)[NCOV] = (double (*)[NCOV]) gsl_matrix_ptr(Mcovinv, 0, 0);
  double P[NCOV];        /* Vector of predicted nu_e signal, and nu_mu signal events */
  int k = 0;
  for (int ir=0; ir < NE; ir++)
    P[k++] = rates_e[ir];
  for (int ir=0; ir < NMU; ir++)
    P[k++] = rates_mu[ir];

  for (int i=0; i < NCOV; i++)
    for (int j=0; j < NCOV; j++)
      _Mcov[i][j] = Mfrac[i][j] * P[i] * P[j];
  #ifdef MUBOONE_CHI2_POISSON
  // FIXME Cov matrix should include nuisance parameters?
  #elif defined MUBOONE_CHI2_CNP
    for (int ir=0; ir < NE; ir++)    // add statistical uncertainty
      if (data_e[ir] > 0.)           // combined Neyman/Pearson chi^2, 1903.07185
        _Mcov[ir][ir] += 3. / (1./data_e[ir] + 2./rates_e[ir]);
      else
        _Mcov[ir][ir] += 0.5 * rates_e[ir];
    for (int ir=0; ir < NMU; ir++)
      if (data_mu[ir] > 0)
        _Mcov[NE+ir][NE+ir] += 3. / (1./data_mu[ir] + 2./rates_mu[ir]);
      else
        _Mcov[NE+ir][NE+ir] += 0.5 * rates_mu[ir];
  #else
    for (int ir=0; ir < NE; ir++)    // add statistical uncertainty
      _Mcov[ir][ir] += rates_e[ir];  // standard Pearson chi^2
    for (int ir=0; ir < NMU; ir++)
      _Mcov[NE+ir][NE+ir] += rates_mu[ir];
  #endif

  #ifdef OSC_NO_NUMU
    for (int i=0; i < NCOV; i++)
    {
      for (int j=NE; j < NE+NMU; j++)
        _Mcov[i][j] = _Mcov[j][i] = 0.;
    }
    for (int i=NE; i < NE+NMU; i++)
      _Mcov[i][i] = 1.;
  #endif

  // Invert covariance matrix and compute log-likelihood
  int signum;
  gsl_linalg_LU_decomp(Mcov, perm, &signum);
  gsl_linalg_LU_invert(Mcov, perm, Mcovinv);

  for (int i=0; i < NE; i++)
    P[i]    = data_e[i] - rates_e[i];
  for (int i=0; i < NMU; i++)
    P[i+NE] = data_mu[i] - rates_mu[i];

  #ifdef OSC_NO_NUMU
    for (int i=NE; i < NE+NMU; i++)
      P[i] = 0.;
  #endif

  #ifdef MUBOONE_CHI2_POISSON
    // use one nuisance parameter per bin, correlated via the covariance
    // matrix, plus Poissonian chi^2. Minimize over nuisance parameters
    int iter;
    double chi2 = 0.0;
    struct MuBooNE_poisson_data d;
    #ifdef OSC_NO_NUMU
      d.n_nuis = NE;
    #else
      d.n_nuis = NE+NMU;
    #endif
    d.Mcov_inv = &_Mcovinv[0][0];
    double x[d.n_nuis];
    for (int i=0; i < NE; i++)
    {
      d.data[i] = data_e[i];
      d.pred[i] = rates_e[i];
      x[i]      = 0.1;
    }
    for (int i=0; i < NMU; i++)
    {
      d.data[i+NE] = data_mu[i];
      d.pred[i+NE] = rates_mu[i];
      x[i]         = 0.1;
    }
    glb_hybrid_minimizer(x, d.n_nuis, 0, 1e-5, &iter, &chi2,
                         &chiMuBooNE_poisson_callback, &d);
  #else
    double chi2 = 0.0;
    for (int i=0; i < NCOV; i++)
      for (int j=0; j < NCOV; j++)
        chi2 += P[i] * _Mcovinv[i][j] * P[j];
    #ifdef MUBOONE_CHI2_CNP
    #else
      chi2 += gsl_linalg_LU_lndet(Mcov);
    #endif
  #endif


  // Output event spectrum if requested
  // (format is [signal, bg, 0, 0] for compatibility with Pedro's code)
  if (print_spectrum)
  {
    // print covariance matrix
    if (print_spectrum >= 3)
    {   
      printf("# P ");
      for (int i=0; i < NCOV; i++)
        printf("%10.5g ", P[i]);
      printf("\n");
      for (int i=0; i < NCOV; i++)
      {
        printf("# McovInv ");
        for (int j=0; j < NCOV; j++)
          printf("%10.5g ", _Mcovinv[i][j]);
        printf("\n");
      }
    }

    if (print_spectrum >= 2)
    {   
      // MicroBooNE
      for (int ir=0; ir < NE; ir++)
        printf("# muBSPECT     %10.7g %10.7g\n",
               rates_e[ir], rates_e_bg[ir]);
      for (int ir=0; ir < NMU; ir++)
        printf("# muBSPECT MU  %10.7g %10.7g %10.7g\n",
               rates_mu[ir], pred_mu[ir]+pred_mu_bg[ir], rates_mu_bg[ir]);
    }

    glb_params p = glbAllocParams();
    snu_get_oscillation_parameters(p, NULL);
    printf("# muBSPECT PARAMS ");
    my_print_params(p);

    // Output for Kevin
    printf("# MB %8.3g %8.3g %8.3g ", glbGetOscParamByName(p, "DM41"),
           glbGetOscParamByName(p, "s22thmue"), glbGetOscParamByName(p, "Um4"));
    for (int ir=0; ir < mb_E_reco_bins_e; ir++)
      printf("%8.3g ", rates_e_mb[ir]);
    printf("\n");

    printf("# CHI2   %10.7g\n", chi2);
    glbFreeParams(p);
  }

  return chi2;
}


/***************************************************************************
 * Callback function for the Feldman-Cousins solver in MicroBooNE.         *
 * Computes chi^2 for given data and for a given test value of s22thmue    *
 ***************************************************************************/
double chiMuBooNE_fc_callback(double logs22thmue_test, void *params)
{
  struct MuBooNE_fc_data *d;
  d = (struct MuBooNE_fc_data *) params;
  glbSetOscParamByName(d->test_params, POW10(logs22thmue_test), "s22thmue");
  glbSetOscillationParameters(d->test_params);

  return chiMuBooNE_internal(d->this_data_e, d->this_data_mu, 0);
}


/***************************************************************************
 * Callback function for turning a p-value into a chi^2                    *
 * Computes 1 - CDF[\chi^2, 1 dof](x) - p, where p is passed in params     *
 ***************************************************************************/
double chiMuBooNE_chi2_pvalue_callback(double chi2, void *params)
{
  double p = * ((double *) params);
  return 1. - gsl_sf_gamma_inc_P(0.5, 0.5*chi2) - p;
}


/***************************************************************************
 * Do a Feldman-Cousins test for MicroBooNE. For a fixed value of dm41,    *
 * taken from glbGetOscillationParameters(), scan over s22thmue to         *
 * accurately determine the p-value of the data                            *
 ***************************************************************************
 * Parameters:                                                             *
 *   print_spectrum: 0: no extra output                                    *
 *                   1: output signal and total BG rates                   *
 ***************************************************************************/
double chiMuBooNE_fc(int verbosity=0)
{
  const int n_pseudo_exp = 500;
  const double logs22thmue_min = -4.;
  const double logs22thmue_max = -0.5;
  const int logs22thmue_steps  = 35;
  double logs22thmue_test;       // the log10(sin^2 2\theta_{\mu e}) at which we want the chi^2
  static double logs22thmue_bfp; // the best-fit log10(sin^2 2\theta_{\mu e}) from the real data
  static glb_params last_params = NULL;
  static double logs22thmue_bin_centers[logs22thmue_steps];
  static double logs22thmue_bin_edges[logs22thmue_steps+1];
  static double pseudo_exp_data[logs22thmue_steps][logs22thmue_steps];
  int gsl_status;
  double chi2;

  glb_params orig_params = glbAllocParams();
  glb_params true_params = glbAllocParams();
  glb_params test_params = glbAllocParams();
  if (!true_params || !test_params)
    return NAN;
  glbGetOscillationParameters(orig_params);
  logs22thmue_test = log10(glbGetOscParamByName(orig_params, "s22thmue"));

  // have the parameters (other than s22thmue) changed since the last call?
  // That is, do we have to re-run pseudo-experiments?
  bool new_params = true;
  if (!last_params)
  {
    if (!(last_params = glbAllocParams()))
      return NAN;
  }
  else
  {
    int dm41_index     = glbFindParamByName("DM41");
    int Um4_index      = glbFindParamByName("Um4");
    if (glbGetOscParams(orig_params,dm41_index)==glbGetOscParams(last_params,dm41_index)
     && glbGetOscParams(orig_params,Um4_index) ==glbGetOscParams(last_params,Um4_index))
      new_params = false;
  }

  if (new_params)
  {
    glbCopyParams(orig_params, last_params);
    glbSetOscParamByName(orig_params, 0., "Ue3"); // set redundant parameters to
    glbSetOscParamByName(orig_params, 0., "Ue4"); //   zero to avoid warning messages
    glbSetOscParamByName(orig_params, 0., "Um4");
    glbSetOscParamByName(orig_params, 0., "s22thmue");
    if (verbosity > 0)
      printf("# chiMuBooNE_fc: running pseudo-experiments at new \\Delta m_{41}^2=%g\n", 
             glbGetOscParamByName(orig_params, "DM41"));

    memset(pseudo_exp_data, 0, SQR(logs22thmue_steps) * sizeof(pseudo_exp_data[0][0]));
    for (int i=0; i < logs22thmue_steps; i++)
      logs22thmue_bin_centers[i] = logs22thmue_min + (i+0.5) * (logs22thmue_max - logs22thmue_min)
                                                             /  logs22thmue_steps;
    for (int i=0; i < logs22thmue_steps+1; i++)
      logs22thmue_bin_edges[i] = logs22thmue_min + i * (logs22thmue_max - logs22thmue_min)
                                                      /  logs22thmue_steps;

    gsl_rng *rng = gsl_rng_alloc(gsl_rng_taus2);
    gsl_min_fminimizer *m = gsl_min_fminimizer_alloc(gsl_min_fminimizer_brent);
    if (!rng || !m)
      return NAN;


    // Loop over "true" values of \sin^2 2\theta_{\mu e}
    // -------------------------------------------------
    for (int i=0; i < logs22thmue_steps; i++)
    {
      double s22thmue_true = POW10(logs22thmue_bin_centers[i]);
      glbCopyParams(orig_params, true_params);
      glbSetOscParamByName(true_params, s22thmue_true, "s22thmue");
      glbSetOscillationParameters(true_params);

      // compute "true" event rates
      double rates_e[E_reco_bins_e], rates_e_bar[E_reco_bins_e];
      double rates_e_bg[E_reco_bins_e], rates_e_bar_bg[E_reco_bins_e];
      double rates_mu[E_reco_bins_mu], rates_mu_bar[E_reco_bins_mu];
      double rates_mu_bg[E_reco_bins_mu], rates_mu_bar_bg[E_reco_bins_mu];
      double rates_e_mb[mb_E_reco_bins_e], rates_e_bar_mb[mb_E_reco_bins_e];
      MuBooNE_rates(rates_e, rates_e_bar, rates_e_bg, rates_e_bar_bg,
                    rates_mu, rates_mu_bar, rates_mu_bg, rates_mu_bar_bg,
                    rates_e_mb, rates_e_bar_mb);

      // Loop over pseudo-experiments
      for (int n=0; n < n_pseudo_exp; n++)
      {
        if (verbosity == 1)
          printf("\rs22thmue step %d/%d, pseudo-exp %d/%d",
                 i, logs22thmue_steps, n, n_pseudo_exp);
        struct MuBooNE_fc_data d;

        glbCopyParams(orig_params, test_params);
        d.test_params = test_params;

        // apply Poisson noise
        for (int ir=0; ir < E_reco_bins_e; ir++)
          d.this_data_e[ir] = gsl_ran_poisson(rng, rates_e[ir]);
        for (int ir=0; ir < E_reco_bins_mu; ir++)
          d.this_data_mu[ir] = gsl_ran_poisson(rng, rates_mu[ir]);

        // perform fit
        gsl_function F = { &chiMuBooNE_fc_callback, &d };
        gsl_status = gsl_min_fminimizer_set(m, &F, log10(s22thmue_true),
                                            logs22thmue_min, logs22thmue_max);
        int iter = 0;
        do
        {
          iter++;
          gsl_status = gsl_min_fminimizer_iterate(m);
          if (gsl_status != GSL_SUCCESS)
          {
            fprintf(stderr, "chiMuBooNE_fc: Minimization failed at "
                            "log(s22thmue_true) = %g, pseudo-exp #%d\n.",
                            log10(s22thmue_true), n);
            return NAN;
          }
        } while (gsl_min_fminimizer_x_upper(m) - gsl_min_fminimizer_x_lower(m) > 0.001
              && iter < 1000);
        int j = logs22thmue_steps * (gsl_min_fminimizer_x_minimum(m) - logs22thmue_min)
                                  / (logs22thmue_max - logs22thmue_min);
        pseudo_exp_data[i][MIN(MAX(0,j),logs22thmue_steps-1)]++;
        if (verbosity > 1)
        {
          printf("s22thmue step %d/%d, pseudo-exp %d/%d -- true = %10g, fit = %10g\n",
                 i, logs22thmue_steps, n, n_pseudo_exp,
                 log10(s22thmue_true), gsl_min_fminimizer_x_minimum(m));
        }

      } // end (loop over pseudo-exps)
    } // end (loop over s22thmue)

    if (verbosity > 1)
    {
      printf("\n");
      printf("Table of pseudo-experiment outcomes (original):\n");
      for (int i=0; i < logs22thmue_steps; i++)
      {
        for (int j=0; j < logs22thmue_steps; j++)
          printf("%4d ", (int) pseudo_exp_data[i][j]);
        printf("\n");
      }
    }

    // compute cumulative distribution of pseudo-experiments
    // (will be needed for determining p-values)
    for (int j=0; j < logs22thmue_steps; j++)   // loop over test values
    {
      for (int i=logs22thmue_steps-2; i >= 0; i--) // loop over true values
        pseudo_exp_data[i][j] += pseudo_exp_data[i+1][j];
      for (int i=logs22thmue_steps-1; i >= 0; i--) // another loop to normalize
        pseudo_exp_data[i][j] /= pseudo_exp_data[0][j];
    }

    if (verbosity > 1)
    {
      printf("\n");
      printf("Table of pseudo-experiment outcomes (cumulative):\n");
      for (int i=0; i < logs22thmue_steps; i++)
      {
        for (int j=0; j < logs22thmue_steps; j++)
          printf("%6.4f ", pseudo_exp_data[i][j]);
        printf("\n");
      }
    }


    // Now do the fit to the real data
    // -------------------------------
    struct MuBooNE_fc_data d;
    glbCopyParams(orig_params, test_params);
    d.test_params = test_params;
    for (int ir=0; ir < E_reco_bins_e; ir++)
      d.this_data_e[ir] = data_e[ir];
    for (int ir=0; ir < E_reco_bins_mu; ir++)
      d.this_data_mu[ir] = data_mu[ir];
    gsl_function F = { &chiMuBooNE_fc_callback, &d };
    gsl_status = gsl_min_fminimizer_set(m, &F, -3.0, logs22thmue_min, logs22thmue_max);
    int iter = 0;
    do
    {
      iter++;
      gsl_status = gsl_min_fminimizer_iterate(m);
      if (gsl_status != GSL_SUCCESS)
      {
        fprintf(stderr, "chiMuBooNE_fc: fit to data failed.\n");
        return NAN;
      }
    } while (gsl_min_fminimizer_x_upper(m) - gsl_min_fminimizer_x_lower(m) > 0.001
             && iter < 1000);
    logs22thmue_bfp = gsl_min_fminimizer_x_minimum(m);

    if (verbosity > 0)
      printf("chiMuBooNE_fc: best-fit s22thmue: %g\n", POW10(logs22thmue_bfp));

    if (rng)  { gsl_rng_free(rng);           rng = NULL; }
    if (m)    { gsl_min_fminimizer_free(m);  m   = NULL; }
  } // end (if new_params)


  // finally, analyze the results
  // ----------------------------

  // determine CL at which the current test value is disfavored from the
  // interpolated cumulative distribution of pseudo-experiments
  int i_test = MAX(0, MIN(logs22thmue_steps-1,
                   logs22thmue_steps * (logs22thmue_bfp - logs22thmue_min)
                                     / (logs22thmue_max - logs22thmue_min)));
  int i_true = MAX(0, MIN(logs22thmue_steps-1,
                   logs22thmue_steps * (logs22thmue_test - logs22thmue_min)
                                     / (logs22thmue_max  - logs22thmue_min)));
  double w_test = (logs22thmue_bfp                 - logs22thmue_bin_edges[i_test])
                / (logs22thmue_bin_edges[i_test+1] - logs22thmue_bin_edges[i_test]);
  double w_true = (logs22thmue_test                - logs22thmue_bin_edges[i_true])
                / (logs22thmue_bin_edges[i_true+1] - logs22thmue_bin_edges[i_true]);
  double p = (1-w_test) * ((1-w_true) * pseudo_exp_data[i_true]  [i_test]
                             + w_true * pseudo_exp_data[i_true+1][i_test]) \
               + w_test * ((1-w_true) * pseudo_exp_data[i_true]  [i_test+1]
                             + w_true * pseudo_exp_data[i_true+1][i_test+1]);
  if (verbosity > 0)
    printf("chiMuBooNE_fc: p-value = %g\n", p);

  // turn p-value into a chi^2 by solving
  // 1 - CDF[\chi^2, 1 dof][x] == p
  gsl_root_fsolver *s = gsl_root_fsolver_alloc(gsl_root_fsolver_brent);
  if (!s)
    return NAN;
  gsl_function F = { &chiMuBooNE_chi2_pvalue_callback, &p };
  gsl_root_fsolver_set(s, &F, 0., 1e6);
  int iter = 0;
  do
  {
    iter++;
    gsl_status = gsl_root_fsolver_iterate(s);
    gsl_status = gsl_root_test_interval(gsl_root_fsolver_x_lower(s),
                                        gsl_root_fsolver_x_upper(s), 0, 0.001);
  }
  while (gsl_status == GSL_CONTINUE && iter < 100);
  chi2 = gsl_root_fsolver_root(s);

  if (s)           { gsl_root_fsolver_free(s);    s           = NULL; }
  if (test_params) { glbFreeParams(test_params);  test_params = NULL; }
  if (true_params) { glbFreeParams(true_params);  true_params = NULL; }
  if (orig_params) { glbFreeParams(orig_params);  orig_params = NULL; }

  return chi2;
}


/***************************************************************************
 * Calculate chi^2 for the MicroBooNE analysis                             *
 * uses chiMuBooNE_internal to do the actual work                          *
 ***************************************************************************
 * Parameters:                                                             *
 *   print_spectrum: 0: no extra output                                    *
 *                   1: output signal and total BG rates                   *
 ***************************************************************************/
double chiMuBooNE(int print_spectrum)
{
  if (use_feldman_cousins)
  {
    return chiMuBooNE_fc(0);
  }

  else
    return chiMuBooNE_internal(data_e, data_mu, print_spectrum);
}


