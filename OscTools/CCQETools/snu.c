// ----------------------------------------------------------------------------
// Probability engine for simulating sterile neutrinos + non-standard
// interactions
// ----------------------------------------------------------------------------
// Author: Joachim Kopp (jkopp@fnal.gov)
// ----------------------------------------------------------------------------
// GLoBES -- General LOng Baseline Experiment Simulator
// (C) 2002 - 2010,  The GLoBES Team
//
// GLoBES as well as this add-on are mainly intended for academic purposes.
// Proper credit must be given if you use GLoBES or parts of it. Please
// read the section 'Credit' in the README file.
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
// ----------------------------------------------------------------------------
// Note: This file is written in C99. To compile in gcc use -std=c99 or
// -std=gnu99
// ----------------------------------------------------------------------------
// ChangeLog:
//   2017-06-12: - Implemented functions for using tabulated probabilities
//   2011-01-14: - Implemented filter feature for n_flavors > 3
//               - New function snu_probability_matrix_all returns
//                 oscillation probabilities to/from sterile flavors
//                 (the standard snu_probability_matrix returns only
//                 probabilities for oscillations among the 3 standard flavors
//                 for compatibility with GLoBES.)
// ----------------------------------------------------------------------------
// Citation information:
//
//      @Article{Kopp:2006wp,
//        author    = "Kopp, Joachim",
//        title     = "{Efficient numerical diagonalization of hermitian
//                     $3 \times 3$ matrices}",
//        journal   = "Int. J. Mod. Phys.",
//        volume    = "C19",
//        year      = "2008",
//        pages     = "523-548",
//        eprint    = "physics/0610206",
//        archivePrefix = "arXiv",
//        doi       = "10.1142/S0129183108012303",
//        SLACcitation  = "%%CITATION = PHYSICS/0610206;%%",
//        note      = "Erratum ibid.\ {\bf C19} (2008) 845",
//        memo      = "Algorithms for fast diagonalization of 3x3 matrices
//          (used for <= 3 neutrino flavors)"
//      }
//
//      @Article{Kopp:2007ne,
//        author    = "Kopp, Joachim and Lindner, Manfred and Ota,
//                     Toshihiko and Sato, Joe",
//        title     = "{Non-standard neutrino interactions in reactor and
//                     superbeam experiments}",
//        journal   = "Phys. Rev.",
//        volume    = "D77",
//        year      = "2008",
//        pages     = "013007",
//        eprint    = "0708.0152",
//        archivePrefix = "arXiv",
//       primaryClass  =  "hep-ph",
//       doi       = "10.1103/PhysRevD.77.013007",
//       SLACcitation  = "%%CITATION = 0708.0152;%%",
//       memo      = "This is the first paper in which an early version
//         of the present NSI engine was used.",
//     }
// ----------------------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include <complex.h>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_eigen.h>
#include "globes/globes.h"
#include "snu.h"

// Constants
#define GLB_V_FACTOR        7.63247e-14 // Conversion factor for matter potentials
#define GLB_Ne_MANTLE       0.5         // Effective electron numbers for calculation
#define GLB_Ne_CORE         0.468       //   of MSW potentials
#define RHO_THRESHOLD       0.001       // The minimum matter density below which
                                        // vacuum algorithms are used
#define M_SQRT3  1.73205080756887729352744634151     // sqrt(3)

// Macros
#define SQR(x)      ((x)*(x))                        // x^2
#define SQR_ABS(x)  (SQR(creal(x)) + SQR(cimag(x)))  // |x|^2
#define POW10(x)    (exp(M_LN10*(x)))                // 10^x
#define MIN(X,Y)    ( ((X) < (Y)) ? (X) : (Y) )
#define MAX(X,Y)    ( ((X) > (Y)) ? (X) : (Y) )
#define SIGN(a,b)   ( (b) > 0.0 ? (fabs(a)) : (-fabs(a)) )
#define KRONECKER(i,j)  ( (i)==(j) ? 1 : 0 )

// Fundamental oscillation parameters
int n_flavors = 0;
static int n_params  = 0;
static int n_angles  = 0;
static int n_phases  = 0;
static double th[SNU_MAX_FLAVORS+1][SNU_MAX_FLAVORS+1];// Mixing angles
static double delta[SNU_MAX_PHASES];           // Dirac CP phases
static double dmsq[SNU_MAX_FLAVORS-1];         // Mass squared differences
static double complex epsilon_s_plus_1[SNU_MAX_FLAVORS][SNU_MAX_FLAVORS]; // NSI in the source
static double complex epsilon_m[SNU_MAX_FLAVORS][SNU_MAX_FLAVORS];        // NSI in propagation
static double complex epsilon_d_plus_1[SNU_MAX_FLAVORS][SNU_MAX_FLAVORS]; // NSI in detector
#ifdef NU_USE_NUSQUIDS
  static double M_A_prime;        // dark force carrier mass
  static double g_prime;          // dark force carrier coupling
#endif

// Names of NSI parameters
char snu_param_strings[SNU_MAX_PARAMS][64];

// Internal temporary variables
static gsl_matrix_complex *U=NULL; // The vacuum mixing matrix
static gsl_matrix_complex *H=NULL; // Neutrino Hamiltonian
static gsl_matrix_complex *Q=NULL; // Eigenvectors of Hamiltonian (= eff. mixing matrix)
static gsl_vector *lambda=NULL;    // Eigenvalues of Hamiltonian
static gsl_matrix_complex *S=NULL; // The neutrino S-matrix

static gsl_matrix_complex *H0_template=NULL;  // Used in the construction of the vac. Hamiltonian
static gsl_matrix_complex *S1=NULL, *T0=NULL; // Temporary matrix storage
static gsl_matrix_complex *Q1=NULL, *Q2=NULL; // More temporary storage

static gsl_eigen_hermv_workspace *w=NULL;     // Workspace for eigenvector algorithm

extern int density_corr[];

// The order in which the rotation matrices corresponding to the different
// mixing angles are multiplied together (numbers are indices to th[][]
static int rotation_order[SNU_MAX_ANGLES][2];

// Which rotation matrices contain the complex phases? Indices are to
// delta[], -1 indicates no complex phase in a particular matrix;
// phase_order[0] corresponds to the leftmost rotation matrix
static int phase_order[SNU_MAX_ANGLES];

//FIXME Check if CP violation works the right way in mass-to-flavor
//
//TODO: Reintroduce static keyword for all variables


// ----------------------------------------------------------------------------
//     3 x 3   E I G E N S Y S T E M   F U N C T I O N S  (physics/0610206)
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
static void zhetrd3(double complex A[3][3], double complex Q[3][3],
                    double d[3], double e[2])
// ----------------------------------------------------------------------------
// Reduces a hermitian 3x3 matrix to real tridiagonal form by applying
// (unitary) Householder transformations:
//            [ d[0]  e[0]       ]
//    A = Q . [ e[0]  d[1]  e[1] ] . Q^T
//            [       e[1]  d[2] ]
// The function accesses only the diagonal and upper triangular parts of
// A. The access is read-only.
// ---------------------------------------------------------------------------
{
  const int n = 3;
  double complex u[n], q[n];
  double complex omega, f;
  double K, h, g;
  
  // Initialize Q to the identitity matrix
#ifndef EVALS_ONLY
  for (int i=0; i < n; i++)
  {
    Q[i][i] = 1.0;
    for (int j=0; j < i; j++)
      Q[i][j] = Q[j][i] = 0.0;
  }
#endif

  // Bring first row and column to the desired form 
  h = SQR_ABS(A[0][1]) + SQR_ABS(A[0][2]);
  if (creal(A[0][1]) > 0)
    g = -sqrt(h);
  else
    g = sqrt(h);
  e[0] = g;
  f    = g * A[0][1];
  u[1] = conj(A[0][1]) - g;
  u[2] = conj(A[0][2]);
  
  omega = h - f;
  if (creal(omega) > 0.0)
  {
    omega = 0.5 * (1.0 + conj(omega)/omega) / creal(omega);
    K = 0.0;
    for (int i=1; i < n; i++)
    {
      f    = conj(A[1][i]) * u[1] + A[i][2] * u[2];
      q[i] = omega * f;                  // p
      K   += creal(conj(u[i]) * f);      // u* A u
    }
    K *= 0.5 * SQR_ABS(omega);

    for (int i=1; i < n; i++)
      q[i] = q[i] - K * u[i];
    
    d[0] = creal(A[0][0]);
    d[1] = creal(A[1][1]) - 2.0*creal(q[1]*conj(u[1]));
    d[2] = creal(A[2][2]) - 2.0*creal(q[2]*conj(u[2]));
    
    // Store inverse Householder transformation in Q
#ifndef EVALS_ONLY
    for (int j=1; j < n; j++)
    {
      f = omega * conj(u[j]);
      for (int i=1; i < n; i++)
        Q[i][j] = Q[i][j] - f*u[i];
    }
#endif

    // Calculate updated A[1][2] and store it in f
    f = A[1][2] - q[1]*conj(u[2]) - u[1]*conj(q[2]);
  }
  else
  {
    for (int i=0; i < n; i++)
      d[i] = creal(A[i][i]);
    f = A[1][2];
  }

  // Make (23) element real
  e[1] = cabs(f);
#ifndef EVALS_ONLY
  if (e[1] != 0.0)
  {
    f = conj(f) / e[1];
    for (int i=1; i < n; i++)
      Q[i][n-1] = Q[i][n-1] * f;
  }
#endif
}


// ----------------------------------------------------------------------------
static int zheevc3(double complex A[3][3], double w[3])
// ----------------------------------------------------------------------------
// Calculates the eigenvalues of a hermitian 3x3 matrix A using Cardano's
// analytical algorithm.
// Only the diagonal and upper triangular parts of A are accessed. The access
// is read-only.
// ----------------------------------------------------------------------------
// Parameters:
//   A: The hermitian input matrix
//   w: Storage buffer for eigenvalues
// ----------------------------------------------------------------------------
// Return value:
//   0: Success
//  -1: Error
// ----------------------------------------------------------------------------
{
  double m, c1, c0;
  
  // Determine coefficients of characteristic poynomial. We write
  //       | a   d   f  |
  //  A =  | d*  b   e  |
  //       | f*  e*  c  |
  double complex de = A[0][1] * A[1][2];                            // d * e
  double dd = SQR_ABS(A[0][1]);                                  // d * conj(d)
  double ee = SQR_ABS(A[1][2]);                                  // e * conj(e)
  double ff = SQR_ABS(A[0][2]);                                  // f * conj(f)
  m  = creal(A[0][0]) + creal(A[1][1]) + creal(A[2][2]);
  c1 = (creal(A[0][0])*creal(A[1][1])  // a*b + a*c + b*c - d*conj(d) - e*conj(e) - f*conj(f)
          + creal(A[0][0])*creal(A[2][2])
          + creal(A[1][1])*creal(A[2][2]))
          - (dd + ee + ff);
  c0 = creal(A[2][2])*dd + creal(A[0][0])*ee + creal(A[1][1])*ff
            - creal(A[0][0])*creal(A[1][1])*creal(A[2][2])
            - 2.0 * (creal(A[0][2])*creal(de) + cimag(A[0][2])*cimag(de));
                             // c*d*conj(d) + a*e*conj(e) + b*f*conj(f) - a*b*c - 2*Re(conj(f)*d*e)

  double p, sqrt_p, q, c, s, phi;
  p = SQR(m) - 3.0*c1;
  q = m*(p - (3.0/2.0)*c1) - (27.0/2.0)*c0;
  sqrt_p = sqrt(fabs(p));

  phi = 27.0 * ( 0.25*SQR(c1)*(p - c1) + c0*(q + 27.0/4.0*c0));
  phi = (1.0/3.0) * atan2(sqrt(fabs(phi)), q);
  
  c = sqrt_p*cos(phi);
  s = (1.0/M_SQRT3)*sqrt_p*sin(phi);

  w[1]  = (1.0/3.0)*(m - c);
  w[2]  = w[1] + s;
  w[0]  = w[1] + c;
  w[1] -= s;

  return 0;
}


// ----------------------------------------------------------------------------
static int zheevq3(double complex A[3][3], double complex Q[3][3], double w[3])
// ----------------------------------------------------------------------------
// Calculates the eigenvalues and normalized eigenvectors of a hermitian 3x3
// matrix A using the QL algorithm with implicit shifts, preceded by a
// Householder reduction to real tridiagonal form.
// The function accesses only the diagonal and upper triangular parts of A.
// The access is read-only.
// ----------------------------------------------------------------------------
// Parameters:
//   A: The hermitian input matrix
//   Q: Storage buffer for eigenvectors
//   w: Storage buffer for eigenvalues
// ----------------------------------------------------------------------------
// Return value:
//   0: Success
//  -1: Error (no convergence)
// ----------------------------------------------------------------------------
// Dependencies:
//   zhetrd3()
// ----------------------------------------------------------------------------
{
  const int n = 3;
  double e[3];                 // The third element is used only as temporary workspace
  double g, r, p, f, b, s, c;  // Intermediate storage
  double complex t;
  int nIter;
  int m;

  // Transform A to real tridiagonal form by the Householder method
  zhetrd3(A, Q, w, e);
  
  // Calculate eigensystem of the remaining real symmetric tridiagonal matrix
  // with the QL method
  //
  // Loop over all off-diagonal elements
  for (int l=0; l < n-1; l++)
  {
    nIter = 0;
    while (1)
    {
      // Check for convergence and exit iteration loop if off-diagonal
      // element e(l) is zero
      for (m=l; m <= n-2; m++)
      {
        g = fabs(w[m])+fabs(w[m+1]);
        if (fabs(e[m]) + g == g)
          break;
      }
      if (m == l)
        break;
      
      if (nIter++ >= 30)
        return -1;

      // Calculate g = d_m - k
      g = (w[l+1] - w[l]) / (e[l] + e[l]);
      r = sqrt(SQR(g) + 1.0);
      if (g > 0)
        g = w[m] - w[l] + e[l]/(g + r);
      else
        g = w[m] - w[l] + e[l]/(g - r);

      s = c = 1.0;
      p = 0.0;
      for (int i=m-1; i >= l; i--)
      {
        f = s * e[i];
        b = c * e[i];
        if (fabs(f) > fabs(g))
        {
          c      = g / f;
          r      = sqrt(SQR(c) + 1.0);
          e[i+1] = f * r;
          c     *= (s = 1.0/r);
        }
        else
        {
          s      = f / g;
          r      = sqrt(SQR(s) + 1.0);
          e[i+1] = g * r;
          s     *= (c = 1.0/r);
        }
        
        g = w[i+1] - p;
        r = (w[i] - g)*s + 2.0*c*b;
        p = s * r;
        w[i+1] = g + p;
        g = c*r - b;

        // Form eigenvectors
#ifndef EVALS_ONLY
        for (int k=0; k < n; k++)
        {
          t = Q[k][i+1];
          Q[k][i+1] = s*Q[k][i] + c*t;
          Q[k][i]   = c*Q[k][i] - s*t;
        }
#endif 
      }
      w[l] -= p;
      e[l]  = g;
      e[m]  = 0.0;
    }
  }

  return 0;
}


// ----------------------------------------------------------------------------
static int zheevh3(double complex A[3][3], double complex Q[3][3], double w[3])
// ----------------------------------------------------------------------------
// Calculates the eigenvalues and normalized eigenvectors of a hermitian 3x3
// matrix A using Cardano's method for the eigenvalues and an analytical
// method based on vector cross products for the eigenvectors. However,
// if conditions are such that a large error in the results is to be
// expected, the routine falls back to using the slower, but more
// accurate QL algorithm. Only the diagonal and upper triangular parts of A need
// to contain meaningful values. Access to A is read-only.
// ----------------------------------------------------------------------------
// Parameters:
//   A: The hermitian input matrix
//   Q: Storage buffer for eigenvectors
//   w: Storage buffer for eigenvalues
// ----------------------------------------------------------------------------
// Return value:
//   0: Success
//  -1: Error
// ----------------------------------------------------------------------------
// Dependencies:
//   zheevc3(), zhetrd3(), zheevq3()
// ----------------------------------------------------------------------------
// Version history:
//   v1.1: Simplified fallback condition --> speed-up
//   v1.0: First released version
// ----------------------------------------------------------------------------
{
#ifndef EVALS_ONLY
  double norm;          // Squared norm or inverse norm of current eigenvector
//  double n0, n1;        // Norm of first and second columns of A
  double error;         // Estimated maximum roundoff error
  double t, u;          // Intermediate storage
  int j;                // Loop counter
#endif

  // Calculate eigenvalues
  zheevc3(A, w);

#ifndef EVALS_ONLY
//  n0 = SQR(creal(A[0][0])) + SQR_ABS(A[0][1]) + SQR_ABS(A[0][2]);
//  n1 = SQR_ABS(A[0][1]) + SQR(creal(A[1][1])) + SQR_ABS(A[1][2]);
  
  t = fabs(w[0]);
  if ((u=fabs(w[1])) > t)
    t = u;
  if ((u=fabs(w[2])) > t)
    t = u;
  if (t < 1.0)
    u = t;
  else
    u = SQR(t);
  error = 256.0 * DBL_EPSILON * SQR(u);
//  error = 256.0 * DBL_EPSILON * (n0 + u) * (n1 + u);

  Q[0][1] = A[0][1]*A[1][2] - A[0][2]*creal(A[1][1]);
  Q[1][1] = A[0][2]*conj(A[0][1]) - A[1][2]*creal(A[0][0]);
  Q[2][1] = SQR_ABS(A[0][1]);

  // Calculate first eigenvector by the formula
  //   v[0] = conj( (A - w[0]).e1 x (A - w[0]).e2 )
  Q[0][0] = Q[0][1] + A[0][2]*w[0];
  Q[1][0] = Q[1][1] + A[1][2]*w[0];
  Q[2][0] = (creal(A[0][0]) - w[0]) * (creal(A[1][1]) - w[0]) - Q[2][1];
  norm    = SQR_ABS(Q[0][0]) + SQR_ABS(Q[1][0]) + SQR(creal(Q[2][0]));

  // If vectors are nearly linearly dependent, or if there might have
  // been large cancellations in the calculation of A(I,I) - W(1), fall
  // back to QL algorithm
  // Note that this simultaneously ensures that multiple eigenvalues do
  // not cause problems: If W(1) = W(2), then A - W(1) * I has rank 1,
  // i.e. all columns of A - W(1) * I are linearly dependent.
  if (norm <= error)
    return zheevq3(A, Q, w);
  else                      // This is the standard branch
  {
    norm = sqrt(1.0 / norm);
    for (j=0; j < 3; j++)
      Q[j][0] = Q[j][0] * norm;
  }
  
  // Calculate second eigenvector by the formula
  //   v[1] = conj( (A - w[1]).e1 x (A - w[1]).e2 )
  Q[0][1]  = Q[0][1] + A[0][2]*w[1];
  Q[1][1]  = Q[1][1] + A[1][2]*w[1];
  Q[2][1]  = (creal(A[0][0]) - w[1]) * (creal(A[1][1]) - w[1]) - creal(Q[2][1]);
  norm     = SQR_ABS(Q[0][1]) + SQR_ABS(Q[1][1]) + SQR(creal(Q[2][1]));
  if (norm <= error)
    return zheevq3(A, Q, w);
  else
  {
    norm = sqrt(1.0 / norm);
    for (j=0; j < 3; j++)
      Q[j][1] = Q[j][1] * norm;
  }
  
  // Calculate third eigenvector according to
  //   v[2] = conj(v[0] x v[1])
  Q[0][2] = conj(Q[1][0]*Q[2][1] - Q[2][0]*Q[1][1]);
  Q[1][2] = conj(Q[2][0]*Q[0][1] - Q[0][0]*Q[2][1]);
  Q[2][2] = conj(Q[0][0]*Q[1][1] - Q[1][0]*Q[0][1]);
#endif

  return 0;
}


// ----------------------------------------------------------------------------
//                    I N T E R N A L   F U N C T I O N S
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
int snu_print_gsl_matrix_complex(gsl_matrix_complex *A)
// ----------------------------------------------------------------------------
// Print entries of a complex GSL matrix in human-readable form
// ----------------------------------------------------------------------------
{
  int i, j;
  for (i=0; i < A->size1; i++)
  {
    for (j=0; j < A->size2; j++)
    {
//      printf("%10.4g +%10.4g*I   ", GSL_REAL(gsl_matrix_complex_get(A, i, j)),
//             GSL_IMAG(gsl_matrix_complex_get(A, i, j)));
      printf("%15.7g +%15.7g*I   ", GSL_REAL(gsl_matrix_complex_get(A, i, j)),
             GSL_IMAG(gsl_matrix_complex_get(A, i, j))); //TODO: Revert to old format?
    } 
    printf("\n");
  }

  return 0;
}


// ----------------------------------------------------------------------------
int snu_init_probability_engine_3()
// ----------------------------------------------------------------------------
// Initialize probability engine for the 3-flavor case with NSI (no steriles)
// ----------------------------------------------------------------------------
// Return values:
//   > 0: number of oscillation parameters defined
//   < 0: error
// ----------------------------------------------------------------------------
{
  int rotation_order[][2] = { {2,3}, {1,3}, {1,2} };
  int phase_order[] = { -1, 0, -1 };
  return snu_init_probability_engine(3, rotation_order, phase_order);
}


// ----------------------------------------------------------------------------
int snu_init_probability_engine(int _n_flavors, int _rotation_order[][2], int _phase_order[])
// ----------------------------------------------------------------------------
// Allocates internal data structures for the probability engine.
// ----------------------------------------------------------------------------
// Return values:
//   > 0: number of oscillation parameters defined
//   < 0: error
// ----------------------------------------------------------------------------
{
  if (_n_flavors < 3 || _n_flavors > SNU_MAX_FLAVORS)
  {
    fprintf(stderr, "snu_init_probability_engine: Too many or too few neutrino flavors (%d).\n",
            _n_flavors);
    return -1;
  }

  // Number of oscillation parameters:
  //   n (n-1)/2 mixing angles, (n-1)(n-2)/2 phases, n-1 mass squared differences
  //   n^2 |\eps^s|, n^2 \phi^s,
  //   n (n+1)/2 |\eps^m|, n(n-1)/2 \phi^m,
  //   n^2 |\eps^d|, n^2 \phi^d
  // = 6 n^2 - n
  n_flavors = _n_flavors;
  n_params  = 6*SQR(n_flavors) - n_flavors;
  n_angles  = (n_flavors * (n_flavors-1))/2;
  n_phases  = ((n_flavors-1)*(n_flavors-2))/2;

  snu_free_probability_engine();
  
  U = gsl_matrix_complex_calloc(n_flavors, n_flavors);
  H = gsl_matrix_complex_calloc(n_flavors, n_flavors);
  Q = gsl_matrix_complex_calloc(n_flavors, n_flavors);
  lambda = gsl_vector_alloc(n_flavors);
  S = gsl_matrix_complex_calloc(n_flavors, n_flavors);
    
  H0_template = gsl_matrix_complex_calloc(n_flavors, n_flavors);
  S1 = gsl_matrix_complex_calloc(n_flavors, n_flavors);
  T0 = gsl_matrix_complex_calloc(n_flavors, n_flavors);
  Q1 = gsl_matrix_complex_calloc(n_flavors, n_flavors);
  Q2 = gsl_matrix_complex_calloc(n_flavors, n_flavors);

  w  = gsl_eigen_hermv_alloc(n_flavors);

  for (int i=0; i < n_angles; i++)
  {
    if (_rotation_order[i][0] < 1 || _rotation_order[i][0] > n_angles ||
        _rotation_order[i][1] < 1 || _rotation_order[i][1] > n_angles)
    {
      fprintf(stderr, "snu_init_probability_engine: Incorrect rotation order specification.\n");
      return -2;
    }
    if (_phase_order[i] >= n_phases)
    {
      fprintf(stderr, "snu_init_probability_engine: Incorrect phase order specification.\n");
      return -3;
    }
    rotation_order[i][0] = _rotation_order[i][0];
    rotation_order[i][1] = _rotation_order[i][1];
    phase_order[i]       = _phase_order[i];
  }

//  printf("Order of rotations:\n");
//  for (int i=0; i < n_angles; i++)
//    printf("{%d,%d} ", rotation_order[i][0], rotation_order[i][1]);
//  printf("\n");
//  printf("Order of phases:\n");
//  for (int i=0; i < n_angles; i++)
//    printf("%d ", phase_order[i]);
//  printf("\n");


  // Define names of oscillation parameters
  sprintf(snu_param_strings[0], "%s", "TH12");    // Standard oscillation parameters
  sprintf(snu_param_strings[1], "%s", "TH13");
  sprintf(snu_param_strings[2], "%s", "TH23");
  sprintf(snu_param_strings[3], "%s", "DELTA_0");
  sprintf(snu_param_strings[4], "%s", "DM21");
  sprintf(snu_param_strings[5], "%s", "DM31");

  int k = 6;
  for (int i=4; i <= n_flavors; i++)            // Mass squared differences
    sprintf(snu_param_strings[k++], "DM%d1", i);

  for (int i=1; i <= n_flavors; i++)            // Sterile mixing angles
    for (int j=MAX(i+1,4); j <= n_flavors; j++)
      sprintf(snu_param_strings[k++], "TH%d%d", i, j);

  for (int i=1; i <= n_phases-1; i++)
    sprintf(snu_param_strings[k++], "DELTA_%d", i); // Sterile phases

  const char *flavors[] = { "E", "MU", "TAU", "S1", "S2", "S3", "S4", "S5", "S6" };
  for (int i=0; i < n_flavors; i++)             // Source NSI
    for (int j=0; j < n_flavors; j++)
    {
      sprintf(snu_param_strings[k++], "ABS_EPS_S_%s%s", flavors[j], flavors[i]);
      sprintf(snu_param_strings[k++], "ARG_EPS_S_%s%s", flavors[j], flavors[i]);
    }

  for (int i=0; i < n_flavors; i++)             // Propagation NSI
  {
    sprintf(snu_param_strings[k++], "EPS_M_%s%s", flavors[i], flavors[i]);
    for (int j=i+1; j < n_flavors; j++)
    {
      sprintf(snu_param_strings[k++], "ABS_EPS_M_%s%s", flavors[i], flavors[j]);
      sprintf(snu_param_strings[k++], "ARG_EPS_M_%s%s", flavors[i], flavors[j]);
    }
  }

  for (int i=0; i < n_flavors; i++)             // Detector NSI
    for (int j=0; j < n_flavors; j++)
    {
      sprintf(snu_param_strings[k++], "ABS_EPS_D_%s%s", flavors[j], flavors[i]);
      sprintf(snu_param_strings[k++], "ARG_EPS_D_%s%s", flavors[j], flavors[i]);
    }

  // ADD-ON: The following is added to allow the alternative parametrization in terms
  // of mixing matrix elements used by Thomas
  if (n_flavors >= 3)
  {
    sprintf(snu_param_strings[k++], "Ue3");
    n_params++;
  }
  if (n_flavors == 4)
  {
    sprintf(snu_param_strings[k++], "Ue4");
    sprintf(snu_param_strings[k++], "Um4");
    sprintf(snu_param_strings[k++], "Ut4");
    sprintf(snu_param_strings[k++], "s22thmue");
    n_params += 4;
  }
  if (n_flavors == 5)
  {
    sprintf(snu_param_strings[k++], "Ue4");
    sprintf(snu_param_strings[k++], "Um4");
    sprintf(snu_param_strings[k++], "Ue5");
    sprintf(snu_param_strings[k++], "Um5");
    sprintf(snu_param_strings[k++], "Ue4Um4");
    sprintf(snu_param_strings[k++], "Ue5Um5");
    n_params += 6;  // Remember to change when adding parameters!
  }

  // Extra parameters for oscillation + decay scenario
#ifdef NU_USE_NUSQUIDS
  sprintf(snu_param_strings[k++], "M_A_PRIME");  // dark force mediator mass
  sprintf(snu_param_strings[k++], "G_PRIME");    // dark force mediator coupling to \nu_s
  sprintf(snu_param_strings[k++], "MA_OVER_M4"); // alternative way of defining M_{A'}
  n_params += 3;
  if (n_flavors > 3)
  {
    sprintf(snu_param_strings[k++], "M4_GAMMA");
                     // m_4*\Gamma_{A'} - alternative way of specifying coupling strength
    n_params++;
  }
#endif

  if (k != n_params)
  {
    fprintf(stderr, "snu_init_probability_engine: n_params has an incorrect value (%d).\n",
            n_params);
    return -2;
  }

//  printf("Oscillation engine initialized for %d neutrino flavors\n", n_flavors);
//  printf("Oscillation parameters are:\n");
//  for (int i=0; i < n_params; i++)
//  {
//    printf("  %-20s", snu_param_strings[i]);
//    if (i % 4 == 3)  printf("\n");
//  }

  return n_params;
}


// ----------------------------------------------------------------------------
int snu_free_probability_engine()
// ----------------------------------------------------------------------------
// Destroys internal data structures of the probability engine.
// ----------------------------------------------------------------------------
{
  if (w !=NULL)     { gsl_eigen_hermv_free(w);      w  = NULL; }

  if (Q2!=NULL)     { gsl_matrix_complex_free(Q2);  Q2 = NULL; }
  if (Q1!=NULL)     { gsl_matrix_complex_free(Q1);  Q1 = NULL; }
  if (T0!=NULL)     { gsl_matrix_complex_free(T0);  T0 = NULL; }
  if (S1!=NULL)     { gsl_matrix_complex_free(S1);  S1 = NULL; }
  if (H0_template!=NULL) { gsl_matrix_complex_free(H0_template);  H0_template = NULL; }
  
  if (S!=NULL)      { gsl_matrix_complex_free(S);   S = NULL; }
  if (lambda!=NULL) { gsl_vector_free(lambda);      lambda = NULL; }
  if (Q!=NULL)      { gsl_matrix_complex_free(Q);   Q = NULL; }
  if (H!=NULL)      { gsl_matrix_complex_free(H);   H = NULL; }
  if (U!=NULL)      { gsl_matrix_complex_free(U);   U = NULL; }

  return 0;
}


// ----------------------------------------------------------------------------
int snu_set_oscillation_parameters(glb_params p, void *user_data)
// ----------------------------------------------------------------------------
// Sets the fundamental oscillation parameters and precomputes the mixing
// matrix and part of the Hamiltonian.
// ----------------------------------------------------------------------------
{
  // FIXME switch back to setting parameters to NaN in case of inconsistencies?
  gsl_matrix_complex *R = gsl_matrix_complex_alloc(n_flavors, n_flavors);
  gsl_matrix_complex *T = gsl_matrix_complex_alloc(n_flavors, n_flavors);
  double complex (*_R)[n_flavors]
    = (double complex (*)[n_flavors]) gsl_matrix_complex_ptr(R, 0, 0);
  int i, j, k;
  int status = 0;

  // Implement correlations between density parameters. This requires that under
  // all circumstances the scaling of the matter density is performed _after_
  // calling set_oscillation_parameters! At present, this works only with
  // the hybrid minimizer (GLB_MIN_POWELL)!
//  for (j=0; j < glb_num_of_exps; j++)
//    if (density_corr[j] != j)
//      glbSetDensityParams(p, glbGetDensityParams(p, density_corr[j]), j);

  // Copy oscillation parameters
  th[1][2] = glbGetOscParams(p, GLB_THETA_12);    // Standard parameters
  th[1][3] = glbGetOscParams(p, GLB_THETA_13);
  th[2][3] = glbGetOscParams(p, GLB_THETA_23);
  delta[0] = glbGetOscParams(p, GLB_DELTA_CP);
  dmsq[0]  = glbGetOscParams(p, GLB_DM_21);
  dmsq[1]  = glbGetOscParams(p, GLB_DM_31);

  k = 6;
  for (i=4; i <= n_flavors; i++)                // Mass squared differences
    dmsq[i-2] = glbGetOscParams(p, k++);

  for (i=1; i <= n_flavors; i++)                // Sterile mixing angles
    for (j=MAX(i+1,4); j <= n_flavors; j++)
      th[i][j] = glbGetOscParams(p, k++);

  for (i=1; i <= n_phases-1; i++)               // Sterile phases
    delta[i] = glbGetOscParams(p, k++);

  for (i=0; i < n_flavors; i++)                 // Source NSI
  {
    for (j=0; j < n_flavors; j++)
    {
      epsilon_s_plus_1[i][j] = glbGetOscParams(p,k) * cexp(I*glbGetOscParams(p,k+1));
      k += 2;
    }
    epsilon_s_plus_1[i][i] += 1.0;
  }

  for (i=0; i < n_flavors; i++)                 // Propagation NSI
  {
    epsilon_m[i][i] = glbGetOscParams(p,k);
    k++;
    for (j=i+1; j < n_flavors; j++)
    {
      epsilon_m[i][j] = glbGetOscParams(p,k) * cexp(I*glbGetOscParams(p,k+1));
      epsilon_m[j][i] = conj(epsilon_m[i][j]);
      k += 2;
    }
  }

  for (i=0; i < n_flavors; i++)                 // Detector NSI
  {
    for (j=0; j < n_flavors; j++)
    {
      epsilon_d_plus_1[i][j] = glbGetOscParams(p,k) * cexp(I*glbGetOscParams(p,k+1));
      k += 2;
    }
    epsilon_d_plus_1[i][i] += 1.0;
  }

  // Extra parameters (Thomas' parameterization in terms of mixing matrix elements)
  // ------------------------------------------------------------------------------

  // 3 flavors: Ue3
  if (n_flavors == 3)
  {
    if (fabs(glbGetOscParamByName(p, "Ue3")) > 1e-12)
    {
      double _th13 = asin(glbGetOscParamByName(p, "Ue3"));
      if (fabs(th[1][3]) > 1e-12)
      {
        fprintf(stderr, "snu_set_oscillation_parameters: Warning: Both TH13 and Ue3 != 0. "
                        "Ignoring Ue3.\n");
        status = -2;
      }
      else if (!isfinite(_th13))
      {
        fprintf(stderr, "snu_set_oscillation_parameters: Warning: Inconsistent Ue3 set to zero.\n");
        status = -1;
        th[1][3] = 0.;
      }
      else
        th[1][3] = _th13;
    }
  }

  // 4 flavors: Ue3, Ue4, Um4, Ut4, s22thmue
  else if (n_flavors == 4)
  {
    if (fabs(glbGetOscParamByName(p, "Ue4")) > 1e-12)
    {
      double _th14 = asin(glbGetOscParamByName(p, "Ue4"));
      if (fabs(th[1][4]) > 1e-12)
      {
        fprintf(stderr, "snu_set_oscillation_parameters: Warning: Both TH14 and Ue4 != 0. "
                        "Ignoring Ue4.\n");
        status = -4;
      }
      else if (!isfinite(_th14))
      {
        fprintf(stderr, "snu_set_oscillation_parameters: Warning: Inconsistent Ue4 set to zero.\n");
        status = -3;
        th[1][4] = 0.0;
      }
      else
        th[1][4] = _th14;
    }

    if (fabs(glbGetOscParamByName(p, "Um4")) > 1e-12)
    {
      double _th24 = asin(glbGetOscParamByName(p, "Um4")/cos(th[1][4]));
      if (fabs(th[2][4]) > 1e-12)
      {
        fprintf(stderr, "snu_set_oscillation_parameters: Warning: Both TH24 and Um4 != 0. "
                        "Ignoring Um4.\n");
        status = -6;
      }
      else if (!isfinite(_th24))
      {
        fprintf(stderr, "snu_set_oscillation_parameters: Warning: Inconsistent Um4 set to zero.\n");
        status = -5;
        th[2][4] = 0.0;
      }
      else
        th[2][4] = _th24;
    }

    if (fabs(glbGetOscParamByName(p, "Ut4")) > 1e-12)
    {
      double _th34 = asin(glbGetOscParamByName(p, "Ut4") / (cos(th[1][4])*cos(th[2][4])));
      if (fabs(th[3][4]) > 1e-12)
      {
        fprintf(stderr, "snu_set_oscillation_parameters: Warning: Both TH34 and Ut4 != 0. "
                        "Ignoring Ut4.\n");
        status = -30;
      }
      else if (!isfinite(_th34))
      {
        fprintf(stderr, "snu_set_oscillation_parameters: Warning: Inconsistent Ut4 set to zero.\n");
        status = -31;
        th[3][4] = 0.0;
      }
      else
        th[3][4] = _th34;
    }

    if (fabs(glbGetOscParamByName(p, "Ue3")) > 1e-12)
    {
      double _th13 = asin(glbGetOscParamByName(p, "Ue3")/cos(th[1][4]));
      if (fabs(th[1][3]) > 1e-12)
      {
        fprintf(stderr, "snu_set_oscillation_parameters: Warning: Both TH13 and Ue3 != 0. "
                        "Ignoring Ue3.\n");
        status = -8;
      }
      else if (!isfinite(_th13))
      {
        fprintf(stderr, "snu_set_oscillation_parameters: Warning: Inconsistent Ue3 set to zero.\n");
        status = -7;
        th[1][3] = 0.0;
      }
      else
        th[1][3] = _th13;
    }

    if (fabs(glbGetOscParamByName(p, "s22thmue")) > 1e-12)
    {
      double _th14;

      // special treatment in case several of the non-canonical parameters are given.
      // FIXME in principle, we should do this for all possible combinations ...
      if (fabs(glbGetOscParamByName(p, "Um4")) > 1e-12)
      {
        double _th24;
        _th14 = asin( 0.5 * sqrt(glbGetOscParamByName(p, "s22thmue"))
                          / glbGetOscParamByName(p, "Um4") );
        _th24 = asin(glbGetOscParamByName(p, "Um4")/cos(_th14));
        if (!isfinite(_th24))
        {
          fprintf(stderr, "snu_set_oscillation_parameters: Warning: Inconsistent Um4 set to zero.\n");
          status = -500;
          th[2][4] = 0.0;
        }
        else
          th[2][4] = _th24;
      }
      else
        _th14 = 0.5 * asin( sqrt(glbGetOscParamByName(p, "s22thmue")) / sin(th[2][4]) );

      if (!isfinite(_th14))
      {
        fprintf(stderr, "snu_set_oscillation_parameters: Warning: inconsistent "
                        "s22thmue=%g set to zero (th24=%g).\n",
                        fabs(glbGetOscParamByName(p, "s22thmue")), th[2][4]);
        status = -9;
        th[1][4] = 0.0;
      }
      else
        th[1][4] = _th14;
    }
  }

  // 5 flavors: Ue3, Ue4, Um4, Ue5, Um5, Ue4Um4, Ue5Um5
  else if (n_flavors == 5)
  {
    // Ue5
    if (fabs(glbGetOscParamByName(p, "Ue5")) > 1e-12)
    {
      double _th15 = asin(glbGetOscParamByName(p, "Ue5"));
      if (fabs(th[1][5]) > 1e-12)
      {
        fprintf(stderr, "snu_set_oscillation_parameters: Warning: Both TH15 and Ue5 != 0. "
                        "Ignoring Ue5.\n");
        status = -11;
      }
      else if (!isfinite(_th15))
      {
        fprintf(stderr, "snu_set_oscillation_parameters: Warning: Inconsistent Ue5 set to zero.\n");
        status = -10;
        th[1][5] = 0.0;
      }
      else
        th[1][5] = _th15;
    }

    // Ue5Um5
    if (fabs(glbGetOscParamByName(p, "Ue5Um5")) > 1e-12)
    {
      if (fabs(th[1][5]) > 1e-12)         // Case 1: Ue5Um5 and th15 or Ue5 specified
      {
        if (fabs(glbGetOscParamByName(p, "Um5")) > 1e-12 || fabs(th[2][5]) > 1e-12)
        {
          fprintf(stderr, "snu_set_oscillation_parameters: Warning: Only two of the parameters "
                          "Ue5Um5, Ue5, Um5, TH15, TH25 can be != 0. Ignoring Ue5Um5.\n");
          status = -20;
        }
        else
        {
          double _th25 = asin(glbGetOscParamByName(p, "Ue5Um5") / (cos(th[1][5])*sin(th[1][5])));
          if (!isfinite(_th25))
          {
            fprintf(stderr, "snu_set_oscillation_parameters: Warning: Inconsistent Ue5Um5 set to zero "
                            "(Ue5Um5=%g, th15=%g).\n", glbGetOscParamByName(p, "Ue5Um5"), th[1][5]);
            status = -21;
            th[2][5] = 0.0;
          }
          else
            th[2][5] = _th25;
        }
      }
      else if (fabs(th[2][5]) > 1e-12)    // Case 2: Ue5Um5 and th25 specified
      {
        if (fabs(glbGetOscParamByName(p, "Um5")) > 1e-12)
        {
          fprintf(stderr, "snu_set_oscillation_parameters: Warning: Ue5Um5, TH25 and Um5 != 0. "
                          "Ignoring Um5.\n");
          status = -22;
        }

        double _th15 = asin(2.0 * glbGetOscParamByName(p, "Ue5Um5") / sin(th[2][5])) / 2.0;
        if (!isfinite(_th15))
        {
          fprintf(stderr, "snu_set_oscillation_parameters: Warning: Inconsistent Ue5Um5 set to zero.\n");
          status = -23;
          th[1][5] = 0.0;
        }
        else
          th[1][5] = _th15;
      }
      else if (fabs(glbGetOscParamByName(p, "Um5")) > 1e-12)  // Case 2: Ue5Um5 and Um5 specified
      {
        double _th15 = asin(glbGetOscParamByName(p, "Ue5Um5") / glbGetOscParamByName(p, "Um5"));
        if (!isfinite(_th15))
        {
          fprintf(stderr, "snu_set_oscillation_parameters: Warning: Inconsistent Ue5Um5 set to zero.\n");
          status = -24;
          th[1][5] = 0.0;
        }
        else
          th[1][5] = _th15;
      }
    }

    // Um5
    if (fabs(glbGetOscParamByName(p, "Um5")) > 1e-12)
    {
      double _th25 = asin(glbGetOscParamByName(p, "Um5")/cos(th[1][5]));
      if (fabs(th[2][5]) > 1e-12)
      {
        fprintf(stderr, "snu_set_oscillation_parameters: Warning: Both TH25 and Um5 != 0. "
                        "Ignoring Um5.\n");
        status = -13;
      }
      else if (!isfinite(_th25))
      {
        fprintf(stderr, "snu_set_oscillation_parameters: Warning: Inconsistent Um5 set to zero.\n");
        status = -12;
        th[2][5] = 0.0;
      }
      else
        th[2][5] = _th25;
    }

    // Ue4
    if (fabs(glbGetOscParamByName(p, "Ue4")) > 1e-12)
    {
      double _th14 = asin(glbGetOscParamByName(p, "Ue4")/cos(th[1][5]));
      if (fabs(th[1][4]) > 1e-12)
      {
        fprintf(stderr, "snu_set_oscillation_parameters: Warning: Both TH14 and Ue4 != 0. "
                        "Ignoring Ue4.\n");
        status = -15;
      }
      else if (!isfinite(_th14))
      {
        fprintf(stderr, "snu_set_oscillation_parameters: Warning: Inconsistent Ue4 set to zero.\n");
        status = -14;
        th[1][4] = 0.0;
      }
      else
        th[1][4] = _th14;
    }

    // Ue4Ume4
    if (fabs(glbGetOscParamByName(p, "Ue4Um4")) > 1e-12)
    {
      if (fabs(th[1][4]) > 1e-12)         // Case 1: Ue4Um4 and th14 or Ue4 specified
      {
        if (fabs(glbGetOscParamByName(p, "Um4")) > 1e-12 || fabs(th[2][4]) > 1e-12)
        {
          fprintf(stderr, "snu_set_oscillation_parameters: Warning: Only two of the parameters "
                          "Ue4Um4, Ue4, Um4, TH14, TH24 can be != 0. Ignoring Ue4Um4.\n");
          status = -30;
        }
        else
        {
          // Write Um4 as A*sin(th24) - B e^{I delta}
          double Um4sq = SQR(glbGetOscParamByName(p, "Ue4Um4") / (cos(th[1][5]) * sin(th[1][4])) );
          double A     = cos(th[1][4]) * cos(th[2][5]);
          double B     = sin(th[1][4]) * sin(th[1][5]) * sin(th[2][5]);
          double delta = glbGetOscParamByName(p, "DELTA_2") - glbGetOscParamByName(p, "DELTA_1");
          double D     = sqrt( Um4sq - SQR(B)*SQR(sin(delta)) );

          if (!isfinite(D))
          {
            fprintf(stderr, "snu_set_oscillation_parameters: Warning: Inconsistent Ue4Um4 set to zero "
                            "(D^2 < 0).\n");
            status = -31;
            th[2][4] = 0.0;
          }
          else
          {
            double _sinth24 = (B*cos(delta) - D) / A; // Take smallest positive solution for th24
            if (_sinth24 < 0)
              _sinth24 = (B*cos(delta) + D) / A;
            else
              fprintf(stderr, "snu_set_oscillation_parameters: Warning: Using |Ue4Um4| to specify TH24 "
                              "is not unique. Using smaller solution for TH24.\n");

            if (fabs(_sinth24) > 1.0)
            {
              fprintf(stderr, "snu_set_oscillation_parameters: Warning: Inconsistent Ue4Um4 set to zero "
                              "(|sin(th24)| > 1).\n");
              status = -32;
              th[2][4] = 0.0;
            }
            else
              th[2][4] = asin(_sinth24);
          }
        }
      }
      else                                // Case 2: Ue4Um4 and th24 or Um4 specified
      {                                   // Leads to transcendental equations :-(
        fprintf(stderr, "snu_set_oscillation_parameters: Warning: Ue4Um4 must be combined "
                        "with TH14 or Ue4. Ignoring Ue4Um4.\n");
        status = -32;
      }
    }

    // Um4
    if (fabs(glbGetOscParamByName(p, "Um4")) > 1e-12)
    {
      // Write Um4 as A*sin(th24) - B e^{I delta}
      double Um4sq = SQR(glbGetOscParamByName(p, "Um4"));
      double A     = cos(th[1][4]) * cos(th[2][5]);
      double B     = sin(th[1][4]) * sin(th[1][5]) * sin(th[2][5]);
      double delta = glbGetOscParamByName(p, "DELTA_2") - glbGetOscParamByName(p, "DELTA_1");
      double D     = sqrt( Um4sq - SQR(B)*SQR(sin(delta)) );

      if (fabs(th[2][4]) > 1e-12)
      {
        fprintf(stderr, "snu_set_oscillation_parameters: Warning: Both TH24 and Um4 != 0. "
                        "Ignoring Um4.\n");
        status = -17;
      }
      else if (!isfinite(D))
      {
        fprintf(stderr, "snu_set_oscillation_parameters: Warning: Inconsistent Um4 set to zero "
                        "(D^2 < 0).\n");
        status = -16;
        th[2][4] = 0.0;
      }
      else
      {
        double _sinth24 = (B*cos(delta) - D) / A; // Take smallest positive solution for th24
        if (_sinth24 < 0)
          _sinth24 = (B*cos(delta) + D) / A;
        else
          fprintf(stderr, "snu_set_oscillation_parameters: Warning: Using |Um4| to specify TH24 "
                          "is not unique. Using smaller solution for TH24.\n");

        if (fabs(_sinth24) > 1.0)
        {
          fprintf(stderr, "snu_set_oscillation_parameters: Warning: Inconsistent Um4 set to zero "
                          "(|sin(th24)| > 1).\n");
          status = -18;
          th[2][4] = 0.0;
        }
        else
          th[2][4] = asin(_sinth24);
      }
    }
 
    // Ue3
    if (fabs(glbGetOscParamByName(p, "Ue3")) > 1e-12)
    {
      double _th13 = asin(glbGetOscParamByName(p, "Ue3")/(cos(th[1][4])*cos(th[1][5])));
      if (fabs(th[1][3]) > 1e-12)
      {
        fprintf(stderr, "snu_set_oscillation_parameters: Warning: Both TH13 and Ue3 != 0. "
                        "Ignoring Ue3.\n");
        status = -19;
      }
      else if (!isfinite(_th13))
      {
        fprintf(stderr, "snu_set_oscillation_parameters: Warning: Inconsistent Ue3 set to zero.\n");
        status = -18;
        th[1][3] = 0.0;
      }
      else
        th[1][3] = _th13;
    }
  }


  // Multiply rotation matrices
  gsl_matrix_complex_set_identity(U);
  for (i=0; i < n_angles; i++)
  {
    int u = rotation_order[i][0] - 1;
    int v = rotation_order[i][1] - 1;
    double complex c = cos(th[u+1][v+1]);
    double complex s = sin(th[u+1][v+1]);
    if (phase_order[i] >= 0)
      s *= cexp(-I * delta[phase_order[i]]);

    gsl_matrix_complex_set_identity(R);
    _R[u][u] = c;
    _R[v][v] = c;
    _R[u][v] = s;
    _R[v][u] = -conj(s);

//    printf("Multiplying in R[%d][%d], phase %d\n", u+1, v+1, phase_order[i]);
//    gsl_matrix_complex_fprintf(stdout, R, "%g");

    gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, GSL_COMPLEX_ONE, U, R,           // T = U.R
                   GSL_COMPLEX_ZERO, T);
    gsl_matrix_complex_memcpy(U, T);                                            // U = T
  }


  // Calculate energy independent matrix H0 * E
  gsl_matrix_complex_set_zero(H0_template);
  gsl_matrix_complex_set_zero(H);
  for (i=1; i < n_flavors; i++)
    gsl_matrix_complex_set(H0_template, i, i, gsl_complex_rect(0.5*dmsq[i-1], 0.0));

  gsl_blas_zgemm(CblasNoTrans, CblasConjTrans, GSL_COMPLEX_ONE, H0_template, U, // T=H0.U^\dagger
                 GSL_COMPLEX_ZERO, T);
  gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, GSL_COMPLEX_ONE, U, T,             // H0=U.T
                 GSL_COMPLEX_ZERO, H0_template);

  gsl_matrix_complex_free(T);
  gsl_matrix_complex_free(R);


  // Extra parameters for oscillation + decay scenario
  // -------------------------------------------------
#ifdef NU_USE_NUSQUIDS
  // transmit parameters to Ivan's osc/decay code; to make sure all parameters
  // are consistently set (including for instance th14, th24 in case the user,
  // gives only Ue4, Um4), we first issue a call to snu_get_oscillation_parameters.
  // afterwards, we *overwrite* all osc/decay related parameters again with the
  // values from vector p because snu_get_oscillation_parameters will give
  // nonsensical results if these parameters are not set yet
  glb_params tmp_params = glbAllocParams();
  if (tmp_params)
  {
    snu_get_oscillation_parameters(tmp_params, user_data);
    glbSetOscParamByName(tmp_params, glbGetOscParamByName(p, "M_A_PRIME"),  "M_A_PRIME");
    glbSetOscParamByName(tmp_params, glbGetOscParamByName(p, "G_PRIME"),    "G_PRIME");
    glbSetOscParamByName(tmp_params, glbGetOscParamByName(p, "MA_OVER_M4"), "MA_OVER_M4");
    glbSetOscParamByName(tmp_params, glbGetOscParamByName(p, "M4_GAMMA"),   "M4_GAMMA");
    status += snu_set_oscillation_parameters_osc_decay_internal(n_flavors, tmp_params);

    // Now calls snu_get_oscillation_parameters *again* to retrieve the final
    // values of M_{A'} and g' and store them
    snu_get_oscillation_parameters_osc_decay_internal(n_flavors, tmp_params);
    M_A_prime = glbGetOscParamByName(tmp_params, "M_A_PRIME");
    g_prime   = glbGetOscParamByName(tmp_params, "G_PRIME");

    glbFreeParams(tmp_params);
  }
  else
  {
    fprintf(stderr, "snu_set_oscillation_parameters: error allocating glb_params vector.\n");
    status = -1003;
  }
#endif

  return status;
}


// ----------------------------------------------------------------------------
int snu_get_oscillation_parameters(glb_params p, void *user_data)
// ----------------------------------------------------------------------------
// Returns the current set of oscillation parameters.
// ----------------------------------------------------------------------------
{
  int i, j, k;
  glbDefineParams(p, th[1][2], th[1][3], th[2][3], delta[0], dmsq[0], dmsq[1]);
  
  k = 6;
  for (i=4; i <= n_flavors; i++)                // Mass squared differences
    glbSetOscParams(p, dmsq[i-2], k++);

  for (i=1; i <= n_flavors; i++)                // Sterile mixing angles
    for (j=MAX(i+1,4); j <= n_flavors; j++)
      glbSetOscParams(p, th[i][j], k++);

  for (i=1; i <= n_phases-1; i++)                // Sterile phases
    glbSetOscParams(p, delta[i], k++);

  for (i=0; i < n_flavors; i++)                 // Source NSI
    for (j=0; j < n_flavors; j++)
    {
      if (i == j)
      {
        glbSetOscParams(p, cabs(epsilon_s_plus_1[i][j] - 1.0), k);
        glbSetOscParams(p, carg(epsilon_s_plus_1[i][j] - 1.0), k+1);
      }
      else
      {
        glbSetOscParams(p, cabs(epsilon_s_plus_1[i][j]), k);
        glbSetOscParams(p, carg(epsilon_s_plus_1[i][j]), k+1);
      }
      k += 2;
    }

  for (i=0; i < n_flavors; i++)                 // Propagation NSI
  {
    glbSetOscParams(p, epsilon_m[i][i], k);
    k++;
    for (j=i+1; j < n_flavors; j++)
    {
      glbSetOscParams(p, cabs(epsilon_m[i][j]), k);
      glbSetOscParams(p, carg(epsilon_m[i][j]), k+1);
      k += 2;
    }
  }
  
  for (i=0; i < n_flavors; i++)                 // Detector NSI
    for (j=0; j < n_flavors; j++)
    {
      if (i == j)
      {
        glbSetOscParams(p, cabs(epsilon_d_plus_1[i][j] - 1.0), k);
        glbSetOscParams(p, carg(epsilon_d_plus_1[i][j] - 1.0), k+1);
      }
      else
      {
        glbSetOscParams(p, cabs(epsilon_d_plus_1[i][j]), k);
        glbSetOscParams(p, carg(epsilon_d_plus_1[i][j]), k+1);
      }
      k += 2;
    }

  if (n_flavors == 3)                           // ADD-ON: Thomas' parametrization
    glbSetOscParamByName(p, sin(th[1][3]), "Ue3");
  else if (n_flavors == 4)
  {
    glbSetOscParamByName(p, sin(th[1][3])*cos(th[1][4]), "Ue3");
    glbSetOscParamByName(p, sin(th[1][4]), "Ue4");
    glbSetOscParamByName(p, sin(th[2][4])*cos(th[1][4]), "Um4");
    glbSetOscParamByName(p, sin(th[3][4])*cos(th[1][4])*cos(th[2][4]), "Ut4");
    glbSetOscParamByName(p, SQR(sin(2.*th[1][4]) * sin(th[2][4])), "s22thmue");
                        // parentheses here were incorrect. fixed on 06.11.2021 - JK
  }
  else if (n_flavors == 5)
  {
    glbSetOscParamByName(p, sin(th[1][3])*cos(th[1][4])*cos(th[1][5]), "Ue3");
    glbSetOscParamByName(p, sin(th[1][4])*cos(th[1][5]), "Ue4");
    glbSetOscParamByName(p, sin(th[1][5]), "Ue5");

    double A = sin(th[2][4])*cos(th[1][4])*cos(th[2][5]);
    double B = sin(th[1][4])*sin(th[1][5])*sin(th[2][5]);
    glbSetOscParamByName(p, sqrt(SQR(A - B*cos(delta[2] - delta[1]))
                               + SQR(B*sin(delta[2] - delta[1]))), "Um4");
    glbSetOscParamByName(p, sin(th[2][5])*cos(th[1][5]), "Um5");
    glbSetOscParamByName(p, glbGetOscParamByName(p,"Ue5")*glbGetOscParamByName(p,"Um5"),
                         "Ue5Um5");
    glbSetOscParamByName(p, glbGetOscParamByName(p,"Ue4")*glbGetOscParamByName(p,"Um4"),
                         "Ue4Um4");
  }

  // Extra parameters for oscillation + decay scenario
#ifdef NU_USE_NUSQUIDS
  snu_get_oscillation_parameters_osc_decay_internal(n_flavors, p);
#endif

  return 0;
}


// ----------------------------------------------------------------------------
gsl_matrix_complex *snu_get_U()
// ----------------------------------------------------------------------------
// Returns a pointer to the mixing matrix
// ----------------------------------------------------------------------------
{
  return U;
}


// ----------------------------------------------------------------------------
int snu_hamiltonian_cd(double E, double rho, double Ne, int cp_sign)
// ----------------------------------------------------------------------------
// Calculates the Hamiltonian for neutrinos (cp_sign=1) or antineutrinos
// (cp_sign=-1) with energy E, propagating in matter of density rho
// (> 0 even for antineutrinos) and stores the result in H. Ne is the
// electron/proton fraction in matter (1 for solar matter, about 0.5 for Earth
// matter)
// ----------------------------------------------------------------------------
{
  double inv_E = 1.0 / E;
  double Ve = cp_sign * rho * (GLB_V_FACTOR * Ne); // Matter potential
  double Vn = cp_sign * rho * (GLB_V_FACTOR * (1.0 - Ne) / 2.0);

  double complex (*_H)[n_flavors]
    = (double complex (*)[n_flavors]) gsl_matrix_complex_ptr(H, 0, 0);
  double complex (*_H0_template)[n_flavors]
    = (double complex (*)[n_flavors]) gsl_matrix_complex_ptr(H0_template, 0, 0);
  int i, j;

  if (cp_sign > 0)
  {
    for (i=0; i < n_flavors; i++)
      for (j=0; j < n_flavors; j++)
        _H[i][j] = _H0_template[i][j] * inv_E  +  Ve*epsilon_m[i][j];
  }
  else
  {
    for (i=0; i < n_flavors; i++)
      for (j=0; j < n_flavors; j++)
        _H[i][j] = conj(_H0_template[i][j] * inv_E  +  Ve*epsilon_m[i][j]);
                                                // delta_CP -> -delta_CP
  }

// _H[i][j] = _H[i][j] + epsilon_m[i][j]
// _H[j][i] = _H[j][i] + conj(epsilon_m[i][j]);
//                                   COMPLEX!
//  for anti-neutrinos:
// _H[i][j] = _H[i][j] - conj(epsilon_m[i][j]);
// _H[j][i] = _H[j][i] - epsilon_m[i][j];
 
  // Add standard matter potential \sqrt{2} G_F (N_e - N_n/2) for \nu_e and
  // - \sqrt{2} G_F N_n / 2 for \nu_\mu and \nu_\tau
  _H[0][0] = _H[0][0] + Ve - Vn;
  _H[1][1] = _H[1][1] - Vn;
  _H[2][2] = _H[2][2] - Vn;

  return 0;
}


// ----------------------------------------------------------------------------
int snu_S_matrix_cd(double E, double L, double rho, int cp_sign, void *user_data)
// ----------------------------------------------------------------------------
// Calculates the S matrix for neutrino oscillations in matter of constant
// density.
// ----------------------------------------------------------------------------
// Parameters:
//   E: Neutrino energy
//   L: Baseline
//   rho: Matter density (must be > 0 even for antineutrinos)
//   cp_sign: +1 for neutrinos, -1 for antineutrinos
//   user_data: User-defined parameters (used for instance to tell the
//            probability engine which experiment it is being run for)
// ----------------------------------------------------------------------------
{
  // Introduce some abbreviations
  double complex (*_S)[n_flavors] =(double complex (*)[n_flavors])gsl_matrix_complex_ptr(S,0,0);
  double complex (*_Q)[n_flavors] =(double complex (*)[n_flavors])gsl_matrix_complex_ptr(Q,0,0);
  double complex (*_T0)[n_flavors]=(double complex (*)[n_flavors])gsl_matrix_complex_ptr(T0,0,0);
  double *_lambda = gsl_vector_ptr(lambda,0);
  int status;
  int i, j, k;
  
  if (fabs(rho) < RHO_THRESHOLD)                   // Vacuum
  {
    // Use vacuum mixing angles and masses
    double inv_E = 0.5/E;
    _lambda[0] = 0.0;
    for (i=1; i < n_flavors; i++)
      _lambda[i] = dmsq[i-1] * inv_E;

    if (cp_sign > 0)
      gsl_matrix_complex_memcpy(Q, U);
    else
    {
      double complex (*_U)[n_flavors]
        = (double complex (*)[n_flavors]) gsl_matrix_complex_ptr(U,0,0);
      for (i=0; i < n_flavors; i++)
        for (j=0; j < n_flavors; j++)
          _Q[i][j] = conj(_U[i][j]);
    }
  }
  else                                             // Matter
  {
    // Calculate neutrino Hamiltonian
    if ((status=snu_hamiltonian_cd(E, rho, GLB_Ne_MANTLE, cp_sign)) != 0)
      return status;
    
    // Calculate eigenvalues of Hamiltonian
    if (n_flavors == 3)
    {
      double complex (*_H)[3] = (double complex (*)[3]) gsl_matrix_complex_ptr(H,0,0);
      double complex (*_Q)[3] = (double complex (*)[3]) gsl_matrix_complex_ptr(Q,0,0);
      double *_lambda = gsl_vector_ptr(lambda,0);
      if ((status=zheevh3(_H, _Q, _lambda)) != 0)
        return status;
    }
    else
    {
      if ((status=gsl_eigen_hermv(H, lambda, Q, w)) != GSL_SUCCESS)
        return status;
    }
  }

  // Calculate S-Matrix in mass basis in matter ...
  double phase;
  gsl_matrix_complex_set_zero(S);
  for (i=0; i < n_flavors; i++)
  {
    phase    = -L * _lambda[i];
    _S[i][i] = cos(phase) + I*sin(phase);
  } 
  
  // ... and transform it to the flavour basis
  gsl_matrix_complex_set_zero(T0);
  double complex *p = &_T0[0][0];
  for (i=0; i < n_flavors; i++)              // T0 = S.Q^\dagger
    for (j=0; j < n_flavors; j++)
    {
      for (int k=0; k < n_flavors; k++)
      {
        *p += ( creal(_S[i][k])*creal(_Q[j][k])+cimag(_S[i][k])*cimag(_Q[j][k]) )
                + I * ( cimag(_S[i][k])*creal(_Q[j][k])-creal(_S[i][k])*cimag(_Q[j][k]) );
      }
      p++;
    }
  gsl_matrix_complex_set_zero(S);
  p = &_S[0][0];
  for (i=0; i < n_flavors; i++)              // S = Q.T0
    for (j=0; j < n_flavors; j++)
    {
      for (k=0; k < n_flavors; k++)
      {
        *p += ( creal(_Q[i][k])*creal(_T0[k][j])-cimag(_Q[i][k])*cimag(_T0[k][j]) )
                + I * ( cimag(_Q[i][k])*creal(_T0[k][j])+creal(_Q[i][k])*cimag(_T0[k][j]) );
      }
      p++;
    }

  // Incorporate non-standard interactions in the source and in the detector
  if (cp_sign > 0)
  {
    gsl_matrix_complex_set_zero(T0);
    for (i=0; i < n_flavors; i++)            // T0 = S.(1+epsilon_s)
      for (j=0; j < n_flavors; j++)
        for (k=0; k < n_flavors; k++)
          _T0[i][j] += _S[i][k] * epsilon_s_plus_1[k][j];
    gsl_matrix_complex_set_zero(S);
    for (i=0; i < n_flavors; i++)            // S = (1+epsilon_d).T0
      for (j=0; j < n_flavors; j++)
        for (k=0; k < n_flavors; k++)
          _S[i][j] += epsilon_d_plus_1[i][k] * _T0[k][j];
  }
  else
  {
    gsl_matrix_complex_set_zero(T0);
    for (i=0; i < n_flavors; i++)            // T0 = S.conj(1+epsilon_s)
      for (j=0; j < n_flavors; j++)
        for (k=0; k < n_flavors; k++)
          _T0[i][j] += _S[i][k] * conj(epsilon_s_plus_1[k][j]);
    gsl_matrix_complex_set_zero(S);
    for (i=0; i < n_flavors; i++)            // S = conj(1+epsilon_d).T0
      for (j=0; j < n_flavors; j++)
        for (k=0; k < n_flavors; k++)
          _S[i][j] += conj(epsilon_d_plus_1[i][k]) * _T0[k][j];
  }

// S --> epsilon_d_plus_1 . S . epsilon_s_plus_1
// for anti-nu: S --> epsilon_d_plus_1^* . S . epsilon_s_plus_1^*

  return 0;
}


// ----------------------------------------------------------------------------
int snu_filtered_probability_matrix_cd(double P[SNU_MAX_FLAVORS][SNU_MAX_FLAVORS],
        double E, double L, double rho, double sigma, int cp_sign, void *user_data)
// ----------------------------------------------------------------------------
// Calculates the probability matrix for neutrino oscillations in matter
// of constant density, including a low pass filter to suppress aliasing
// due to very fast oscillations.
// ----------------------------------------------------------------------------
// Parameters:
//   P: Storage buffer for the probability matrix
//   E: Neutrino energy (in eV)
//   L: Baseline (in eV^-1)
//   rho: Matter density (must be > 0 even for antineutrinos)
//   sigma: Width of Gaussian filter (in GeV)
//   cp_sign: +1 for neutrinos, -1 for antineutrinos
//   user_data: User-defined parameters (used for instance to tell the
//            probability engine which experiment it is being run for)
// ----------------------------------------------------------------------------
{
  // Introduce some abbreviations
  double complex (*_Q)[n_flavors]  = (double complex (*)[n_flavors]) gsl_matrix_complex_ptr(Q,0,0);
  double complex (*_T0)[n_flavors] = (double complex (*)[n_flavors]) gsl_matrix_complex_ptr(T0,0,0);
  double complex (*_Q1)[n_flavors] = (double complex (*)[n_flavors]) gsl_matrix_complex_ptr(Q1,0,0);
  double complex (*_Q2)[n_flavors] = (double complex (*)[n_flavors]) gsl_matrix_complex_ptr(Q2,0,0);
  double *_lambda = gsl_vector_ptr(lambda,0);
  int status;
  int i, j, k, l;

  // Special treatment of MINOS near detector, where additional smearing is introduced to
  // emulate the effect of the non-vanishing length of the decay pip 
  if (user_data != NULL  &&  *(int *)user_data == MINOS_ND_PROBABILITY)
  {
    sigma = sqrt(SQR(sigma) + SQR(-3.0 / (E/1.e9) + 2.0));
//    printf("MINOS Probability! L=%g, E=%g, sigma=%g", L, E/1.e9, sigma);
//    getchar();
  }

  // Vacuum: Use vacuum mixing angles and masses
  if (fabs(rho) < RHO_THRESHOLD)
  {
    double inv_E = 0.5/E;
    _lambda[0] = 0.0;
    for (i=1; i < n_flavors; i++)
      _lambda[i] = dmsq[i-1] * inv_E;

    if (cp_sign > 0)
      gsl_matrix_complex_memcpy(Q, U);
    else
    {
      double complex (*_U)[n_flavors]
        = (double complex (*)[n_flavors]) gsl_matrix_complex_ptr(U,0,0);
      for (i=0; i < n_flavors; i++)
        for (j=0; j < n_flavors; j++)
          _Q[i][j] = conj(_U[i][j]);
    }
  }

  // Matter: Rediagonalize Hamiltonian
  else
  {
    // Calculate neutrino Hamiltonian
    if ((status=snu_hamiltonian_cd(E, rho, GLB_Ne_MANTLE, cp_sign)) != 0)
      return status;
    
    // Calculate eigenvalues and eigenvectors of Hamiltonian
    if (n_flavors == 3)
    {
      double complex (*_H)[3] = (double complex (*)[3]) gsl_matrix_complex_ptr(H,0,0);
      double complex (*_Q)[3] = (double complex (*)[3]) gsl_matrix_complex_ptr(Q,0,0);
      double *_lambda = gsl_vector_ptr(lambda,0);
      if ((status=zheevh3(_H, _Q, _lambda)) != 0)
        return status;
    }
    else
    {
      if ((status=gsl_eigen_hermv(H, lambda, Q, w)) != GSL_SUCCESS)
        return status;
    }
  }

  // Define Q_1^\dag = Q^\dag . (1 + \eps^s) and Q_2 = (1 + \eps^d) . Q
  // (for anti-neutrinos: \eps^{s,d} -> (\eps^{s,d})^*
  gsl_matrix_complex_set_zero(Q1);
  gsl_matrix_complex_set_zero(Q2);
  if (cp_sign > 0)
  {
    for (i=0; i < n_flavors; i++)
      for (j=0; j < n_flavors; j++)
        for (k=0; k < n_flavors; k++)
          _Q1[i][j] += conj(epsilon_s_plus_1[k][i]) * _Q[k][j];
    for (i=0; i < n_flavors; i++)
      for (j=0; j < n_flavors; j++)
        for (k=0; k < n_flavors; k++)
          _Q2[i][j] += epsilon_d_plus_1[i][k] * _Q[k][j];
  }
  else
  {
    for (i=0; i < n_flavors; i++)
      for (j=0; j < n_flavors; j++)
        for (k=0; k < n_flavors; k++)
          _Q1[i][j] += epsilon_s_plus_1[k][i] * _Q[k][j];
    for (i=0; i < n_flavors; i++)
      for (j=0; j < n_flavors; j++)
        for (k=0; k < n_flavors; k++)
          _Q2[i][j] += conj(epsilon_d_plus_1[i][k]) * _Q[k][j];
  }
        

  // Calculate probability matrix (see GLoBES manual for a discussion of the algorithm)
  double phase, filter_factor;
  double t = -0.5/1.0e-18 * SQR(sigma) / SQR(E);
  gsl_matrix_complex_set_zero(T0);
  for (i=0; i < n_flavors; i++)
    for (j=i+1; j < n_flavors; j++)
    {
      phase         = -L * (_lambda[i] - _lambda[j]);
      filter_factor = exp(t * SQR(phase));
      _T0[i][j]     = filter_factor * (cos(phase) + I*sin(phase));
    }

  for (k=0; k < n_flavors; k++)
    for (l=0; l < n_flavors; l++)
    {
      P[k][l] = 0.0;
      for (i=0; i < n_flavors; i++)
      {
        complex t = conj(_Q1[k][i]) * _Q2[l][i];
        for (j=i+1; j < n_flavors; j++)
          P[k][l] += 2.0 * creal(_Q1[k][j] * conj(_Q2[l][j]) * t * _T0[i][j]);
        P[k][l] += SQR_ABS(_Q1[k][i]) * SQR_ABS(_Q2[l][i]);
      }
    }
    
  return 0;
}


// ----------------------------------------------------------------------------
int snu_probability_matrix(double _P[3][3], int cp_sign, double E,
    int psteps, const double *length, const double *density,
    double filter_sigma, void *user_data)
// ----------------------------------------------------------------------------
// Calculates the neutrino oscillation probability matrix for use by GLoBES.
// The problem is that GLoBES expects P to be a 3x3 matrix, so we compute the
// full matrix and then extract the upper left 3x3 submatrix.
// ----------------------------------------------------------------------------
{
  double P[SNU_MAX_FLAVORS][SNU_MAX_FLAVORS];
  int status;
  int i, j;

  status = snu_probability_matrix_all(P, cp_sign, E, psteps, length, density,
                                      filter_sigma, user_data);
  for (i=0; i < 3; i++)
    for (j=0; j < 3; j++)
      _P[j][i] = P[j][i];

  return status;
}


// ----------------------------------------------------------------------------
int snu_probability_matrix_all(double P[SNU_MAX_FLAVORS][SNU_MAX_FLAVORS], int cp_sign, double E,
    int psteps, const double *length, const double *density,
    double filter_sigma, void *user_data)
// ----------------------------------------------------------------------------
// Calculates the neutrino oscillation probability matrix.
// ----------------------------------------------------------------------------
// Parameters:
//   P:       Buffer for the storage of the matrix
//   cp_sign: +1 for neutrinos, -1 for antineutrinos
//   E:       Neutrino energy (in GeV)
//   psteps:  Number of layers in the matter density profile
//   length:  Lengths of the layers in the matter density profile in km
//   density: The matter densities in g/cm^3
//   filter_sigma: Width of low-pass filter or <0 for no filter
//   user_data: User-defined parameters (used for instance to tell the
//            probability engine which experiment it is being run for)
// ----------------------------------------------------------------------------
{
  // --------------------------------------------------------------------
  // The following is for debugging the 3x3 matrix diagonalization routines
//  {
//    double complex A[3][3] = {{-0.00820883,  0.01365408, -0.00603976},
//                              { 0.01365408, -0.01874123, -0.0001346},
//                              {-0.00603976, -0.0001346,  -0.00294713}};
//    double w[3];
//    zheevc3(A, w);
//    printf("%g %g %g\n", w[0], w[1], w[2]);
//  }
//  {
//    double complex A[3][3] = {{-0.00820883,  0.01365408, -0.00603976},
//                              { 0.01365408, -0.01874123, -0.0001346},
//                              {-0.00603976, -0.0001346,  -0.00294713}};
//    double complex Q[3][3];
//    double w[3];
//    zheevq3(A, Q, w);
//    printf("%g %g %g\n", w[0], w[1], w[2]);
//    exit(1);
//  }

  int status;
  int i, j;

  // Convert energy to eV
  E *= 1.0e9;
  
  if (filter_sigma > 0.0)                     // With low-pass filter
  {
    if (psteps == 1)
      snu_filtered_probability_matrix_cd(P, E, GLB_KM_TO_EV(length[0]),
                                         density[0], filter_sigma, cp_sign, user_data);
    else
      return -1;
  }
  else                                        // Without low-pass filter
  {
    if (psteps > 1)
    {
      gsl_matrix_complex_set_identity(S1);                                 // S1 = 1
      for (i=0; i < psteps; i++)
      {
        status = snu_S_matrix_cd(E, GLB_KM_TO_EV(length[i]), density[i], cp_sign, user_data);
        if (status != 0)
          return status;
        gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, GSL_COMPLEX_ONE, S, S1, // T0 = S.S1
                       GSL_COMPLEX_ZERO, T0);
        gsl_matrix_complex_memcpy(S1, T0);                                 // S1 = T0
      } 
      gsl_matrix_complex_memcpy(S, S1);                                    // S = S1
    }
    else
    {
      status = snu_S_matrix_cd(E, GLB_KM_TO_EV(length[0]), density[0], cp_sign, user_data);
      if (status != 0)
        return status;
    }

    double complex (*_S)[n_flavors]
      = (double complex (*)[n_flavors]) gsl_matrix_complex_ptr(S,0,0);
    for (i=0; i < n_flavors; i++)
      for (j=0; j < n_flavors; j++)
        P[j][i] = SQR_ABS(_S[i][j]);
  }

  return 0;
}


// ----------------------------------------------------------------------------
int snu_probability_matrix_m_to_f(double P[SNU_MAX_FLAVORS][SNU_MAX_FLAVORS], int cp_sign,
    double E, int psteps, const double *length, const double *density,
    double filter_sigma, void *user_data)
// ----------------------------------------------------------------------------
// Calculates the neutrino oscillation probability matrix, assuming that the
// initial state is a vacuum _mass_ eigenstate (e.g. for solar neutrinos)
// ----------------------------------------------------------------------------
// Parameters:
//   P:       Buffer for the storage of the matrix
//   cp_sign: +1 for neutrinos, -1 for antineutrinos
//   E:       Neutrino energy (in GeV)
//   psteps:  Number of layers in the matter density profile
//   length:  Lengths of the layers in the matter density profile in km
//   density: The matter densities in g/cm^3
//   filter_sigma: Width of low-pass filter or <0 for no filter
//   user_data: User-defined parameters (used for instance to tell the
//            probability engine which experiment it is being run for)
// ----------------------------------------------------------------------------
{
  int status;
  int i, j;

  // Convert energy to eV
  E *= 1.0e9;
  
  if (filter_sigma > 0.0)                     // With low-pass filter
  {
    if (psteps == 1)
      snu_filtered_probability_matrix_m_to_f(P, E, GLB_KM_TO_EV(length[0]),
                                             density[0], filter_sigma, cp_sign, user_data);
    else
    {
      fprintf(stderr, "ERROR: Filter feature not implemented for non-constant density\n");
      memset(P, 0, SNU_MAX_FLAVORS*SNU_MAX_FLAVORS*sizeof(P[0][0]));
      return -1;
    }
  }
  else                                        // Without low-pass filter
  {
    if (psteps > 1)
    {
      gsl_matrix_complex_set_identity(S1);                                 // S1 = 1
      for (i=0; i < psteps; i++)
      {
        status = snu_S_matrix_cd(E, GLB_KM_TO_EV(length[i]), density[i], cp_sign, user_data);
        if (status != 0)
          return status;
        gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, GSL_COMPLEX_ONE, S, S1, // T0 = S.S1
                       GSL_COMPLEX_ZERO, T0);
        gsl_matrix_complex_memcpy(S1, T0);                                 // S1 = T0
      } 
      gsl_matrix_complex_memcpy(S, S1);                                    // S  = S1
    }
    else
    {
      status = snu_S_matrix_cd(E, GLB_KM_TO_EV(length[0]), density[0], cp_sign, user_data);
      if (status != 0)
        return status;
    }

    // Convert initial states from mass to flavor basis
    if (cp_sign > 0)
    {
      gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, GSL_COMPLEX_ONE, S, U,    // S1 = S.U
                     GSL_COMPLEX_ZERO, S1);
      double complex (*_S1)[n_flavors]
        = (double complex (*)[n_flavors]) gsl_matrix_complex_ptr(S1,0,0);
      for (i=0; i < n_flavors; i++)
        for (j=0; j < n_flavors; j++)
          P[j][i] = SQR_ABS(_S1[i][j]);
    }
    else
    {
      gsl_blas_zgemm(CblasConjTrans, CblasTrans, GSL_COMPLEX_ONE, U, S,    // S1 = (S.U*)^T
                     GSL_COMPLEX_ZERO, S1);
      double complex (*_S1)[n_flavors]
        = (double complex (*)[n_flavors]) gsl_matrix_complex_ptr(S1,0,0);
      for (i=0; i < n_flavors; i++)
        for (j=0; j < n_flavors; j++)
          P[j][i] = SQR_ABS(_S1[j][i]);                                    // Un-transpose!
    }
  }

  return 0;
}


// ----------------------------------------------------------------------------
int snu_filtered_probability_matrix_m_to_f(double P[SNU_MAX_FLAVORS][SNU_MAX_FLAVORS],
        double E, double L, double rho, double sigma, int cp_sign, void *user_data)
// ----------------------------------------------------------------------------
// Calculates the probability matrix for neutrino oscillations in matter
// of constant density, assuming that the initial state is a vacuum _mass_
// eigenstate (e.g. for solar neutrinos) and including a low pass filter to
// suppress aliasing due to very fast oscillations.
// NOTE: The implementation is not yet as efficient as it could be (it is
// analogous to snu_filtered_probability_matrix_cd, omitting possible
// optimizations for mass -> flavor oscillations)
// WARNING: Source NSI are IGNORED by this function
// ----------------------------------------------------------------------------
// Parameters:
//   P: Storage buffer for the probability matrix
//   E: Neutrino energy (in GeV)
//   L: Baseline
//   rho: Matter density (must be > 0 even for antineutrinos)
//   sigma: Width of Gaussian filter (in GeV)
//   cp_sign: +1 for neutrinos, -1 for antineutrinos
//   user_data: User-defined parameters (used for instance to tell the
//            probability engine which experiment it is being run for)
// ----------------------------------------------------------------------------
{
  // Introduce some abbreviations
  double complex (*_U)[n_flavors]  = (double complex (*)[n_flavors]) gsl_matrix_complex_ptr(U,0,0);
  double complex (*_Q)[n_flavors]  = (double complex (*)[n_flavors]) gsl_matrix_complex_ptr(Q,0,0);
  double complex (*_T0)[n_flavors] = (double complex (*)[n_flavors]) gsl_matrix_complex_ptr(T0,0,0);
  double complex (*_Q1)[n_flavors] = (double complex (*)[n_flavors]) gsl_matrix_complex_ptr(Q1,0,0);
  double complex (*_Q2)[n_flavors] = (double complex (*)[n_flavors]) gsl_matrix_complex_ptr(Q2,0,0);
  double *_lambda = gsl_vector_ptr(lambda,0);
  int status;
  int i, j, k, l;

  // Vacuum: Use vacuum mixing angles and masses
  if (fabs(rho) < RHO_THRESHOLD)
  {
    double inv_E = 0.5/E;
    _lambda[0] = 0.0;
    for (i=1; i < n_flavors; i++)
      _lambda[i] = dmsq[i-1] * inv_E;

    if (cp_sign > 0)
      gsl_matrix_complex_memcpy(Q, U);
    else
    {
      for (i=0; i < n_flavors; i++)
        for (j=0; j < n_flavors; j++)
          _Q[i][j] = conj(_U[i][j]);
    }
  }

  // Matter: Rediagonalize Hamiltonian
  else
  {
    // Calculate neutrino Hamiltonian
    if ((status=snu_hamiltonian_cd(E, rho, GLB_Ne_MANTLE, cp_sign)) != 0)
      return status;
    
    // Calculate eigenvalues and eigenvectors of Hamiltonian
    if (n_flavors == 3)
    {
      double complex (*_H)[3] = (double complex (*)[3]) gsl_matrix_complex_ptr(H,0,0);
      double complex (*_Q)[3] = (double complex (*)[3]) gsl_matrix_complex_ptr(Q,0,0);
      double *_lambda = gsl_vector_ptr(lambda,0);
      if ((status=zheevh3(_H, _Q, _lambda)) != 0)
        return status;
    }
    else
    {
      if ((status=gsl_eigen_hermv(H, lambda, Q, w)) != GSL_SUCCESS)
        return status;
    }
  }

  // Define Q_1^\dag = Q^\dag . (1 + \eps^s) and Q_2 = (1 + \eps^d) . Q
  // (for anti-neutrinos: \eps^{s,d} -> (\eps^{s,d})^*
  gsl_matrix_complex_set_zero(Q1);
  gsl_matrix_complex_set_zero(Q2);
  if (cp_sign > 0)
  {
    for (i=0; i < n_flavors; i++)   // Convert initial states from mass to flavor basis
      for (j=0; j < n_flavors; j++)
        for (k=0; k < n_flavors; k++)
          _Q1[i][j] += conj(_U[k][i]) * _Q[k][j];
    for (i=0; i < n_flavors; i++)
      for (j=0; j < n_flavors; j++)
        for (k=0; k < n_flavors; k++)
          _Q2[i][j] += epsilon_d_plus_1[i][k] * _Q[k][j];
  }
  else
  {
    for (i=0; i < n_flavors; i++)
      for (j=0; j < n_flavors; j++)
        for (k=0; k < n_flavors; k++)
          _Q1[i][j] += _U[k][i] * _Q[k][j];
    for (i=0; i < n_flavors; i++)
      for (j=0; j < n_flavors; j++)
        for (k=0; k < n_flavors; k++)
          _Q2[i][j] += conj(epsilon_d_plus_1[i][k]) * _Q[k][j];
  }
        

  // Calculate probability matrix (see GLoBES manual for a discussion of the algorithm)
  double phase, filter_factor;
  double t = -0.5/1.0e-18 * SQR(sigma) / SQR(E);
  gsl_matrix_complex_set_zero(T0);
  for (i=0; i < n_flavors; i++)
    for (j=i+1; j < n_flavors; j++)
    {
      phase         = -L * (_lambda[i] - _lambda[j]);
      filter_factor = exp(t * SQR(phase));
      _T0[i][j]     = filter_factor * (cos(phase) + I*sin(phase));
    }

  for (k=0; k < n_flavors; k++)
    for (l=0; l < n_flavors; l++)
    {
      P[k][l] = 0.0;
      for (i=0; i < n_flavors; i++)
      {
        complex t = conj(_Q1[k][i]) * _Q2[l][i];
        for (j=i+1; j < n_flavors; j++)
          P[k][l] += 2.0 * creal(_Q1[k][j] * conj(_Q2[l][j]) * t * _T0[i][j]);
        P[k][l] += SQR_ABS(_Q1[k][i]) * SQR_ABS(_Q2[l][i]);
      }
    }
    
  return 0;
}


// ----------------------------------------------------------------------------
//                    N U S Q U I D S   I N T E R F A C E
// ----------------------------------------------------------------------------

#ifdef NU_USE_NUSQUIDS

#include "glb_error.h"

static double *nusquids_spectrum = NULL;
glb_probability_nusquids_function nu_hook_probability_matrix_nusquids
                                        = snu_probability_matrix_osc_decay;

// ----------------------------------------------------------------------------
int snu_probability_matrix_nusquids(double P[][2][3],
      unsigned n_E, double *E, double ini_state_nu[][3], double ini_state_nubar[][3],
      int psteps, const double *length, const double *density, const double filter_value)
// ----------------------------------------------------------------------------
// Calculates the neutrino oscillation probability matrix using nuSQuIDS. As
// nuSQuIDS simulates also processes like tau regeneration or neutrino decay,
// there can be migration between energy bins, hence the function computes
// the probability matrix for a whole range of energy bins at once.
// ----------------------------------------------------------------------------
// Parameters:
//   P:         Buffer for the storage of the final weights. The indices of P
//              correspond to energy, cp sign (nu/nubar), final flavor.
//   cp_sign:   +1 for neutrinos, -1 for antineutrinos
//   n_E:       number of energy bins
//   E:         list of neutrino energies (in GeV)
//   ini_state_nu/nubar: initial state for neutrinos/antineutrinos. Each entry
//              is a list of length n_E specifying the unoscillated energy
//              spectrum of one neutrino flavor
//   psteps:    Number of layers in the matter density profile
//   length:    Lengths of the layers in the matter density profile in km
//   density:   The matter densities in g/cm^3
//   filter_value: width of low-pass filter for smoothing fast oscillations
//              *** currently ignored by NuSQuIDS ***
// ----------------------------------------------------------------------------
{
  int status;

  // compute n-flavor probabilities matrix
  double P_full[n_E][2][SNU_MAX_FLAVORS];
  status = snu_probability_matrix_nusquids_internal(P_full, n_E, E,
             ini_state_nu, ini_state_nubar, psteps, length, density,
             n_flavors, th, delta, dmsq, M_A_prime, g_prime);

  // copy probabilities for the active flavors
  size_t n_bytes = sizeof(double) * n_E * 2 * SNU_MAX_FLAVORS;
  for (unsigned k=0; k < n_E; k++)
    for (unsigned j=0; j < 3; j++)
    {
      P[k][0][j] = P_full[k][0][j];
      P[k][1][j] = P_full[k][1][j];
    }
 

  // save computed spectrum for later use
  if (nusquids_spectrum)
    nusquids_spectrum = glb_realloc(nusquids_spectrum, n_bytes);
  else
    nusquids_spectrum = glb_malloc(n_bytes);
  memcpy(nusquids_spectrum, P_full, n_bytes);

  return status;
}


// ----------------------------------------------------------------------------
int snu_nusquids_rates(double _R[3], int cp_sign, int bin_number)
// ----------------------------------------------------------------------------
// Retrieves the oscillated rates from a previous nuSQuIDS run, i.e.
// from a previous call to snu_probability_matrix_nusquids.
// ----------------------------------------------------------------------------
// Parameters:
//   R:          Storage buffer for the rates for the three active flavors
//   cp_sign:    +1 for neutrinos, -1 for antineutrinos
//   bin_number: the index of the energy bin for which P should be returned
// ----------------------------------------------------------------------------
{
  double (*P)[][2][SNU_MAX_FLAVORS] = (double (*)[][2][SNU_MAX_FLAVORS]) nusquids_spectrum;
  if (!nusquids_spectrum)
    return -1;

  if (cp_sign > 0)
    for (int i=0; i < 3; i++)
      _R[i] = (*P)[bin_number][0][i];

  return 0;
}


// ----------------------------------------------------------------------------
int snu_probability_matrix_osc_decay(double P[][2][3],
      unsigned n_E, double *E, double ini_state_nu[][3], double ini_state_nubar[][3],
      int psteps, const double *length, const double *density, const double filter_value)
// ----------------------------------------------------------------------------
// Calculates the neutrino oscillation probability matrix using Ivan Esteban's
// analytic treatment of oscillations + decay.  As there can be migration between
// energy bins, the function computes the probability matrix for a whole range
// of energy bins at once.
// ----------------------------------------------------------------------------
// Parameters:
//   P:         Buffer for the storage of the final weights. The indices of P
//              correspond to energy, cp sign (nu/nubar), final flavor.
//   cp_sign:   +1 for neutrinos, -1 for antineutrinos
//   n_E:       number of energy bins
//   E:         list of neutrino energies (in GeV)
//   ini_state_nu/nubar: initial state for neutrinos/antineutrinos. Each entry
//              is a list of length n_E specifying the unoscillated energy
//              spectrum of one neutrino flavor
//   psteps:    Number of layers in the matter density profile
//   length:    Lengths of the layers in the matter density profile in km
//   density:   The matter densities in g/cm^3
//   filter_value: width of low-pass filter for smoothing fast oscillations
//              (implemented as in GLoBES, see GLoBES manual for details)//
// ----------------------------------------------------------------------------
{
  int status;

  // compute n-flavor probabilities matrix
  double P_full[n_E][2][SNU_MAX_FLAVORS];
  status = snu_probability_matrix_osc_decay_internal(P_full, n_E, E,
             ini_state_nu, ini_state_nubar, psteps, length, density, n_flavors, filter_value);

  // copy probabilities for the active flavors
  size_t n_bytes = sizeof(double) * n_E * 2 * SNU_MAX_FLAVORS;
  for (unsigned k=0; k < n_E; k++)
    for (unsigned j=0; j < 3; j++)
    {
      P[k][0][j] = P_full[k][0][j];
      P[k][1][j] = P_full[k][1][j];
    }

  // save computed spectrum for later use
  if (nusquids_spectrum)
    nusquids_spectrum = glb_realloc(nusquids_spectrum, n_bytes);
  else
    nusquids_spectrum = glb_malloc(n_bytes);
  memcpy(nusquids_spectrum, P_full, n_bytes);

  return status;
}

#endif // NU_USE_NUSQUIDS


// ----------------------------------------------------------------------------
//   U S I N G   F O R   P R E - C O M P U T E D   P R O B A B I L I T I E S
// ----------------------------------------------------------------------------

#include "glb_error.h"
#include "glb_probability.h"
#include "nu.h"

// -------------------------------------------------------------------------
struct snu_probability_table *snu_alloc_probability_table()
// -------------------------------------------------------------------------
// Allocates a new data structure for holding a probability table
// -------------------------------------------------------------------------
{
  struct snu_probability_table *p = glb_malloc(sizeof(struct snu_probability_table));
  if (!p)
  {
    glb_error("snu_alloc_probability_table: cannot allocate probability table.\n");
    return NULL;
  }

  p->default_values = glbAllocParams();
  p->probability_table = NULL;
  p->n_p = 0;
  for (int i=0; i < SNU_MAX_PARAMS+1; i++)
  {
    p->params[i]  = NULL;
    p->p_min[i]   = NAN;
    p->p_max[i]   = NAN;
    p->p_steps[i] = -1;
    p->p_flags[i] = 0;
  }

  return p;
}


// -------------------------------------------------------------------------
int snu_free_probability_table(struct snu_probability_table *p)
// -------------------------------------------------------------------------
// Frees memory associated with a probability table
// -------------------------------------------------------------------------
{
  if (p)
  {
    if (p->default_values) { glbFreeParams(p->default_values);  p->default_values=NULL; }
    if (p->probability_table) { glb_free(p->probability_table); p->probability_table=NULL; }
    p->n_p = 0;
    for (int i=0; i < SNU_MAX_PARAMS+1; i++)
    {
      if (p->params[i])  { glb_free(p->params[i]);  p->params[i]  = NULL; }
      p->p_min[i]   = NAN;
      p->p_max[i]   = NAN;
      p->p_steps[i] = -1;
      p->p_flags[i] = 0;
    }
  }

  return GLB_SUCCESS;
}


// -------------------------------------------------------------------------
int snu_compute_probability_table(int experiment, struct snu_probability_table *p,
                                  const char *output_file)
// -------------------------------------------------------------------------
// Compute a probability table for the given experiment, based on the
// parameters in p that must have been filled in already. If an output file
// is given, the table is immediately written to disk.
// -------------------------------------------------------------------------
{
  struct glb_experiment *e = glb_experiment_list[experiment];

  if (!p)
  {
    glb_error("snu_compute_probability_table: NULL input");
    return GLBERR_INVALID_ARGS;
  }
  if (!p->default_values)
  {
    glb_error("snu_compute_probability_table: p->default_values not given");
    return GLBERR_INVALID_ARGS;
  }
  if (p->n_p <= 0)
  {
    glb_error("snu_compute_probability_table: missing scan specification (n_p=0)");
    return GLBERR_INVALID_ARGS;
  }
  if (experiment < 0  ||  experiment >= glb_num_of_exps)
  {
    glb_error("snu_compute_probability_table: invalid experiment number: %d", experiment);
    return GLBERR_INVALID_ARGS;
  }

  // Check for invalid oscillation parameter names
  for (int i=0; i < p->n_p; i++)
  {
    if (!p->params[i])
    {
      glb_error("snu_compute_probability_table: scan parameter #%d not given", i);
      return GLBERR_INVALID_ARGS;
    }
    if (glbFindParamByName(p->params[i]) < 0)
    {
      glb_error("snu_compute_probability_table: Invalid oscillation parameter: %s.\n",
                p->params[i]);
      return GLBERR_INVALID_ARGS;
    }
    if (p->p_steps[i] <= 0)
    {
      glb_error("snu_compute_probability_table: invalid input p_steps=%d for "
                "parameter #%d", p->p_steps[i], i);
      return GLBERR_INVALID_ARGS;
    }
  }

  // Allocate buffer for the actual probability table
  unsigned long n_points = 1;
  for (int i=0; i < p->n_p; i++)
    n_points *= p->p_steps[i] + 1;
  p->probability_table = glb_malloc(sizeof(*p->probability_table) * n_points
               * e->simbins * 18);
  if (!p->probability_table)
  {
    glb_error("snu_compute_probability_table: cannot allocate probability table.");
    return GLBERR_MALLOC_FAILED;
  }

  // Open output file
  FILE *f = NULL;
  if (output_file)
  {
    f = fopen(output_file, "w");
    if (!f)
    {
      glb_error("snu_compute_probability_table: Cannot open file %s", output_file);
      return GLBERR_FILE_NOT_FOUND;
    }

    // Write header consisting of number, names and ranges of parameters
    fprintf(f, "# GLoBES pre-computed probability table\n");
    fprintf(f, "# for experiment %s\n", e->filename);
    fprintf(f, "N_DATA %lu\n", n_points * e->simbins);
    for (int i=0; i < p->n_p; i++)
    {
      fprintf(f, "PARAM %s %10.5g %10.5g %5d %5lu\n",
              p->params[i], p->p_min[i], p->p_max[i], p->p_steps[i], p->p_flags[i]);
    }
  }

  // The main loop
  glb_params test_values = glbAllocParams();
  glbCopyParams(p->default_values, test_values);
  MPIFOR(j, 0, (int) n_points-1)
  {
    double p_test_values[p->n_p];
    for (int i=0; i < p->n_p; i++)
    {
      // Convert 1d index to a multi-dimensional index for the i-th dimension
      int k = n_points;
      int m = j;
      for (int n=0; n <= i; n++)
      {
        m %= k;
        k /= p->p_steps[n] + 1;
      }
      m /= k;

      if (p->p_flags[i] & DEG_LOGSCALE)
      {
        if (p->p_steps[i] == 0)
          p_test_values[i] = POW10(p->p_min[i]);
        else
          p_test_values[i] = POW10(p->p_min[i] + m * (p->p_max[i]-p->p_min[i])/p->p_steps[i]);
      }
      else
      {
        if (p->p_steps[i] == 0)
          p_test_values[i] = p->p_min[i];
        else
          p_test_values[i] = p->p_min[i] + m * (p->p_max[i]-p->p_min[i])/p->p_steps[i];
      }
      glbSetOscParamByName(test_values, p_test_values[i], p->params[i]);
    }

    // For 5-neutrino scenarios, the distinction between 3+2 and 1+3+1 is hardcoded
    // for compatibility with Thomas' code TODO: Find a better solution
    if (n_flavors >= 5)
    {
#ifndef Ip3pI
      glbSetOscParamByName(test_values,  fabs(glbGetOscParamByName(test_values, "DM41")),      "DM41");
      glbSetOscParamByName(test_values,  fabs(glbGetOscParamByName(test_values, "DM51")),      "DM51");
#else
      glbSetOscParamByName(test_values, -fabs(glbGetOscParamByName(test_values, "DM41")),      "DM41");
      glbSetOscParamByName(test_values,  fabs(glbGetOscParamByName(test_values, "DM51")),      "DM51");
#endif
    }

    // Test if GLoBES will accept the chosen oscillation parameters (it may not,
    // for instance if the chosen values of s22thmue and Um4 are inconsistent),
    // and if so, run degfinder
    if (glbSetOscillationParameters(test_values) == 0)
    {
      double filter = (e->filter_state == GLB_ON) ? e->filter_value : -1.0;
      glb_probability_matrix_function probability_matrix = e->probability_matrix
                       ? e->probability_matrix : glb_hook_probability_matrix;
      void *user_data = e->probability_user_data
                       ? e->probability_user_data : glb_probability_user_data;

      for (int k=0; k < e->simbins; k++)
      {
        double E = e->smear_data[0]->simbincenter[k];
        double Pnu[3][3], Pnubar[3][3];
        if (probability_matrix(Pnu, +1, E, e->psteps, e->lengthtab,
                               e->densitytab, filter, user_data) != GLB_SUCCESS)
          glb_error("snu_compute_probability_table: Calculation of osc. probabilities (CP=+1) failed.");
        if (probability_matrix(Pnubar, -1, E, e->psteps, e->lengthtab,
                               e->densitytab, filter, user_data) != GLB_SUCCESS)
          glb_error("snu_compute_probability_table: Calculation of osc. probabilities (CP=-1) failed.");

        memcpy(p->probability_table + j*e->simbins*18 + k*18, Pnu, 9*sizeof(Pnu[0][0]));
        memcpy(p->probability_table + j*e->simbins*18 + k*18 + 9, Pnubar, 9*sizeof(Pnubar[0][0]));
        if (f)
        {
          for (int l=0; l < 9; l++)
            fprintf(f, "%10.5g ", ((double *) Pnu)[l]);
          fprintf(f, "    ");
          for (int l=0; l < 9; l++)
            fprintf(f, "%10.5g ", ((double *) Pnubar)[l]);
          fprintf(f, "\n");
        } // if (f)
      } // for (k=simbins)
    } // if (glbSetOscillationParameters)
  } // MPIFOR(j)

  if (f)  fclose(f);
  glbFreeParams(test_values);
  return GLB_SUCCESS;
}


// ----------------------------------------------------------------------------
int snu_load_probability_table(const char *input_file, struct snu_probability_table *p)
// ----------------------------------------------------------------------------
// Load table of pre-computed probabilities from file. The receiving data
// structure p is assumed to be acllocated but uninitialized; all required
// memory (p->probability_table, p->params) will be allocated.
// ----------------------------------------------------------------------------
{
  if (!p)
  {
    glb_error("snu_load_probability_table: NULL input");
    return GLBERR_INVALID_ARGS;
  }

  // Open input file
  FILE *f = glb_fopen(input_file, "r");
  if (!f)
  {
    glb_error("snu_load_probability_table: Cannot open file %s", input_file);
    return GLBERR_FILE_NOT_FOUND;
  }

  p->n_p = 0;
  if (p->probability_table)
  {
    glb_free(p->probability_table);
    p->probability_table = NULL;
  }

  const int max_line = 1024;  // Maximum length of line
  char this_line[max_line];   // Buffer for current line
  int nl = 0;                 // Number of lines
  int nd = 0;                 // Number of data lines read
  int ne = 0;                 // Number of expected data lines
  while (fgets(this_line, max_line, f))
  {
    nl++;
    if (strlen(this_line) > max_line - 2)
    {
      glb_error("snu_load_probability_table: Line %d too long in file %s",
                nl, input_file);
      fclose(f);
      return GLBERR_INVALID_FILE_FORMAT;
    }

    // Ignore comments and blank l ines
    if (this_line[0] == '#'  ||  this_line[strspn(this_line, " \t")] == '\n')
      continue;                             /* Ignore comments and blank lines */

    // Read parameter declarations
    char this_param[64];
    if (sscanf(this_line, "PARAM %64s %lf %lf %d %lu", this_param,
               &p->p_min[p->n_p], &p->p_max[p->n_p], &p->p_steps[p->n_p],
               &p->p_flags[p->n_p]) == 5)
    {
      if (nd != 0)
      {
        glb_error("snu_load_probability_table: Error in file %s, line %d: "
                  "parameter declarations must precede data lines", nl, input_file);
        fclose(f);
        return GLBERR_INVALID_FILE_FORMAT;
      }
      else
      {
        p->params[p->n_p] = strdup(this_param);
        p->n_p++;
        if (p->n_p > SNU_MAX_PARAMS)
        {
          glb_error("snu_load_probability_table: Error in file %s, line %d: "
                    "too many scan parameters", nl, input_file);
          fclose(f);
          return GLBERR_INVALID_FILE_FORMAT;
        }
      }
    }

    // Read number of parameters
    else if (sscanf(this_line, "N_DATA %d", &ne) == 1)
    {
    }

    // Read probability data
    else
    {
      if (ne <= 0)
      {
        glb_error("snu_load_probability_table: Error in file %s, line %d: "
                  "N_DATA entry missing or invalid", nl, input_file);
        fclose(f);
        return GLBERR_INVALID_FILE_FORMAT;
      }

      // Allocate memory for probability table
      if (!p->probability_table)
        p->probability_table = glb_malloc(sizeof(*p->probability_table) * ne * 18);

      double P[18];
      if (sscanf(this_line, "%lf %lf %lf  %lf %lf %lf  %lf %lf %lf "
                            "%lf %lf %lf  %lf %lf %lf  %lf %lf %lf",
                 &P[0], &P[1],  &P[2],  &P[3],  &P[4],  &P[5],  &P[6],  &P[7],  &P[8],
                 &P[9], &P[10], &P[11], &P[12], &P[13], &P[14], &P[15], &P[16], &P[18]) != 18)
      {
        glb_error("snu_load_probability_table: Error reading probability data "
                  "in file %s, line %d.", nl, input_file);
        fclose(f);
        return GLBERR_INVALID_FILE_FORMAT;
      }
      else
      {
        if (nd >= ne)
        {
          glb_error("snu_load_probability_table: Too many data entries "
                    "in file %s, line %d.", nl, input_file);
          fclose(f);
          return GLBERR_INVALID_FILE_FORMAT;
        }
        memcpy(&p->probability_table[nd++], P, sizeof(*P) * 18);
      } // if (fscanf(probability data))
    } // if (fscanf(meta data)
  } // while (fgets)

  fclose(f);
  return GLB_SUCCESS;  
}


// ----------------------------------------------------------------------------
int snu_tabulated_probability_matrix(double _P[3][3], int cp_sign, double E,
    int psteps, const double *length, const double *density,
    double filter_sigma, void *user_data)
// ----------------------------------------------------------------------------
// Get oscillation probabilities from pre-computed table using linear
// interpolation. For parameter values outside the tabulated range, call
// snu_probability_matrix
// ----------------------------------------------------------------------------
{
  struct snu_probability_matrix *p = (struct snu_probability_matrix *) user_data;

  if (!p)
  {
    glb_error("snu_tabulated_probability_matrix: no probability table given.");
    return GLBERR_INVALID_ARGS;
  }

//  int idx[p->n_p];
//  for (int i=0; i < p->n_p; i++)
//  {
//    double x = glbGetOscParamByName(osc_params, p->params[i]);
//    if (p->p_flags[i] & DEG_LOGSCALE)
//      x = log10(x);
//    if (x < p->p_min[i] || x >= p->p_max[i])
//      return snu_probability_matrix(P, cp_sign, E, psteps, length, density,
//                                    filter_sigma, user_data);
//    else
//      idx[i] = p->p_steps[i] * (x - p->p_min[i]) / (p->p_max[i] - p->p_min[i]);
//  }
  return GLB_SUCCESS;
}



