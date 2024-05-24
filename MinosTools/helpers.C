#include <iostream>
#include <fstream>
#include <vector>
#include <utility>
#include <sys/stat.h>
#include <math.h>
#include <assert.h>
#include <complex>

#include "TH1D.h"
#include "TH2D.h"
#include "TString.h"
#include "THStack.h"
#include "TMatrixD.h"
#include "TFile.h"
#include "TDecompSVD.h"
#include "TMarker.h"
#include "TRandom3.h"
#include "TTree.h"
#include "TLeaf.h"
#include "TMarker.h"
#include "TGraph.h"
#include "TGraphSmooth.h"
#include "TMath.h"

using namespace std;

// template<typename T>
std::vector<double> linspace(double start_in, double end_in, int num_in)
{

  std::vector<double> linspaced;

  double start = static_cast<double>(start_in);
  double end = static_cast<double>(end_in);
  double num = static_cast<double>(num_in);

  if (num == 0) { return linspaced; }
  if (num == 1) 
    {
      linspaced.push_back(start);
      return linspaced;
    }

  double delta = (end - start) / (num - 1);

  for(int i=0; i < num-1; ++i)
    {
      linspaced.push_back(start + delta * i);
    }
  linspaced.push_back(end); // I want to ensure that start and end
                            // are exactly the same as the input
  return linspaced;
}
// template<typename T>
std::vector<double> geomspace(double start_in, double end_in, int num_in)
{
    std::vector<double> geospaced;

    double start = static_cast<double>(start_in);
    double end = static_cast<double>(end_in);
    double num = static_cast<double>(num_in);

    if (num == 0) { return geospaced; }
    if (num == 1) 
    {
        geospaced.push_back(start);
        return geospaced;
    }

    double ratio = std::pow(end / start, 1.0 / (num - 1));

    for(int i = 0; i < num; ++i)
    {
        double value = start * std::pow(ratio, i);
        geospaced.push_back(value);
    }

    return geospaced;
}
