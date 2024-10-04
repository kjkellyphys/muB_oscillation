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

#include "helpers.C"

using namespace std;


const double k1267 = 1.26693276;
const double kKmUnits = 1000.;
double Penalty_nuisance;
double Penalty_dm232;
double totalChi2_CC;
double totalChi2_NC;
double totalChi2;
bool got_beamOptics = false;
bool got_inputHistos = false;
  
// Near MC RecoVtrue MINOS
TH2D *NDNC_TrueNC_minos, *NDNC_NuMu_minos, *NDNC_BeamNue_minos, *NDNC_AppNue_minos, *NDNC_AppNuTau_minos;
TH2D *NDCC_TrueNC_minos, *NDCC_NuMu_minos, *NDCC_BeamNue_minos, *NDCC_AppNue_minos, *NDCC_AppNuTau_minos;

// Far MC RecoVtrue MINOS
TH2D *FDNC_TrueNC_minos, *FDNC_NuMu_minos, *FDNC_BeamNue_minos, *FDNC_AppNue_minos, *FDNC_AppNuTau_minos;
TH2D *FDCC_TrueNC_minos, *FDCC_NuMu_minos, *FDCC_BeamNue_minos, *FDCC_AppNue_minos, *FDCC_AppNuTau_minos;

// MC (Unoscillated) MINOS
TH1D *FDUnOscCC_MC_minos, *NDUnOscCC_MC_minos, *FDUnOscNC_MC_minos, *NDUnOscNC_MC_minos;

// MC (Oscillated) MINOS
TH1D *FDOscCC_MC_minos, *NDOscCC_MC_minos, *FDOscNC_MC_minos, *NDOscNC_MC_minos;

// Data MINOS
TH1D *FD_dataNC_minos, *FD_dataCC_minos, *ND_dataNC_minos, *ND_dataCC_minos;


// Near MC RecoVtrue MINOS+
TH2D *NDNC_TrueNC_minosPlus, *NDNC_NuMu_minosPlus, *NDNC_BeamNue_minosPlus, *NDNC_AppNue_minosPlus, *NDNC_AppNuTau_minosPlus;
TH2D *NDCC_TrueNC_minosPlus, *NDCC_NuMu_minosPlus, *NDCC_BeamNue_minosPlus, *NDCC_AppNue_minosPlus, *NDCC_AppNuTau_minosPlus;

// Far MC RecoVtrue MINOS+
TH2D *FDNC_TrueNC_minosPlus, *FDNC_NuMu_minosPlus, *FDNC_BeamNue_minosPlus, *FDNC_AppNue_minosPlus, *FDNC_AppNuTau_minosPlus;
TH2D *FDCC_TrueNC_minosPlus, *FDCC_NuMu_minosPlus, *FDCC_BeamNue_minosPlus, *FDCC_AppNue_minosPlus, *FDCC_AppNuTau_minosPlus;

// MC (Unoscillated) MINOS+
TH1D *FDUnOscCC_MC_minosPlus, *NDUnOscCC_MC_minosPlus, *FDUnOscNC_MC_minosPlus, *NDUnOscNC_MC_minosPlus;

// MC (Oscillated) MINOS+
TH1D *FDOscCC_MC_minosPlus, *NDOscCC_MC_minosPlus, *FDOscNC_MC_minosPlus, *NDOscNC_MC_minosPlus;

// Data MINOS+
TH1D *FD_dataNC_minosPlus, *FD_dataCC_minosPlus, *ND_dataNC_minosPlus, *ND_dataCC_minosPlus;


// MC (Unoscillated) Joint MINOS/MINOS+
TH1D *FDUnOscCC_MC, *NDUnOscCC_MC, *FDUnOscNC_MC, *NDUnOscNC_MC;

// MC (Oscillated) Joint MINOS/MINOS+
TH1D *FDOscCC_MC, *NDOscCC_MC, *FDOscNC_MC, *NDOscNC_MC;

// Data Joint MINOS/MINOS+
TH1D *FD_dataNC, *FD_dataCC, *ND_dataNC, *ND_dataCC;


//Covariance Matrices -- relative variance
TMatrixD* CoVarCC_relative;
TMatrixD* CoVarNC_relative;

//Covariance Matrices -- scaled, inverted
TMatrixD* CoVarCC_inverted;
TMatrixD* CoVarNC_inverted;
  

//---------------------------------------------------------------------------------
struct params
{
  double Dm232;
  double Dm221;
  double th23;
  double th12;
  double th13;
  double deltaCP;
  double Dm241;
  double th24;
  double th34;
  double th14;
  double delta24;
  double g_decay;
};
//---------------------------------------------------------------------------------
void Zombie(TFile* f);
void LoadInputHistograms(TFile* f);
void GenerateUnOscillatedSpectra();
void GenerateOscillatedSpectra(params my_pars);
void PrintParStatus(params my_pars);
TH1D* GetTwoDetSpectrum(TH1D* hND, TH1D* hFD);
TMatrixD* ScaleCovarianceMatrix(TH1D* pred, TMatrixD* mtx);
Double_t PenaltyTermDm232(Double_t dm232);
Double_t PenaltyTermNuisance(Double_t par, Double_t mean, Double_t sigma);
Double_t ChiSqFunction(TH1D* rPred, TH1D* rData, TMatrixD* CoVarInvert);
Double_t ComparePredWithData(TH1D* predCC, 
			     TH1D* dataCC, 
			     TMatrixD* CoVarCC_inverted, 
			     TH1D* predNC, 
			     TH1D* dataNC, 
			     TMatrixD* CoVarNC_inverted,
			     Double_t Dm2);
TH1D* CreateTotalSpectrum(params my_pars,
			  TH2D* TrueNC,
			  TH2D* NuMu,
			  TH2D* BeamNue,
			  TH2D* AppNue,
			  TH2D* AppNuTau,
			  double baseline);
double FourFlavourNuMuToNuSProbability
(const double energy, 
 double dm232, const double theta23, double dm221, 
 double dm243, const double theta12, 
 const double theta13, const double theta14,
 const double theta24, const double theta34,
 double delta1, double delta2, double delta3, double g_decay,
 const double baseline);
double FourFlavourDisappearanceWeight
(const double energy, 
 double dm232, const double theta23, double dm221, 
 double dm243, const double theta12, 
 const double theta13, const double theta14,
 const double theta24, const double theta34,
 double delta1, double delta2, double delta3, double g_decay,
 const double baseline);
double FourFlavourNuESurvivalProbability
(const double energy, 
 double dm232, const double theta23, double dm221, 
 double dm243, const double theta12, 
 const double theta13, const double theta14,
 const double theta24, const double theta34,
 double delta1, double delta2, double delta3, double g_decay,
 const double baseline);
double FourFlavourNuMuToNuEProbability
(const double energy, 
 double dm232, const double theta23, double dm221, 
 double dm243, const double theta12, 
 const double theta13, const double theta14,
 const double theta24, const double theta34,
 double delta1, double delta2, double delta3, double g_decay,
 const double baseline);
double FourFlavourNuMuToNuTauProbability
(const double energy, 
 double dm232, const double theta23, double dm221, 
 double dm243, const double theta12, 
 const double theta13, const double theta14,
 const double theta24, const double theta34,
 double delta1, double delta2, double delta3, double g_decay,
 const double baseline);
TH1D* CreateSpectrumComponent(params my_pars, TString OscType, TH2D* oscDummy, Double_t baseline);
//---------------------------------------------------------------------------------
void Zombie(TFile* f){

  if(f->IsZombie() && (!f->IsOpen())){
    std::cout << "File " << f->GetName() << " failed to open." << std::endl;
    assert(false);
  }
//   else{
    // std::cout << "File " << f->GetName() << " opened successfully" << std::endl;
//   }
}
//---------------------------------------------------------------------------------
void LoadInputHistograms(TFile* f)
{
  if(got_inputHistos) return;
  
  //Extract RecoToTrue MC simulations for MINOS
  f->GetObject("hRecoToTrueNDNCSelectedTrueNC_minos",   NDNC_TrueNC_minos);   assert(NDNC_TrueNC_minos);
  f->GetObject("hRecoToTrueNDNCSelectedNuMu_minos",     NDNC_NuMu_minos);     assert(NDNC_NuMu_minos);
  f->GetObject("hRecoToTrueNDNCSelectedBeamNue_minos",  NDNC_BeamNue_minos);  assert(NDNC_BeamNue_minos);
  f->GetObject("hRecoToTrueNDNCSelectedAppNue_minos",   NDNC_AppNue_minos);   assert(NDNC_AppNue_minos);
  f->GetObject("hRecoToTrueNDNCSelectedAppNuTau_minos", NDNC_AppNuTau_minos); assert(NDNC_AppNuTau_minos);

  f->GetObject("hRecoToTrueNDCCSelectedTrueNC_minos",   NDCC_TrueNC_minos);   assert(NDCC_TrueNC_minos);
  f->GetObject("hRecoToTrueNDCCSelectedNuMu_minos",     NDCC_NuMu_minos);     assert(NDCC_NuMu_minos);
  f->GetObject("hRecoToTrueNDCCSelectedBeamNue_minos",  NDCC_BeamNue_minos);  assert(NDCC_BeamNue_minos);
  f->GetObject("hRecoToTrueNDCCSelectedAppNue_minos",   NDCC_AppNue_minos);   assert(NDCC_AppNue_minos);
  f->GetObject("hRecoToTrueNDCCSelectedAppNuTau_minos", NDCC_AppNuTau_minos); assert(NDCC_AppNuTau_minos);

  f->GetObject("hRecoToTrueFDNCSelectedTrueNC_minos",   FDNC_TrueNC_minos);   assert(FDNC_TrueNC_minos);
  f->GetObject("hRecoToTrueFDNCSelectedNuMu_minos",     FDNC_NuMu_minos);     assert(FDNC_NuMu_minos);
  f->GetObject("hRecoToTrueFDNCSelectedBeamNue_minos",  FDNC_BeamNue_minos);  assert(FDNC_BeamNue_minos);
  f->GetObject("hRecoToTrueFDNCSelectedAppNue_minos",   FDNC_AppNue_minos);   assert(FDNC_AppNue_minos);
  f->GetObject("hRecoToTrueFDNCSelectedAppNuTau_minos", FDNC_AppNuTau_minos); assert(FDNC_AppNuTau_minos);

  f->GetObject("hRecoToTrueFDCCSelectedTrueNC_minos",   FDCC_TrueNC_minos);   assert(FDCC_TrueNC_minos);
  f->GetObject("hRecoToTrueFDCCSelectedNuMu_minos",     FDCC_NuMu_minos);     assert(FDCC_NuMu_minos);
  f->GetObject("hRecoToTrueFDCCSelectedBeamNue_minos",  FDCC_BeamNue_minos);  assert(FDCC_BeamNue_minos);
  f->GetObject("hRecoToTrueFDCCSelectedAppNue_minos",   FDCC_AppNue_minos);   assert(FDCC_AppNue_minos);
  f->GetObject("hRecoToTrueFDCCSelectedAppNuTau_minos", FDCC_AppNuTau_minos); assert(FDCC_AppNuTau_minos);

  //Extract RecoToTrue MC simulations for MINOS+
  f->GetObject("hRecoToTrueNDNCSelectedTrueNC_minosPlus",   NDNC_TrueNC_minosPlus);   assert(NDNC_TrueNC_minosPlus);
  f->GetObject("hRecoToTrueNDNCSelectedNuMu_minosPlus",     NDNC_NuMu_minosPlus);     assert(NDNC_NuMu_minosPlus);
  f->GetObject("hRecoToTrueNDNCSelectedBeamNue_minosPlus",  NDNC_BeamNue_minosPlus);  assert(NDNC_BeamNue_minosPlus);
  f->GetObject("hRecoToTrueNDNCSelectedAppNue_minosPlus",   NDNC_AppNue_minosPlus);   assert(NDNC_AppNue_minosPlus);
  f->GetObject("hRecoToTrueNDNCSelectedAppNuTau_minosPlus", NDNC_AppNuTau_minosPlus); assert(NDNC_AppNuTau_minosPlus);

  f->GetObject("hRecoToTrueNDCCSelectedTrueNC_minosPlus",   NDCC_TrueNC_minosPlus);   assert(NDCC_TrueNC_minosPlus);
  f->GetObject("hRecoToTrueNDCCSelectedNuMu_minosPlus",     NDCC_NuMu_minosPlus);     assert(NDCC_NuMu_minosPlus);
  f->GetObject("hRecoToTrueNDCCSelectedBeamNue_minosPlus",  NDCC_BeamNue_minosPlus);  assert(NDCC_BeamNue_minosPlus);
  f->GetObject("hRecoToTrueNDCCSelectedAppNue_minosPlus",   NDCC_AppNue_minosPlus);   assert(NDCC_AppNue_minosPlus);
  f->GetObject("hRecoToTrueNDCCSelectedAppNuTau_minosPlus", NDCC_AppNuTau_minosPlus); assert(NDCC_AppNuTau_minosPlus);

  f->GetObject("hRecoToTrueFDNCSelectedTrueNC_minosPlus",   FDNC_TrueNC_minosPlus);   assert(FDNC_TrueNC_minosPlus);
  f->GetObject("hRecoToTrueFDNCSelectedNuMu_minosPlus",     FDNC_NuMu_minosPlus);     assert(FDNC_NuMu_minosPlus);
  f->GetObject("hRecoToTrueFDNCSelectedBeamNue_minosPlus",  FDNC_BeamNue_minosPlus);  assert(FDNC_BeamNue_minosPlus);
  f->GetObject("hRecoToTrueFDNCSelectedAppNue_minosPlus",   FDNC_AppNue_minosPlus);   assert(FDNC_AppNue_minosPlus);
  f->GetObject("hRecoToTrueFDNCSelectedAppNuTau_minosPlus", FDNC_AppNuTau_minosPlus); assert(FDNC_AppNuTau_minosPlus);

  f->GetObject("hRecoToTrueFDCCSelectedTrueNC_minosPlus",   FDCC_TrueNC_minosPlus);   assert(FDCC_TrueNC_minosPlus);
  f->GetObject("hRecoToTrueFDCCSelectedNuMu_minosPlus",     FDCC_NuMu_minosPlus);     assert(FDCC_NuMu_minosPlus);
  f->GetObject("hRecoToTrueFDCCSelectedBeamNue_minosPlus",  FDCC_BeamNue_minosPlus);  assert(FDCC_BeamNue_minosPlus);
  f->GetObject("hRecoToTrueFDCCSelectedAppNue_minosPlus",   FDCC_AppNue_minosPlus);   assert(FDCC_AppNue_minosPlus);
  f->GetObject("hRecoToTrueFDCCSelectedAppNuTau_minosPlus", FDCC_AppNuTau_minosPlus); assert(FDCC_AppNuTau_minosPlus);
  
  //Extract data histograms
  //MINOS
  f->GetObject("dataFDNC_minos", FD_dataNC_minos); assert(FD_dataNC_minos);
  f->GetObject("dataFDCC_minos", FD_dataCC_minos); assert(FD_dataCC_minos);

  f->GetObject("dataNDNC_minos", ND_dataNC_minos); assert(ND_dataNC_minos);
  f->GetObject("dataNDCC_minos", ND_dataCC_minos); assert(ND_dataCC_minos);
  
  //MINOS+
  f->GetObject("dataFDNC_minosPlus", FD_dataNC_minosPlus); assert(FD_dataNC_minosPlus);
  f->GetObject("dataFDCC_minosPlus", FD_dataCC_minosPlus); assert(FD_dataCC_minosPlus);

  f->GetObject("dataNDNC_minosPlus", ND_dataNC_minosPlus); assert(ND_dataNC_minosPlus);
  f->GetObject("dataNDCC_minosPlus", ND_dataCC_minosPlus); assert(ND_dataCC_minosPlus);  
  
  got_inputHistos = true;
}
//---------------------------------------------------------------------------------
void GenerateUnOscillatedSpectra()
{
  params my_pars0;
  my_pars0.Dm232 = 0.0;
  my_pars0.Dm221 = 0.0;
  my_pars0.th23  = 0.0;
  my_pars0.th13  = 0.0;
  my_pars0.th12  = 0.0;
  my_pars0.deltaCP = 0.0;
  my_pars0.Dm241   = 0.0;
  my_pars0.th24    = 0.0;
  my_pars0.th34    = 0.0;
  my_pars0.th14    = 0.0;
  my_pars0.delta24 = 0.0;
  my_pars0.g_decay = 0.0;

  //UnOscillated CC MC -- MINOS
  NDUnOscCC_MC_minos = (TH1D*)CreateTotalSpectrum(	my_pars0, 
							NDCC_TrueNC_minos, 
							NDCC_NuMu_minos, 
							NDCC_BeamNue_minos, 
							NDCC_AppNue_minos, 
							NDCC_AppNuTau_minos, 
							1.04*kKmUnits);
  FDUnOscCC_MC_minos = (TH1D*)CreateTotalSpectrum(	my_pars0, 
							FDCC_TrueNC_minos, 
							FDCC_NuMu_minos, 
							FDCC_BeamNue_minos, 
							FDCC_AppNue_minos, 
							FDCC_AppNuTau_minos, 
							735.0*kKmUnits);

  //UnOscillated NC MC -- MINOS
  NDUnOscNC_MC_minos = (TH1D*)CreateTotalSpectrum(	my_pars0, 
							NDNC_TrueNC_minos, 
							NDNC_NuMu_minos, 
							NDNC_BeamNue_minos, 
							NDNC_AppNue_minos, 
							NDNC_AppNuTau_minos, 
							1.04*kKmUnits);
  FDUnOscNC_MC_minos = (TH1D*)CreateTotalSpectrum(	my_pars0, 
							FDNC_TrueNC_minos, 
							FDNC_NuMu_minos, 
							FDNC_BeamNue_minos, 
							FDNC_AppNue_minos, 
							FDNC_AppNuTau_minos, 
							735.0*kKmUnits);
  
  //UnOscillated CC MC -- MINOS+
  NDUnOscCC_MC_minosPlus = (TH1D*)CreateTotalSpectrum(	my_pars0, 
							NDCC_TrueNC_minosPlus, 
							NDCC_NuMu_minosPlus, 
							NDCC_BeamNue_minosPlus, 
							NDCC_AppNue_minosPlus, 
							NDCC_AppNuTau_minosPlus, 
							1.04*kKmUnits);
  FDUnOscCC_MC_minosPlus = (TH1D*)CreateTotalSpectrum(	my_pars0, 
							FDCC_TrueNC_minosPlus, 
							FDCC_NuMu_minosPlus, 
							FDCC_BeamNue_minosPlus, 
							FDCC_AppNue_minosPlus, 
							FDCC_AppNuTau_minosPlus, 
							735.0*kKmUnits);

  //UnOscillated NC MC -- MINOS+
  NDUnOscNC_MC_minosPlus = (TH1D*)CreateTotalSpectrum(	my_pars0, 
							NDNC_TrueNC_minosPlus, 
							NDNC_NuMu_minosPlus, 
							NDNC_BeamNue_minosPlus, 
							NDNC_AppNue_minosPlus, 
							NDNC_AppNuTau_minosPlus, 
							1.04*kKmUnits);
  FDUnOscNC_MC_minosPlus = (TH1D*)CreateTotalSpectrum(	my_pars0, 
							FDNC_TrueNC_minosPlus, 
							FDNC_NuMu_minosPlus, 
							FDNC_BeamNue_minosPlus, 
							FDNC_AppNue_minosPlus, 
							FDNC_AppNuTau_minosPlus, 
							735.0*kKmUnits);
}
//---------------------------------------------------------------------------------
void GenerateOscillatedSpectra(params my_pars)
{
  //Oscillated CC MC -- MINOS
  NDOscCC_MC_minos = (TH1D*)CreateTotalSpectrum(	my_pars, 
							NDCC_TrueNC_minos, 
							NDCC_NuMu_minos, 
							NDCC_BeamNue_minos, 
							NDCC_AppNue_minos, 
							NDCC_AppNuTau_minos, 
							1.04*kKmUnits);
  FDOscCC_MC_minos = (TH1D*)CreateTotalSpectrum(	my_pars, 
							FDCC_TrueNC_minos, 
							FDCC_NuMu_minos, 
							FDCC_BeamNue_minos, 
							FDCC_AppNue_minos, 
							FDCC_AppNuTau_minos, 
							735.0*kKmUnits);
  //Oscillated NC MC -- MINOS
  NDOscNC_MC_minos = (TH1D*)CreateTotalSpectrum(	my_pars, 
							NDNC_TrueNC_minos, 
							NDNC_NuMu_minos, 
							NDNC_BeamNue_minos, 
							NDNC_AppNue_minos, 
							NDNC_AppNuTau_minos, 
							1.04*kKmUnits);
  FDOscNC_MC_minos = (TH1D*)CreateTotalSpectrum(	my_pars, 
							FDNC_TrueNC_minos, 
							FDNC_NuMu_minos, 
							FDNC_BeamNue_minos, 
							FDNC_AppNue_minos, 
							FDNC_AppNuTau_minos, 
							735.0*kKmUnits);
  //Oscillated CC MC -- MINOS+
  NDOscCC_MC_minosPlus = (TH1D*)CreateTotalSpectrum(	my_pars, 
							NDCC_TrueNC_minosPlus, 
							NDCC_NuMu_minosPlus, 
							NDCC_BeamNue_minosPlus, 
							NDCC_AppNue_minosPlus, 
							NDCC_AppNuTau_minosPlus, 
							1.04*kKmUnits);
  FDOscCC_MC_minosPlus = (TH1D*)CreateTotalSpectrum(	my_pars, 
							FDCC_TrueNC_minosPlus, 
							FDCC_NuMu_minosPlus, 
							FDCC_BeamNue_minosPlus, 
							FDCC_AppNue_minosPlus, 
							FDCC_AppNuTau_minosPlus, 
							735.0*kKmUnits);
  //Oscillated NC MC -- MINOS+
  NDOscNC_MC_minosPlus = (TH1D*)CreateTotalSpectrum(	my_pars, 
							NDNC_TrueNC_minosPlus, 
							NDNC_NuMu_minosPlus, 
							NDNC_BeamNue_minosPlus, 
							NDNC_AppNue_minosPlus, 
							NDNC_AppNuTau_minosPlus, 
							1.04*kKmUnits);
  FDOscNC_MC_minosPlus = (TH1D*)CreateTotalSpectrum(	my_pars, 
							FDNC_TrueNC_minosPlus, 
							FDNC_NuMu_minosPlus, 
							FDNC_BeamNue_minosPlus, 
							FDNC_AppNue_minosPlus, 
							FDNC_AppNuTau_minosPlus, 
							735.0*kKmUnits);
}
//---------------------------------------------------------------------------------
void PrintParStatus(params my_pars)
{
  std::cout << "" << std::endl;
  std::cout << "/==========Parameter Status==========/" << std::endl;
  std::cout << "" << std::endl;
  std::cout << "Dm232                =      " << my_pars.Dm232 << std::endl;
  std::cout << "Dm221                =      " << my_pars.Dm221 << std::endl;
  std::cout << "Theta23              =      " << my_pars.th23 << std::endl;
  std::cout << "Theta12              =      " << my_pars.th12 << std::endl;
  std::cout << "Theta13              =      " << my_pars.th13 << std::endl;
  std::cout << "DeltaCP              =      " << my_pars.deltaCP << std::endl;
  std::cout << "Dm241                =      " << my_pars.Dm241 << std::endl;
  std::cout << "Theta24              =      " << my_pars.th24 << std::endl;
  std::cout << "Theta34              =      " << my_pars.th34 << std::endl;
  std::cout << "Theta14              =      " << my_pars.th14 << std::endl;
  std::cout << "Delta24              =      " << my_pars.delta24 << std::endl;
  std::cout << "g_decay              =      " << my_pars.g_decay << std::endl;
  std::cout << "" << std::endl;
  std::cout << "/====================================/" << std::endl;
  std::cout << "" << std::endl;
}
//---------------------------------------------------------------------------------
TH1D* GetTwoDetSpectrum(TH1D* hND, TH1D* hFD){

  int NDbins = hND->GetNbinsX();
  int FDbins = hFD->GetNbinsX();
  int Nbins = NDbins + FDbins;
  const int Nedges = Nbins + 1; 
 
  Double_t edges[Nedges];

  edges[0]=0;
////// 
  double shift = 40.0;	//shift bin edges of ND spectrum by maximum energy 
			//of first spectrum to force increasing bin edges
  
  for(int i=1;i<=FDbins;i++){
    edges[i] = hFD->GetXaxis()->GetBinUpEdge(i);
  }
  for(int i=1;i<=NDbins;i++){
    edges[i+FDbins] = hND->GetXaxis()->GetBinUpEdge(i) + shift;
  }
//////
  TH1D* hSpec = new TH1D("","",Nbins,edges);
  hSpec->Sumw2();

  for(int i=1;i<=FDbins;i++){
    hSpec->SetBinContent(i,hFD->GetBinContent(i));
    hSpec->SetBinError(i,hFD->GetBinError(i));
  }
  for(int i=1;i<=NDbins;i++){
    hSpec->SetBinContent(i+FDbins,hND->GetBinContent(i));
    hSpec->SetBinError(i+FDbins,hND->GetBinError(i));
  }

  return hSpec;
}
//---------------------------------------------------------------------------------
TMatrixD* ScaleCovarianceMatrix(TH1D* pred, TMatrixD* mtx)
{
  Double_t binNumber = pred->GetNbinsX();
  TMatrixD* scaled_mtx = (TMatrixD*)mtx->Clone();
  TMatrixD* inverted_mtx = (TMatrixD*)mtx->Clone();

  Double_t bci, bcj, stat, syst, sig2;
  for(Int_t i=1; i<=binNumber; ++i){
    for(Int_t j=1; j<=binNumber; ++j){
      bci = pred->GetBinContent(i);
      bcj = pred->GetBinContent(j);
      //Poisson statistical uncertainty
      stat = 0;
      if(i==j){
        stat = bci;
      }
      //Systematic uncertainty
      syst = 0;
      syst = mtx->operator()(i-1,j-1);
      syst = bci*bcj*syst;
      //Sum stat and syst uncertainty
      sig2 = stat + syst;
      scaled_mtx->operator()(i-1,j-1) = sig2;
    }
  }
 
  TDecompSVD* DeCom = new TDecompSVD(*scaled_mtx);
  *inverted_mtx = DeCom->Invert();
  
  delete DeCom;
  
  return inverted_mtx;
}
//---------------------------------------------------------------------------------
Double_t PenaltyTermDm232(Double_t dm232)
{
  Double_t dm232_pen = 0.0;
  dm232_pen = TMath::Power( (TMath::Abs(dm232) - 0.0025) , 2); // numerator
  dm232_pen /= TMath::Power( 0.0005, 2);

  return dm232_pen;
}
//---------------------------------------------------------------------------------
Double_t PenaltyTermNuisance(Double_t par, Double_t mean, Double_t sigma)
{
  Double_t pen = 0.0;
  pen = TMath::Power( (par - mean) , 2); // numerator
  pen /= TMath::Power( sigma, 2);

  return pen;
}
//---------------------------------------------------------------------------------
Double_t ChiSqFunction(TH1D* rPred, TH1D* rData, TMatrixD* CoVarInvert)
{
  if(!(rPred->GetNbinsX() == rData->GetNbinsX())){ 
    std::cout << "Binning Error. Asserting" << std::endl;
    assert(false);
  }

  Int_t NumberOfBins = rPred->GetNbinsX();

  TVectorD Difference(NumberOfBins);

  for(Int_t i=1; i<=NumberOfBins; ++i){
    Difference(i-1) = (rData->GetBinContent(i) - rPred->GetBinContent(i));
  }

  TVectorD temp = Difference;
  temp *= (*CoVarInvert);

  Double_t TotalChiSq = temp*Difference;

  return TotalChiSq;

}
//---------------------------------------------------------------------------------
Double_t ComparePredWithData(TH1D* predCC, 
			     TH1D* dataCC, 
			     TMatrixD* CoVarCC_inverted, 
			     TH1D* predNC, 
			     TH1D* dataNC, 
			     TMatrixD* CoVarNC_inverted,
			     Double_t Dm2
			    )
{
  totalChi2_CC = ChiSqFunction(predCC, dataCC, CoVarCC_inverted);
  totalChi2_NC = ChiSqFunction(predNC, dataNC, CoVarNC_inverted);
  
  //Atmospheric mass-splitting penalty term
  Penalty_dm232   = PenaltyTermDm232(Dm2);
  
  totalChi2 = totalChi2_CC + totalChi2_NC + Penalty_dm232;

//   std::cout << "" << std::endl;
//   std::cout << "/=======Chi2 Calculator Output=======/" << std::endl;
//   std::cout << "" << std::endl;
//   std::cout << "Penalty Dm232         =     " << Penalty_dm232 << std::endl;
//   std::cout << "Total Chi2 CC         =     " << totalChi2_CC << std::endl;
//   std::cout << "Total Chi2 NC         =     " << totalChi2_NC << std::endl;
//   std::cout << "Total Chi2            =     " << totalChi2 << std::endl;
//   std::cout << "" << std::endl;
//   std::cout << "/====================================/" << std::endl;
//   std::cout << "" << std::endl;

  return totalChi2;
}
//---------------------------------------------------------------------------------
TH1D* CreateTotalSpectrum(params my_pars,
			  TH2D* TrueNC,
			  TH2D* NuMu,
			  TH2D* BeamNue,
			  TH2D* AppNue,
			  TH2D* AppNuTau,
			  double baseline
			 )
{
  TH1D* vtruenc   = (TH1D*)CreateSpectrumComponent(my_pars, "TrueNC",   TrueNC,   baseline);
  TH1D* vnumu     = (TH1D*)CreateSpectrumComponent(my_pars, "NuMu",     NuMu,     baseline);
  TH1D* vbeamnue  = (TH1D*)CreateSpectrumComponent(my_pars, "BeamNue",  BeamNue,  baseline);
  TH1D* vappnue   = (TH1D*)CreateSpectrumComponent(my_pars, "AppNue",   AppNue,   baseline);
  TH1D* vappnutau = (TH1D*)CreateSpectrumComponent(my_pars, "AppNuTau", AppNuTau, baseline);

  TH1D* hTotal = new TH1D(*vtruenc);
  hTotal->Add(vnumu);
  hTotal->Add(vbeamnue);
  hTotal->Add(vappnue);
  hTotal->Add(vappnutau);

  return hTotal;
}
//---------------------------------------------------------------------------------
double FourFlavourNuMuToNuSProbability
(const double energy, 
 double dm232, const double theta23, double dm221, 
 double dm243, const double theta12, 
 const double theta13, const double theta14,
 const double theta24, const double theta34,
 double delta1, double delta2, double delta3, double g_decay,
 const double baseline)
{ 

  // Calculate other mass splittings
  const double dm231 = dm221 + dm232;
  const double dm241 = dm231 + dm243;

  const double c12 = cos(theta12); const double s12 = sin(theta12);
  const double c13 = cos(theta13); const double s13 = sin(theta13);
  const double c14 = cos(theta14); const double s14 = sin(theta14);
  const double c23 = cos(theta23); const double s23 = sin(theta23);
  const double c24 = cos(theta24); const double s24 = sin(theta24);
  const double c34 = cos(theta34); const double s34 = sin(theta34);

  complex<double> expNegCP13 = complex<double>(cos(delta1), -sin(delta1));
  complex<double> expNegCP14 = complex<double>(cos(delta2), -sin(delta2));
  complex<double> expNegCP24 = complex<double>(cos(delta3), -sin(delta3));

  complex<double> Umu2  =  c12 * c23 * c24
                        -  c24 * s12 * s13 * s23 * conj(expNegCP13)
                        -  c13 * s12 * s14 * s24 * expNegCP24 * conj(expNegCP14);

  complex<double> Umu3  =  c13 * c24 * s23
                        -  s13 * s14 * s24 * expNegCP13 * expNegCP24 * conj(expNegCP14);
  
  complex<double> Umu4  =  c14 * s24 * expNegCP24;
  

  complex<double> Us2   =  -c13 * c24 * c34 * s12 * s14 * conj(expNegCP14)
                           -c12 * c23 * c34 * s24 * conj(expNegCP24)
                           +c34 * s12 * s13 * s23 * s24 * conj(expNegCP13 * expNegCP24)
                           +c23 * s12 * s13 * s34 * conj(expNegCP13)
                           +c12 * s23 * s34;
  
  complex<double> Us3   =  -c24 * c34 * s13 * s14 * expNegCP13 * conj(expNegCP14)
                           -c13 * c34 * s23 * s24 * conj(expNegCP24)
                           -c13 * c23 * s34;
  
  complex<double> Us4   =  c14 * c24 * c34;
  
  
  complex<double> i(0.0, 1.0);
  
  double DeltaM21 = (k1267 * dm221 * baseline / kKmUnits) / energy;
  double DeltaM31 = (k1267 * dm231 * baseline / kKmUnits) / energy;
  double DeltaM41 = (k1267 * dm241 * baseline / kKmUnits) / energy;
  
    // Decay
  double m4_Gamma4 = 2 * g_decay*g_decay * dm241 / (16 * M_PI);
  double DeltaDecay4 = (k1267 * m4_Gamma4 * baseline / kKmUnits) / energy;
  
  complex<double> expDeltaM21 = complex<double>(cos(DeltaM21), -sin(DeltaM21));
  complex<double> expDeltaM31 = complex<double>(cos(DeltaM31), -sin(DeltaM31));
  complex<double> expTwoDeltaM41 = complex<double>(cos(2*DeltaM41), -sin(2*DeltaM41));
  
  double oscProb = norm(-2.0 * i * conj(Umu2) * Us2 * sin(DeltaM21) * expDeltaM21
                        -2.0 * i * conj(Umu3) * Us3 * sin(DeltaM31) * expDeltaM31
                        + conj(Umu4) * Us4 * (expTwoDeltaM41 * exp(-DeltaDecay4) - 1.0));  
  return oscProb;
}
//---------------------------------------------------------------------------------
double FourFlavourDisappearanceWeight
(const double energy, 
 double dm232, const double theta23, double dm221, 
 double dm243, const double theta12, 
 const double theta13, const double theta14,
 const double theta24, const double theta34,
 double delta1, double delta2, double delta3, double g_decay,
 const double baseline)
{

  // Calculate other mass splittings
  const double dm231 = dm221 + dm232;
  const double dm241 = dm231 + dm243;

  const double c12 = cos(theta12); const double s12 = sin(theta12);
  const double c13 = cos(theta13); const double s13 = sin(theta13);
  const double c14 = cos(theta14); const double s14 = sin(theta14);
  const double c23 = cos(theta23); const double s23 = sin(theta23);
  const double c24 = cos(theta24); const double s24 = sin(theta24);

  complex<double> expNegCP13 = complex<double>(cos(delta1), -sin(delta1));
  complex<double> expNegCP14 = complex<double>(cos(delta2), -sin(delta2));
  complex<double> expNegCP24 = complex<double>(cos(delta3), -sin(delta3));

  complex<double> Umu2  =  c12 * c23 * c24
                        -  c24 * s12 * s13 * s23 * conj(expNegCP13)
                        -  c13 * s12 * s14 * s24 * expNegCP24 * conj(expNegCP14);

  complex<double> Umu3  =  c13 * c24 * s23
                        -  s13 * s14 * s24 * expNegCP13 * expNegCP24 * conj(expNegCP14);
  
  complex<double> Umu4  =  c14 * s24 * expNegCP24;
  
  complex<double> i(0.0, 1.0);

  
  double DeltaM21 = (k1267 * dm221 * baseline / kKmUnits) / energy;
  double DeltaM31 = (k1267 * dm231 * baseline / kKmUnits) / energy;
  double DeltaM41 = (k1267 * dm241 * baseline / kKmUnits) / energy;
  
  // Decay
  double m4_Gamma4 = 2 * g_decay*g_decay * dm241 / (16 * M_PI);
  double DeltaDecay4 = (k1267 * m4_Gamma4 * baseline / kKmUnits) / energy;
  
  complex<double> expDeltaM21 = complex<double>(cos(DeltaM21), -sin(DeltaM21));
  complex<double> expDeltaM31 = complex<double>(cos(DeltaM31), -sin(DeltaM31));
  complex<double> expTwoDeltaM41 = complex<double>(cos(DeltaM41*2), -sin(DeltaM41*2));
  
  double oscProb  = norm(1.0 
			 - 2.0 * i * conj(Umu2) * Umu2 * sin(DeltaM21) * expDeltaM21 
			 - 2.0 * i * conj(Umu3) * Umu3 * sin(DeltaM31) * expDeltaM31 
            + conj(Umu4) * Umu4 * (expTwoDeltaM41 * exp(-DeltaDecay4) - 1.0));  
  
  return oscProb;
}
//---------------------------------------------------------------------------------
double FourFlavourNuESurvivalProbability
(const double energy, 
 double dm232, const double theta23, double dm221, 
 double dm243, const double theta12, 
 const double theta13, const double theta14,
 const double theta24, const double theta34,
 double delta1, double delta2, double delta3, double g_decay,
 const double baseline)
{

  // Calculate other mass splittings
  const double dm231 = dm221 + dm232;
  const double dm241 = dm231 + dm243;

  const double s12 = sin(theta12);
  const double c13 = cos(theta13); const double s13 = sin(theta13);
  const double c14 = cos(theta14); const double s14 = sin(theta14);

  complex<double> expNegCP13 = complex<double>(cos(delta1), -sin(delta1));
  complex<double> expNegCP14 = complex<double>(cos(delta2), -sin(delta2));

  complex<double> Ue2   =  c13 * c14 * s12;
  complex<double> Ue3   =  c14 * s13 * expNegCP13;
  complex<double> Ue4   =  s14 * expNegCP14;
  
  complex<double> i(0.0, 1.0);
  
  double DeltaM21 = (k1267 * dm221 * baseline / kKmUnits) / energy;
  double DeltaM31 = (k1267 * dm231 * baseline / kKmUnits) / energy;
  double DeltaM41 = (k1267 * dm241 * baseline / kKmUnits) / energy;
  
  // Decay
  double m4_Gamma4 = 2 * g_decay*g_decay * dm241 / (16 * M_PI);
  double DeltaDecay4 = (k1267 * m4_Gamma4 * baseline / kKmUnits) / energy;
  
  complex<double> expDeltaM21 = complex<double>(cos(DeltaM21), -sin(DeltaM21));
  complex<double> expDeltaM31 = complex<double>(cos(DeltaM31), -sin(DeltaM31));
  complex<double> expTwoDeltaM41 = complex<double>(cos(DeltaM41*2), -sin(DeltaM41*2));
  
  double oscProb = norm(1.0 
			- 2.0 * i * conj(Ue2) * Ue2 * sin(DeltaM21) * expDeltaM21 
			- 2.0 * i * conj(Ue3) * Ue3 * sin(DeltaM31) * expDeltaM31 
			+ conj(Ue4) * Ue4 * (expTwoDeltaM41 * exp(-DeltaDecay4) - 1.0));

  return oscProb;
}
//---------------------------------------------------------------------------------
double FourFlavourNuMuToNuEProbability
(const double energy, 
 double dm232, const double theta23, double dm221, 
 double dm243, const double theta12, 
 const double theta13, const double theta14,
 const double theta24, const double theta34,
 double delta1, double delta2, double delta3, double g_decay,
 const double baseline)
{

  // Calculate other mass splittings
  const double dm231 = dm221 + dm232;
  const double dm241 = dm231 + dm243;

  const double c12 = cos(theta12); const double s12 = sin(theta12);
  const double c13 = cos(theta13); const double s13 = sin(theta13);
  const double c14 = cos(theta14); const double s14 = sin(theta14);
  const double c23 = cos(theta23); const double s23 = sin(theta23);
  const double c24 = cos(theta24); const double s24 = sin(theta24);

  complex<double> expNegCP13 = complex<double>(cos(delta1), -sin(delta1));
  complex<double> expNegCP14 = complex<double>(cos(delta2), -sin(delta2));
  complex<double> expNegCP24 = complex<double>(cos(delta3), -sin(delta3));
  
  complex<double> Umu2  =  c12 * c23 * c24
                        -  c24 * s12 * s13 * s23 * conj(expNegCP13)
                        -  c13 * s12 * s14 * s24 * expNegCP24 * conj(expNegCP14);

  complex<double> Umu3  =  c13 * c24 * s23
                        -  s13 * s14 * s24 * expNegCP13 * expNegCP24 * conj(expNegCP14);
  
  complex<double> Umu4  =  c14 * s24 * expNegCP24;


  complex<double> Ue2   =  c13 * c14 * s12;
  complex<double> Ue3   =  c14 * s13 * expNegCP13;
  complex<double> Ue4   =  s14 * expNegCP14;
  
  complex<double> i(0.0, 1.0);
  
  double DeltaM21 = (k1267 * dm221 * baseline / kKmUnits) / energy;
  double DeltaM31 = (k1267 * dm231 * baseline / kKmUnits) / energy;
  double DeltaM41 = (k1267 * dm241 * baseline / kKmUnits) / energy;
  
  // Decay
  double m4_Gamma4 = 2 * g_decay*g_decay * dm241 / (16 * M_PI);
  double DeltaDecay4 = (k1267 * m4_Gamma4 * baseline / kKmUnits) / energy;
  
  complex<double> expDeltaM21 = complex<double>(cos(DeltaM21), -sin(DeltaM21));
  complex<double> expDeltaM31 = complex<double>(cos(DeltaM31), -sin(DeltaM31));
  complex<double> expTwoDeltaM41 = complex<double>(cos(2*DeltaM41), -sin(2*DeltaM41));
  
  double oscProb = norm(-2.0 * i * conj(Umu2) * Ue2 * sin(DeltaM21) * expDeltaM21       
			-2.0 * i * conj(Umu3) * Ue3 * sin(DeltaM31) * expDeltaM31
            + conj(Umu4) * Ue4 * (expTwoDeltaM41 * exp(-DeltaDecay4) - 1.0));  
  return oscProb;
}
//---------------------------------------------------------------------------------
double FourFlavourNuMuToNuTauProbability
(const double energy, 
 double dm232, const double theta23, double dm221, 
 double dm243, const double theta12, 
 const double theta13, const double theta14,
 const double theta24, const double theta34,
 double delta1, double delta2, double delta3, double g_decay,
 const double baseline)
{

  // Calculate other mass splittings
  const double dm231 = dm221 + dm232;
  const double dm241 = dm231 + dm243;

  const double c12 = cos(theta12); const double s12 = sin(theta12);
  const double c13 = cos(theta13); const double s13 = sin(theta13);
  const double c14 = cos(theta14); const double s14 = sin(theta14);
  const double c23 = cos(theta23); const double s23 = sin(theta23);
  const double c24 = cos(theta24); const double s24 = sin(theta24);
  const double c34 = cos(theta34); const double s34 = sin(theta34);

  complex<double> expNegCP13 = complex<double>(cos(delta1), -sin(delta1));
  complex<double> expNegCP14 = complex<double>(cos(delta2), -sin(delta2));
  complex<double> expNegCP24 = complex<double>(cos(delta3), -sin(delta3));

  complex<double> Umu2  =  c12 * c23 * c24
                        -  c24 * s12 * s13 * s23 * conj(expNegCP13)
                        -  c13 * s12 * s14 * s24 * expNegCP24 * conj(expNegCP14);

  complex<double> Umu3  =  c13 * c24 * s23
                        -  s13 * s14 * s24 * expNegCP13 * expNegCP24 * conj(expNegCP14);
  
  complex<double> Umu4  =  c14 * s24 * expNegCP24;


  complex<double> Utau2 =  -c12 * c34 * s23
                           -c23 * c34 * s12 * s13 * conj(expNegCP13)
                           -c13 * c24 * s12 * s14 * s34 * conj(expNegCP14)
                           -c12 * c23 * s24 * s34 * conj(expNegCP24)
                           +s12 * s13 * s23 * s24 * s34 * conj(expNegCP13 * expNegCP24);

  complex<double> Utau3 =  c13 * c23 * c34
                        -  c24 * s13 * s14 * s34 * expNegCP13 * conj(expNegCP14)
                        -  c13 * s23 * s24 * s34 * conj(expNegCP24);

  complex<double> Utau4 =  c14 * c24 * s34;

  complex<double> i(0.0, 1.0);
  
  double DeltaM21 = (k1267 * dm221 * baseline / kKmUnits) / energy;
  double DeltaM31 = (k1267 * dm231 * baseline / kKmUnits) / energy;
  double DeltaM41 = (k1267 * dm241 * baseline / kKmUnits) / energy;
  
  complex<double> expDeltaM21 = complex<double>(cos(DeltaM21), -sin(DeltaM21));
  complex<double> expDeltaM31 = complex<double>(cos(DeltaM31), -sin(DeltaM31));
  complex<double> expTwoDeltaM41 = complex<double>(cos(2*DeltaM41), -sin(2*DeltaM41));
  
  // Decay
  double m4_Gamma4 = 2 * g_decay*g_decay * dm241 / (16 * M_PI);
  double DeltaDecay4 = (k1267 * m4_Gamma4 * baseline / kKmUnits) / energy;

  double oscProb =  norm(-2.0 * i * conj(Umu2) * Utau2 * sin(DeltaM21) * expDeltaM21     
			 -2.0 * i * conj(Umu3) * Utau3 * sin(DeltaM31) * expDeltaM31
            + conj(Umu4) * Utau4 * (expTwoDeltaM41 * exp(-DeltaDecay4) - 1.0));  

  return oscProb;
}
//---------------------------------------------------------------------------------
TH1D* CreateSpectrumComponent(params my_pars, TString OscType, TH2D* oscDummy, Double_t baseline)
{
  TH1D* bintemplate = oscDummy->ProjectionY();
  bintemplate->Reset();
 
  const double k1267 = 1.26693276;

  // Loop over every true energy bin in the reco vs. true matrices, then loop over every reco energy in that bin                                                 
  // to calculate an oscillation weight for that reco energy based on the true energy. 
  TAxis *Yaxis = oscDummy->GetYaxis();
  TAxis *Xaxis = oscDummy->GetXaxis();

  // Define Dm243 such that its actually Dm241 being altered.
  //41 = 43 + 32 + 21 
  //43 = 41 - 32 - 21
  Double_t dm243 = 0.0;

  dm243 = my_pars.Dm241 - my_pars.Dm232 - my_pars.Dm221;

  for(Int_t x = 1; x <= Xaxis->GetNbins(); x++){
    Double_t OscWeight = 0.0;
    
    if(baseline > 0){
      
      // Default iterations (1 at bin center)
      Int_t n_LoverE = 1;
      Double_t LoverE[5];
      LoverE[0] = Xaxis->GetBinCenter(x);
      
      // This is averaging oscialltions in true energy bins - see Technical Note http://minos-docdb.fnal.gov/cgi-bin/RetrieveFile?docid=10203&version=2
      const Double_t W = Xaxis->GetBinWidth(x);
      const Double_t arg = k1267*dm243*W; // half-period of oscillation
      Double_t sample = W/2/sqrt(3);

      if(arg!=0) sample = TMath::ACos(TMath::Sin(arg)/arg)/arg*W/2;

      n_LoverE = 2;
      Double_t bc = LoverE[0]; // bin center
      LoverE[0] = bc - sample;
      LoverE[1] = bc + sample;

      const Double_t E = 1.0;

      for(int i = 0; i < n_LoverE; i++){
	
	// each Osctype has a different probability function
	if(OscType == "TrueNC"){

	  OscWeight += FourFlavourNuMuToNuSProbability( E, 
  							my_pars.Dm232, 
							my_pars.th23, 
							my_pars.Dm221, 
							dm243, 
							my_pars.th12, 
							my_pars.th13, 
							my_pars.th14, 
							my_pars.th24, 
							my_pars.th34, 
							my_pars.deltaCP,  
							0, 
							my_pars.delta24,
                            my_pars.g_decay,
							LoverE[i]*kKmUnits);
	}
	if(OscType == "NuMu"){

	  OscWeight += FourFlavourDisappearanceWeight( E, 
  							my_pars.Dm232, 
							my_pars.th23, 
							my_pars.Dm221, 
							dm243, 
							my_pars.th12, 
							my_pars.th13, 
							my_pars.th14, 
							my_pars.th24, 
							my_pars.th34, 
							my_pars.deltaCP,  
							0, 
							my_pars.delta24,
                            my_pars.g_decay,
							LoverE[i]*kKmUnits);
	}
	if(OscType == "BeamNue"){
	  
	  OscWeight += FourFlavourNuESurvivalProbability( E, 
  							my_pars.Dm232, 
							my_pars.th23, 
							my_pars.Dm221, 
							dm243, 
							my_pars.th12, 
							my_pars.th13, 
							my_pars.th14, 
							my_pars.th24, 
							my_pars.th34, 
							my_pars.deltaCP,  
							0, 
							my_pars.delta24,
                            my_pars.g_decay,
							LoverE[i]*kKmUnits);
	}
	if(OscType == "AppNue"){
	  
	  OscWeight += FourFlavourNuMuToNuEProbability( E, 
  							my_pars.Dm232, 
							my_pars.th23, 
							my_pars.Dm221, 
							dm243, 
							my_pars.th12, 
							my_pars.th13, 
							my_pars.th14, 
							my_pars.th24, 
							my_pars.th34, 
							my_pars.deltaCP,  
							0, 
							my_pars.delta24,
                            my_pars.g_decay,
							LoverE[i]*kKmUnits);
	}
	if(OscType == "AppNuTau"){

	  OscWeight += FourFlavourNuMuToNuTauProbability( E, 
  							my_pars.Dm232, 
							my_pars.th23, 
							my_pars.Dm221, 
							dm243, 
							my_pars.th12, 
							my_pars.th13, 
							my_pars.th14, 
							my_pars.th24, 
							my_pars.th34, 
							my_pars.deltaCP,  
							0, 
							my_pars.delta24,
                            my_pars.g_decay,
							LoverE[i]*kKmUnits);
	}
      }
      // Now average this
      OscWeight /= n_LoverE;
    }
    else // if baseline < 0
      {
	
        if(OscType == "TrueNC")   OscWeight = 0.0;
        if(OscType == "NuMu")     OscWeight = 1.0;
        if(OscType == "BeamNue")  OscWeight = 1.0;
        if(OscType == "AppNue")   OscWeight = 0.0;
        if(OscType == "AppNuTau") OscWeight = 0.0;
      }

    // using the oscillation weight, fill a 1d histogram for each type of event with the oscillated reco energy 
    for(Int_t y = 1; y <= Yaxis->GetNbins(); y++){
      
      Double_t sumWeights = 0;
      
      if(OscType == "TrueNC"){
	sumWeights += oscDummy->GetBinContent(x,y)*(1.0-OscWeight);
      }
      else{
	sumWeights += oscDummy->GetBinContent(x,y)*(OscWeight);
      }
      Double_t currBinContents = bintemplate->GetBinContent( y );
      bintemplate->SetBinContent( y, sumWeights + currBinContents);

    }

  }

  return bintemplate;
}
//---------------------------------------------------------------------------------

//begin main function
void dataRelease_chi2Calc_compile(string path = "dataRelease.root",
					double Dm232 = 2.43005123913581740e-03,
					double Dm221 = 0.0000754,
					double th23  = 9.28598228704929918e-01,
					double th12  = 0.5540758073,
					double th13  = 0.149116,
					double deltaCP = 0.0,
					double Dm241 = 2.32492426050590582e-03,
					double th24  = 1.05321302928372360e-02,
					double th34  = 8.35186824552614469e-03,
					double th14  = 0.0,
					double delta24 = 0.0,
					double g_decay = 0.0
				       )
{
  
  TString fileName = path;
  
  TFile *f  = new TFile(fileName);

  Zombie(f); 

  params my_pars;
  my_pars.Dm232   = Dm232;
  my_pars.Dm221   = Dm221;
  my_pars.th23    = th23;
  my_pars.th12    = th12;
  my_pars.th13    = th13;
  my_pars.deltaCP = deltaCP;
  my_pars.Dm241   = Dm241;
  my_pars.th24    = th24;
  my_pars.th34    = th34;
  my_pars.th14    = th14;
  my_pars.delta24 = delta24;
  my_pars.g_decay = g_decay;

  PrintParStatus(my_pars);

////////////////////////Begin Calculation//////////////////////////////////////
  //Extract covariance matrices
  CoVarCC_relative = (TMatrixD*)f->Get("TotalCCCovar"); assert(CoVarCC_relative);
  CoVarNC_relative = (TMatrixD*)f->Get("TotalNCCovar"); assert(CoVarNC_relative);
  
  //Load inputs 
  LoadInputHistograms(f);
  
  //Construct MINOS/MINOS+ two detector data spectra
  TH1D* h2det_data_NC_minos = (TH1D*)GetTwoDetSpectrum(ND_dataNC_minos,FD_dataNC_minos);
  TH1D* h2det_data_CC_minos = (TH1D*)GetTwoDetSpectrum(ND_dataCC_minos,FD_dataCC_minos);
  
  TH1D* h2det_data_NC_minosPlus = (TH1D*)GetTwoDetSpectrum(ND_dataNC_minosPlus,FD_dataNC_minosPlus);
  TH1D* h2det_data_CC_minosPlus = (TH1D*)GetTwoDetSpectrum(ND_dataCC_minosPlus,FD_dataCC_minosPlus);
 
  //Combine MINOS & MINOS+ data spectra 
  TH1D* dataCC = (TH1D*)h2det_data_CC_minos->Clone();
  dataCC->Add(h2det_data_CC_minosPlus);
  
  TH1D* dataNC = (TH1D*)h2det_data_NC_minos->Clone();
  dataNC->Add(h2det_data_NC_minosPlus); 
 
  //Generate predictions 
  GenerateUnOscillatedSpectra();
  GenerateOscillatedSpectra(my_pars);

  //Construct MINOS/MINOS+ two detector prediction spectra
  TH1D* h2det_MC_CC_minos = (TH1D*)GetTwoDetSpectrum(NDOscCC_MC_minos,FDOscCC_MC_minos);
  TH1D* h2det_MC_NC_minos = (TH1D*)GetTwoDetSpectrum(NDOscNC_MC_minos,FDOscNC_MC_minos);
  
  TH1D* h2det_MC_CC_minosPlus = (TH1D*)GetTwoDetSpectrum(NDOscCC_MC_minosPlus,FDOscCC_MC_minosPlus);
  TH1D* h2det_MC_NC_minosPlus = (TH1D*)GetTwoDetSpectrum(NDOscNC_MC_minosPlus,FDOscNC_MC_minosPlus);
 
  //Combine MINOS & MINOS+ prediction spectra 
  TH1D* predCC = (TH1D*)h2det_MC_CC_minos->Clone();
  predCC->Add(h2det_MC_CC_minosPlus);
  
  TH1D* predNC = (TH1D*)h2det_MC_NC_minos->Clone();
  predNC->Add(h2det_MC_NC_minosPlus); 
 
  //Scale and invert covariance matrices for fitting
  CoVarCC_inverted = (TMatrixD*)ScaleCovarianceMatrix(predCC,CoVarCC_relative);
  CoVarNC_inverted = (TMatrixD*)ScaleCovarianceMatrix(predNC,CoVarNC_relative);

  Double_t chi2 = ComparePredWithData(predCC, 
				      dataCC, 
				      CoVarCC_inverted, 
				      predNC, 
				      dataNC, 
				      CoVarNC_inverted,
				      my_pars.Dm232);

//////////////////////////End Calculation//////////////////////////////////////

//////////////////////////Begin Output File////////////////////////////////////

  TString outputfilename = "./chi2Calc_output.root";

  TFile* save = new TFile(outputfilename,"RECREATE");
  save->cd();
  
  TTree* tree = new TTree("tree", "tree");
  tree->Branch("chi2", &chi2 );
  tree->Branch("output_totalChi2CC", &totalChi2_CC );
  tree->Branch("output_totalChi2NC", &totalChi2_NC );
  tree->Branch("output_PenDm232", &Penalty_dm232 );
  tree->Branch("param_dm232", &my_pars.Dm232 );
  tree->Branch("param_dm221", &my_pars.Dm221 );
  tree->Branch("param_theta23", &my_pars.th23 );
  tree->Branch("param_theta12", &my_pars.th12 );
  tree->Branch("param_theta13", &my_pars.th13 );
  tree->Branch("param_deltaCP", &my_pars.deltaCP );
  tree->Branch("param_dm241", &my_pars.Dm241 );
  tree->Branch("param_th24", &my_pars.th14 );
  tree->Branch("param_th24", &my_pars.th24 );
  tree->Branch("param_th34", &my_pars.th34 );
  tree->Branch("param_delta24", &my_pars.delta24 );
  tree->Branch("param_g_decay", &my_pars.g_decay );
  tree->Fill();
  tree->Write();

  //Two Detector spectra data
  h2det_data_NC_minos->Write("TwoDetectorSpectrum_minos_data_NC"); 
  h2det_data_CC_minos->Write("TwoDetectorSpectrum_minos_data_CC"); 
  h2det_data_NC_minosPlus->Write("TwoDetectorSpectrum_minosPlus_data_NC"); 
  h2det_data_CC_minosPlus->Write("TwoDetectorSpectrum_minosPlus_data_CC"); 
  dataCC->Write("TwoDetectorSpectrum_data_CC"); 
  dataNC->Write("TwoDetectorSpectrum_data_NC");
  
  //Two Detector spectra prediction
  h2det_MC_NC_minos->Write("TwoDetectorSpectrum_minos_prediction_NC"); 
  h2det_MC_CC_minos->Write("TwoDetectorSpectrum_minos_prediction_CC"); 
  h2det_MC_NC_minosPlus->Write("TwoDetectorSpectrum_minosPlus_prediction_NC"); 
  h2det_MC_CC_minosPlus->Write("TwoDetectorSpectrum_minosPlus_prediction_CC"); 
  predCC->Write("TwoDetectorSpectrum_prediction_CC"); 
  predNC->Write("TwoDetectorSpectrum_prediction_NC"); 
  
  //ND spectra MINOS
  NDUnOscCC_MC_minos->Write("NearDetector_minos_unoscillated_CC"); 
  NDOscCC_MC_minos->Write("NearDetector_minos_prediction_CC"); 
  ND_dataCC_minos->Write("NearDetector_minos_data_CC"); 
  
  NDUnOscNC_MC_minos->Write("NearDetector_minos_unoscillated_NC"); 
  NDOscNC_MC_minos->Write("NearDetector_minos_prediction_NC"); 
  ND_dataNC_minos->Write("NearDetector_minos_data_NC");
  
  //FD spectra MINOS
  FDUnOscCC_MC_minos->Write("FarDetector_minos_unoscillated_CC"); 
  FDOscCC_MC_minos->Write("FarDetector_minos_prediction_CC"); 
  FD_dataCC_minos->Write("FarDetector_minos_data_CC"); 
  
  FDUnOscNC_MC_minos->Write("FarDetector_minos_unoscillated_NC"); 
  FDOscNC_MC_minos->Write("FarDetector_minos_prediction_NC"); 
  FD_dataNC_minos->Write("FarDetector_minos_data_NC");
  
  //ND spectra MINOS+
  NDUnOscCC_MC_minosPlus->Write("NearDetector_minosPlus_unoscillated_CC"); 
  NDOscCC_MC_minosPlus->Write("NearDetector_minosPlus_prediction_CC"); 
  ND_dataCC_minosPlus->Write("NearDetector_minosPlus_data_CC"); 
  
  NDUnOscNC_MC_minosPlus->Write("NearDetector_minosPlus_unoscillated_NC"); 
  NDOscNC_MC_minosPlus->Write("NearDetector_minosPlus_prediction_NC"); 
  ND_dataNC_minosPlus->Write("NearDetector_minosPlus_data_NC");
  
  //FD spectra MINOS+
  FDUnOscCC_MC_minosPlus->Write("FarDetector_minosPlus_unoscillated_CC"); 
  FDOscCC_MC_minosPlus->Write("FarDetector_minosPlus_prediction_CC"); 
  FD_dataCC_minosPlus->Write("FarDetector_minosPlus_data_CC"); 
  
  FDUnOscNC_MC_minosPlus->Write("FarDetector_minosPlus_unoscillated_NC"); 
  FDOscNC_MC_minosPlus->Write("FarDetector_minosPlus_prediction_NC"); 
  FD_dataNC_minosPlus->Write("FarDetector_minosPlus_data_NC");
  
  //Inverted covariance matrices
  CoVarCC_inverted->Write("Inverse_CovarianceMatrix_CC");
  CoVarNC_inverted->Write("Inverse_CovarianceMatrix_NC"); 
  
  save->Close();

//////////////////////////End Output File//////////////////////////////////////
 
  std::cout << " " << std::endl;
}


//begin simplified main function
Double_t Simple_dataRelease_chi2Calc(
          TFile* f,
					double Dm232 = 2.43005123913581740e-03,
					double Dm221 = 0.0000754,
					double th23  = 9.28598228704929918e-01,
					double th12  = 0.5540758073,
					double th13  = 0.149116,
					double deltaCP = 0.0,
					double Dm241 = 2.32492426050590582e-03,
					double th24  = 1.05321302928372360e-02,
					double th34  = 8.35186824552614469e-03,
					double th14  = 0.0,
					double delta24 = 0.0,
					double g_decay = 0.0
				       )
{

  params my_pars;
  my_pars.Dm232   = Dm232;
  my_pars.Dm221   = Dm221;
  my_pars.th23    = th23;
  my_pars.th12    = th12;
  my_pars.th13    = th13;
  my_pars.deltaCP = deltaCP;
  my_pars.Dm241   = Dm241;
  my_pars.th24    = th24;
  my_pars.th34    = th34;
  my_pars.th14    = th14;
  my_pars.delta24 = delta24;
  my_pars.g_decay = g_decay;
    
////////////////////////Begin Calculation//////////////////////////////////////
  //Extract covariance matrices
  CoVarCC_relative = (TMatrixD*)f->Get("TotalCCCovar"); assert(CoVarCC_relative);
  CoVarNC_relative = (TMatrixD*)f->Get("TotalNCCovar"); assert(CoVarNC_relative);
  
  //Load inputs 
  LoadInputHistograms(f);
  
  //Construct MINOS/MINOS+ two detector data spectra
  TH1D* h2det_data_NC_minos = (TH1D*)GetTwoDetSpectrum(ND_dataNC_minos,FD_dataNC_minos);
  TH1D* h2det_data_CC_minos = (TH1D*)GetTwoDetSpectrum(ND_dataCC_minos,FD_dataCC_minos);
  
  TH1D* h2det_data_NC_minosPlus = (TH1D*)GetTwoDetSpectrum(ND_dataNC_minosPlus,FD_dataNC_minosPlus);
  TH1D* h2det_data_CC_minosPlus = (TH1D*)GetTwoDetSpectrum(ND_dataCC_minosPlus,FD_dataCC_minosPlus);
 
  //Combine MINOS & MINOS+ data spectra 
  TH1D* dataCC = (TH1D*)h2det_data_CC_minos->Clone();
  dataCC->Add(h2det_data_CC_minosPlus);
  
  TH1D* dataNC = (TH1D*)h2det_data_NC_minos->Clone();
  dataNC->Add(h2det_data_NC_minosPlus); 
 
  //Generate predictions 
  GenerateUnOscillatedSpectra();
  GenerateOscillatedSpectra(my_pars);

  //Construct MINOS/MINOS+ two detector prediction spectra
  TH1D* h2det_MC_CC_minos = (TH1D*)GetTwoDetSpectrum(NDOscCC_MC_minos,FDOscCC_MC_minos);
  TH1D* h2det_MC_NC_minos = (TH1D*)GetTwoDetSpectrum(NDOscNC_MC_minos,FDOscNC_MC_minos);
  
  TH1D* h2det_MC_CC_minosPlus = (TH1D*)GetTwoDetSpectrum(NDOscCC_MC_minosPlus,FDOscCC_MC_minosPlus);
  TH1D* h2det_MC_NC_minosPlus = (TH1D*)GetTwoDetSpectrum(NDOscNC_MC_minosPlus,FDOscNC_MC_minosPlus);
 
  //Combine MINOS & MINOS+ prediction spectra 
  TH1D* predCC = (TH1D*)h2det_MC_CC_minos->Clone();
  predCC->Add(h2det_MC_CC_minosPlus);
  
  TH1D* predNC = (TH1D*)h2det_MC_NC_minos->Clone();
  predNC->Add(h2det_MC_NC_minosPlus); 
 
  //Scale and invert covariance matrices for fitting
  CoVarCC_inverted = (TMatrixD*)ScaleCovarianceMatrix(predCC,CoVarCC_relative);
  CoVarNC_inverted = (TMatrixD*)ScaleCovarianceMatrix(predNC,CoVarNC_relative);

  Double_t chi2 = ComparePredWithData(predCC, 
				      dataCC, 
				      CoVarCC_inverted, 
				      predNC, 
				      dataNC, 
				      CoVarNC_inverted,
				      my_pars.Dm232);

  // f->Close(); // Close the file
  // delete f; // Delete the TFile object to release memory
//////////////////////////End Calulation//////////////////////////////////////
  return chi2;
}

// void printChi2ForTh24(std::vector<double> th24Values, std::vector<double> dm241Values, double g_decay, const std::string& fileName) {
//     std::ofstream outputFile(fileName);
//     if (!outputFile.is_open()) {
//         std::cerr << "Error opening file: " << fileName << std::endl;
//         return;
//     }
//     else{
//         outputFile << "# g_decay th24 dm2_41/eV^2 chi2_total" << std::endl;
//     }

//     TString fileName = "dataRelease.root";
//     // TFile *f  = new TFile(fileName, "READONLY");
//     TFile *f  = new TFile(fileName);
//     Zombie(f); 

//     for (double th24 : th24Values) {
//         for (double dm241 : dm241Values){
//             // Call the modified dataRelease_chi2Calc_compile function with each th24 value
//             Double_t chi2Results = Simple_dataRelease_chi2Calc(
//             f, 
//             2.43005123913581740e-03,    // Dm232 = 2.43005123913581740e-03,
//             0.0000754,  // Dm221 = 0.0000754,
//             9.28598228704929918e-01,    // th23  = 9.28598228704929918e-01,
//             0.5540758073,   // th12  = 0.5540758073,
//             0.149116,   // th13  = 0.149116,
//             0.0,    // deltaCP = 0.0,
//             dm241,  // Dm241 = 2.32492426050590582e-03,
//             th24,   // th24  = 1.05321302928372360e-02,
//             8.35186824552614469e-03,    // th34  = 8.35186824552614469e-03,
//             0.3,    // th14  = 0.0,
//             0.0,    // delta24 = 0.0
//             g_decay
//             );
//             // Print the chi2 values for the current th24 to the file
//             outputFile << g_decay << " " << th24 << " " << dm241 << " " << chi2Results  << std::endl;
//         }    
//     }

//     outputFile.close();
//     f->Close();
// }

void calculateAndPrintChi2(TFile* f, double th23, double th24, double dm232, double dm241, double g_decay, int& currentIteration, int totalIterations, std::ofstream& outputFile) {
  // Call the modified dataRelease_chi2Calc_compile function with each th24 value
  Double_t chi2Results = Simple_dataRelease_chi2Calc(
    f,
    dm232,    // Dm232 = 2.43005123913581740e-03,
    0.0000754,  // Dm221 = 0.0000754,
    th23,    // th23  = 9.28598228704929918e-01,
    0.5540758073, // th12  = 0.5540758073,
    0.149116,   // th13  = 0.149116,
    0.0,    // deltaCP = 0.0,
    dm241,  // Dm241 = 2.32492426050590582e-03,
    th24,   // th24  = 1.05321302928372360e-02,
    0.0,    // th34  = 8.35186824552614469e-03,
    0.0,    // th14  = 0.0,
    0.0,    // delta24 = 0.0
    g_decay // fixed g_decay
  );

  outputFile << g_decay << " " << th23 << " " << th24 << " " << dm232 << " " << dm241 << " " << chi2Results << std::endl;

  // Update progress bar
  currentIteration++;
  float progress = static_cast<float>(currentIteration) / totalIterations;
  int barWidth = 70;
  std::cout << "[";
  int pos = barWidth * progress;
  for (int i = 0; i < barWidth; ++i) {
    if (i < pos) std::cout << "=";
    else if (i == pos) std::cout << ">";
    else std::cout << " ";
  }
  std::cout << "] " << std::setw(3) << static_cast<int>(progress * 100.0) << "%\r";
  std::cout.flush();
}


void Chi2_3D(std::vector<double> th23Values, std::vector<double> th24Values, std::vector<double> dm241Values, double g_decay, const std::string& fileName) {
  std::ofstream outputFile(fileName);
  if (!outputFile.is_open()) {
    std::cerr << "Error opening file: " << fileName << std::endl;
    return;
  }
  else {
    outputFile << "# g_decay th23 th24 dm2_41/eV^2 chi2_total" << std::endl;
  }

  TString datafile = "dataRelease.root";
  // TFile *f  = new TFile(datafile, "READONLY");
  TFile *f  = new TFile(datafile);
  Zombie(f); 


  int totalIterations = th23Values.size() * th24Values.size() * dm241Values.size();
  int currentIteration = 0;

  for (double th23 : th23Values) {
    for (double th24 : th24Values) {
      for (double dm241 : dm241Values) {
          calculateAndPrintChi2(f, th23, th24, 2.5e-03, dm241, g_decay, currentIteration, totalIterations, outputFile);
      }
    }
  }
  
  std::cout << std::endl;
  outputFile.close();
  // f->Close();
  // delete f;

}

// void Chi2_4D(std::vector<double> th34Values, std::vector<double> th24Values, std::vector<double> dm232Values, std::vector<double> dm241Values, double g_decay, const std::string& fileName) {
//   std::ofstream outputFile(fileName);
//   if (!outputFile.is_open()) {
//     std::cerr << "Error opening file: " << fileName << std::endl;
//     return;
//   }
//   else {
//     outputFile << "# g_decay th34 th24 dm2_32/eV^2 dm2_41/eV^2 chi2_total" << std::endl;
//   }

//   TString datafile = "dataRelease.root";
//   // TFile *f  = new TFile(datafile, "READONLY");
//   TFile *f  = new TFile(datafile);
//   Zombie(f); 

//   int totalIterations = th34Values.size() * th24Values.size() * dm241Values.size() * dm232Values.size();
//   int currentIteration = 0;

//   for (double th34 : th34Values) {
//     for (double th24 : th24Values) {
//       for (double dm232 : dm232Values) {
//         for (double dm241 : dm241Values) {
//           calculateAndPrintChi2(f, th34, th24, dm232, dm241, g_decay, currentIteration, totalIterations, outputFile);
//         }
//       }
//     }
//   }
//   std::cout << std::endl;
//   outputFile.close();
//   // f->Close();
//   // delete f;
// }


void scan_3D() {
  for (double gcoupl : linspace(0, 4.0, 9)) {
    Chi2_3D(linspace(0.5, 1.0, 10), geomspace(0.01, 0.2, 40), geomspace(1e-2, 1e4, 40),  gcoupl*M_PI,  "test_chi2_sterile_g_decay_"+to_string(gcoupl)+"Pi_custom_3D.dat");
  }
}

// double Dm232 = 2.43005123913581740e-03,
// double Dm221 = 0.0000754,
// double th23  = 9.28598228704929918e-01,
// double th12  = 0.5540758073,
// double th13  = 0.149116,
// double deltaCP = 0.0,
// double Dm241 = 2.32492426050590582e-03,
// double th24  = 1.05321302928372360e-02,
// double th34  = 8.35186824552614469e-03,
// double th14  = 0.0,
// double delta24 = 0.0