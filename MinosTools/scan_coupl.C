
#include "data_release.C"


for (double gcoupl : linspace(0., 4.0, 9)) {
    
    Chi2_3D(geomspace(0.01, 0.2, 3), geomspace(0.01, 0.3, 4), geomspace(1e-2, 1e4, 3), gcoupl*M_PI,  "test_chi2_sterile_g_decay_"+to_string(gcoupl)+"Pi_custom_3D.dat");

}