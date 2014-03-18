/*

   Defines the Gauss class representing a 2-dimensional gaussian with full
   covariance, and the GMix class represting a set of those.

*/
#ifndef _GMIX_HEADER_GUARD
#define _GMIX_HEADER_GUARD

#include <stdexcept>
#include <vector>
#include <cmath>
#include <cstdio>

//#include "image.h"
#include "fastexp.h"

namespace gmix {

    using std::vector;
    //using image::Image;

    static const double TWO_PI=6.28318530717958647693;
    static const double GAUSS_EXP_MAX_CHI2=25;

    struct Gauss {

        public:

            // argh no default initialization...
            Gauss() {
                p_=-9999;
                row_=-9999; col_=-9999;
                irr_=-9999; irc_=-9999; icc_=-9999;
                det_=-9999;

                drr_=-9999; drc_=-9999; dcc_=-9999;
                norm_=-9999; pnorm_=-9999;
            }
            Gauss(double p,
                  double row, double col,
                  double irr, double irc, double icc) {

                  set(p,row,col,irr,irc,icc);
            }
 
            void set(double p,
                     double row, double col,
                     double irr, double irc, double icc)
            {

                det_ = irr*icc - irc*irc;

                if (det_ <= 0) {
                    throw std::out_of_range("det <= 0");
                }

                p_   = p;
                row_ = row;
                col_ = col;
                irr_ = irr;
                irc_ = irc;
                icc_ = icc;

                drr_ = irr_/det_;
                drc_ = irc_/det_;
                dcc_ = icc_/det_;
                norm_ = 1./(TWO_PI*std::sqrt(det_));

                pnorm_ = p_*norm_;
            }

            inline double operator()(double row, double col) const {
                double u = row - row_;
                double v = col - col_;

                double chi2 =
                    dcc_*u*u + drr_*v*v - 2.0*drc_*u*v;

                double val=0.0;
                if (chi2 < GAUSS_EXP_MAX_CHI2) {
                    val = pnorm_*fastexp::expd( -0.5*chi2 );
                }

                return val;
            }

            void show() const {
                std::printf("p: %-12.8g row: %-12.8g col: %-12.8g irr: %-12.8g irc: %-12.8g icc: %-12.8g\n", p_, row_, col_, irr_, irc_, icc_);
            }

        private:

            double p_;
            double row_;
            double col_;
            double irr_;
            double irc_;
            double icc_;

            // derived quantities
            double det_;

            double drr_; // irr/det
            double drc_;
            double dcc_;

            double norm_; // 1/( 2*pi*sqrt(det) )
            double pnorm_; // p*norm

    };



    class GMix {
        public:
            GMix() {
                resize(0);
            }
            GMix(long ngauss) {
                resize(ngauss);
            }
            GMix(const vector<double> &pars) {
                set_from_pars(pars);
            }

            long size() const {
                return ngauss_;
            }

            inline double operator()(double row, double col) const {

                double val=0.0;

                for (long i=0; i<ngauss_; i++) {
                    const Gauss& gauss = data_[i];
                    val += gauss(row,col);
                }

                return val;
            }

            void resize(long ngauss) {
                if (ngauss < 0) {
                    throw std::runtime_error("ngauss must be >= 0");
                }
                ngauss_=ngauss;
                data_.resize(ngauss);
            }

            void set_from_pars(const vector<double> &pars) {
                long npars=pars.size();
                if ( (npars % 6) != 0) {
                    std::string err =
                        "GMix error: len(pars) must be multiple of 6";
                    throw std::runtime_error(err);
                }

                long ngauss = npars/6;
                this->resize(ngauss);

                for (long i=0; i<ngauss; i++) {
                    long beg=i*6;

                    Gauss &gauss = data_[i];

                    gauss.set(pars[beg+0],
                              pars[beg+1],
                              pars[beg+2],
                              pars[beg+3],
                              pars[beg+4],
                              pars[beg+5]);
                }
            }


            // no jacobian
            /*
            void render(Image& image, int nsub=1) const {
                double stepsize = 1./nsub;
                double offset = (nsub-1)*stepsize/2.;
                double areafac = 1./(nsub*nsub);

                long nrows=image.get_nrows();
                long ncols=image.get_ncols();

                for (long row=0; row<nrows; row++) {
                    for (long col=0; col<ncols; col++) {

                        double tval=0.0;
                        double trow = row-offset;
                        for (long irowsub=0; irowsub<nsub; irowsub++) {
                            double tcol = col-offset;
                            for (long icolsub=0; icolsub<nsub; icolsub++) {
                                tval += this(trow,tcol);
                                tcol += stepsize;
                            }
                            trow += stepsize;
                        }

                        tval *= areafac;
                        image(row,col) = tval;
                    }
                }

            }
            */

            void show() {
                for (long i=0; i<ngauss_; i++) {
                    data_[i].show();
                }
            }

        private:
            long ngauss_;
            vector<Gauss> data_;

    };

}

#endif
