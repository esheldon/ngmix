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
#include <sstream>

#include "shape.h"
#include "image.h"
#include "jacobian.h"
#include "fastexp.h"

namespace NGMix {

    using std::vector;
    //using image::Image;

    static const double TWO_PI=6.28318530717958647693;

    // For speed mainly, although some upper limit is needed because fast exp
    // fails at very large arguments
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
                    val = pnorm_*expd( -0.5*chi2 );
                }

                return val;
            }

            void print() const {
                std::printf("p: %-12.8g row: %-12.8g col: %-12.8g irr: %-12.8g irc: %-12.8g icc: %-12.8g\n", p_, row_, col_, irr_, irc_, icc_);
            }

        protected:

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
            // must be virtual!
            virtual ~GMix() {}

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

            // virtual because more specific models will deal with
            // this differently

            virtual void set_from_pars(const vector<double> &pars) {
                long npars=pars.size();
                if ( (npars % 6) != 0) {
                    std::stringstream err;
                    err << "GMix error: expected number of pars to be "
                        <<" multiple of 6 but got "<<npars<<"\n";
                    throw std::runtime_error(err.str());
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


            // This just creates a default jacobian
            void render(image::Image &image, int nsub=1) const {
                jacobian::Jacobian j;
                render(image, j, nsub);
            }

            // render using the jacobian image transformation
            void render(image::Image &image,
                        const jacobian::Jacobian &jacob,
                        int nsub=1) const {

                double col0=jacob.col0;
                double row0=jacob.row0;
                double dudrow=jacob.dudrow;
                double dudcol=jacob.dudcol;
                double dvdrow=jacob.dvdrow;
                double dvdcol=jacob.dvdcol;

                double stepsize = 1./nsub;
                double offset = (nsub-1)*stepsize/2.;
                double areafac = 1./(nsub*nsub);

                double ustepsize = stepsize*dudcol;
                double vstepsize = stepsize*dvdcol;

                long nrows=image.get_nrows();
                long ncols=image.get_ncols();

                for (long row=0; row<nrows; row++) {
                    for (long col=0; col<ncols; col++) {

                        double tval=0.0;
                        double trow = row-offset;
                        double lowcol = col-offset;

                        for (long irowsub=0; irowsub<nsub; irowsub++) {

                            // always start from lowcol position, then step u,v later
                            double u=dudrow*(trow - row0) + dudcol*(lowcol - col0);
                            double v=dvdrow*(trow - row0) + dvdcol*(lowcol - col0);

                            for (long icolsub=0; icolsub<nsub; icolsub++) {

                                tval += (*this)(u,v);

                                u += ustepsize;
                                v += vstepsize;
                            }
                            // step to next sub-row
                            trow += stepsize;
                        }

                        tval *= areafac;
                        image(row,col) = tval;
                    }
                }
            }


            void get_loglike(const image::Image &image,
                             const image::Image &weight,
                             const jacobian::Jacobian &jacob,
                             double *loglike,
                             double *s2n_numer,
                             double *s2n_denom) const {

                double col0=jacob.col0;
                double row0=jacob.row0;
                double dudrow=jacob.dudrow;
                double dudcol=jacob.dudcol;
                double dvdrow=jacob.dvdrow;
                double dvdcol=jacob.dvdcol;

                *loglike=0;
                *s2n_numer=0;
                *s2n_denom=0;
                long nrows=image.get_nrows();
                long ncols=image.get_ncols();

                for (long row=0; row<nrows; row++) {

                    // always start from lowcol position, then step u,v later
                    double u=dudrow*(row - row0) + dudcol*(0.0 - col0);
                    double v=dvdrow*(row - row0) + dvdcol*(0.0 - col0);

                    for (long col=0; col<ncols; col++) {

                        double ivar=weight(row,col);
                        if (ivar <= 0.0) {
                            continue;
                        }

                        double model_val = (*this)(row,col);
                        double pixel_val = image(row,col);

                        double diff=model_val-pixel_val;

                        (*loglike) += diff*diff*ivar;
                        (*s2n_numer) += pixel_val*model_val*ivar;
                        (*s2n_denom) += model_val*model_val*ivar;

                        u += dudcol;
                        v += dvdcol;

                    } // columns
                } //rows

                (*loglike) *= (-0.5);

            } // get loglike

            void print() {
                for (long i=0; i<ngauss_; i++) {
                    data_[i].print();
                }
            }

        protected:
            long ngauss_;
            vector<Gauss> data_;

    };

    enum gmix_model {
        GMIX_FULL,
        GMIX_GAUSS,
        GMIX_COELLIP,
        GMIX_TURB,
        GMIX_EXP,
        GMIX_DEV,
        GMIX_BDF,
    };


    static const size_t SIMPLE_NPARS=6;

    static const long GAUSS_NGAUSS=1;
    static const double GAUSS_PVALS[] = {1.0};
    static const double GAUSS_FVALS[] = {1.0};

    static const long EXP_NGAUSS=6;
    static const double EXP_PVALS[] = {
        0.00061601229677880041, 
        0.0079461395724623237, 
        0.053280454055540001, 
        0.21797364640726541, 
        0.45496740582554868, 
        0.26521634184240478
    };
    static const double EXP_FVALS[] = {
        0.002467115141477932, 
        0.018147435573256168, 
        0.07944063151366336, 
        0.27137669897479122, 
        0.79782256866993773, 
        2.1623306025075739
        };

    static const long DEV_NGAUSS=10;
    static const double DEV_PVALS[] = {
        6.5288960012625658e-05, 
        0.00044199216814302695, 
        0.0020859587871659754, 
        0.0075913681418996841, 
        0.02260266219257237, 
        0.056532254390212859, 
        0.11939049233042602, 
        0.20969545753234975, 
        0.29254151133139222, 
        0.28905301416582552
    };
    static const double DEV_FVALS[] = {
        3.068330909892871e-07,
        3.551788624668698e-06,
        2.542810833482682e-05,
        0.0001466508940804874,
        0.0007457199853069548,
        0.003544702600428794,
        0.01648881157673708,
        0.07893194619504579,
        0.4203787615506401,
        3.055782252301236
    };


    static const long TURB_NGAUSS=3;

    static const double TURB_PVALS[] = {
        0.596510042804182,
        0.4034898268889178,
        1.303069003078001e-07
    };
    static const double TURB_FVALS[] = {
        0.5793612389470884,
        1.621860687127999,
        7.019347162356363
    };


    // this is meant to be inherited
    class GMixSimple : public GMix {

        protected:
            // this is actually generic for simple
            virtual void _set_from_pars_fp(const vector<double> &pars,
                                   long ngauss,
                                   const double* pvals,
                                   const double* fvals) {

                if ( pars.size() != this->npars_ ) {
                    std::stringstream err;
                    err << "GMix error: expected "<<this->npars_
                        <<" pars but got "<<pars.size()<<"\n";
                    throw std::runtime_error(err.str());
                }
                if ( ngauss != this->ngauss_ ) {
                    std::stringstream err;
                    err << "GMix error: expected "<<this->ngauss_
                        <<" gaussians but got "<<ngauss<<"\n";
                    throw std::runtime_error(err.str());
                }

                this->resize(ngauss);

                Shape shape(pars[2], pars[3]);

                double row    = pars[0];
                double col    = pars[1];
                double T      = pars[4];
                double counts = pars[5];

                for (long i=0; i<ngauss_; i++) {
                    Gauss &gauss = data_[i];

                    double T_i = T*fvals[i];
                    double counts_i=counts*pvals[i];

                    // set can throw out_of_range
                    gauss.set(counts_i,
                              row,
                              col, 
                              (T_i/2.)*(1-shape.e1), 
                              (T_i/2.)*shape.e2,
                              (T_i/2.)*(1+shape.e1));
                }

                return;
            }

            size_t npars_;

    }; // class GMixSimple


    class GMixGauss : public GMixSimple {
    
        public:
            GMixGauss() {
                npars_=SIMPLE_NPARS;
                resize(GAUSS_NGAUSS);
            }
            GMixGauss(const vector<double> &pars) {
                npars_=SIMPLE_NPARS;
                set_from_pars(pars);
            }
            virtual void set_from_pars(const vector<double> &pars) {
                resize(GAUSS_NGAUSS);
                _set_from_pars_fp(pars, GAUSS_NGAUSS, GAUSS_PVALS, GAUSS_FVALS);
            }

    }; // class GMixGauss


    class GMixExp : public GMixSimple {
    
        public:
            GMixExp() {
                npars_=SIMPLE_NPARS;
                resize(EXP_NGAUSS);
            }
            GMixExp(const vector<double> &pars) {
                npars_=SIMPLE_NPARS;
                set_from_pars(pars);
            }
            virtual void set_from_pars(const vector<double> &pars) {
                resize(EXP_NGAUSS);
                _set_from_pars_fp(pars, EXP_NGAUSS, EXP_PVALS, EXP_FVALS);
            }

    }; // class GMixExp

    class GMixDev : public GMixSimple {
    
        public:
            GMixDev() {
                npars_=SIMPLE_NPARS;
                resize(DEV_NGAUSS);
            }
            GMixDev(const vector<double> &pars) {
                npars_=SIMPLE_NPARS;
                set_from_pars(pars);
            }
            virtual void set_from_pars(const vector<double> &pars) {
                resize(DEV_NGAUSS);
                _set_from_pars_fp(pars, DEV_NGAUSS, DEV_PVALS, DEV_FVALS);
            }

    }; // class GMixDev

    class GMixTurb : public GMixSimple {
    
        public:
            GMixTurb() {
                npars_=SIMPLE_NPARS;
                resize(TURB_NGAUSS);
            }
            GMixTurb(const vector<double> &pars) {
                npars_=SIMPLE_NPARS;
                set_from_pars(pars);
            }
            virtual void set_from_pars(const vector<double> &pars) {
                resize(TURB_NGAUSS);
                _set_from_pars_fp(pars, TURB_NGAUSS, TURB_PVALS, TURB_FVALS);
            }

    }; // class GMixTurb


}

#endif
