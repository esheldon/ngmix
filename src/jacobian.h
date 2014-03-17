/*

   Jacobian matrix

*/
#ifndef _JACOBIAN_HEADER_GUARD
#define _JACOBIAN_HEADER_GUARD

#include <cmath>

namespace jacobian {


    struct Jacobian {

        public:
            double row0;
            double col0;
            double dudrow;
            double dudcol;
            double dvdrow;
            double dvdcol;
            double det;
            double sdet;

            // init with "unit jacobian" at center 0,0
            Jacobian() {
                set(0.0, 0.0);
            }

            // give center but otherwise unit
            Jacobian(double row0, double col0) {
                // init with "unit jacobian"
                set(row0, col0);
            }

            // full specification
            Jacobian(double row0, double col0,
                     double dudrow, double dudcol,
                     double dvdrow, double dvdcol) {
                // init with "unit jacobian"
                set(row0, col0,
                    dudrow, dudcol,
                    dvdrow, dvdcol);
            }
 
            // set only center, otherwise it is unit
            void set(double row0, double col0) {
                set(row0, col0, 1.0, 0.0, 0.0, 1.0);
            }
            void set(double row0, double col0,
                     double dudrow, double dudcol,
                     double dvdrow, double dvdcol) {

                this->row0=row0;
                this->col0=col0;
                this->dudrow=dudrow;
                this->dudcol=dudcol;
                this->dvdrow=dvdrow;
                this->dvdcol=dvdcol;

                this->det = abs(dudrow*dvdcol-dudcol*dvdrow);
                this->sdet = std::sqrt( this->det );
            }



    };



}

#endif
