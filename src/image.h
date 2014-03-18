/*

   A very simple image class.  It just has a fixed sized
   buffer, determined at compile time.  The dimensions of
   the image represent a view into the memory.

*/
#ifndef _SIMPLE_IMAGE_HEADER_GUARD
#define _SIMPLE_IMAGE_HEADER_GUARD

#include <cstring>
#include <stdexcept>
#include "MtRng.h"

namespace image {

    static const long MAXSIZE=256;

    class Image {

        public:

            // this is public, but be careful
            double data[MAXSIZE][MAXSIZE];

            Image() {
                nrows_=0;
                ncols_=0;
            }

            Image(long nrows, long ncols) {
                resize(nrows, ncols);
                zero();
            }


            // getter
            inline double operator()(long row, long col) const {
                if (row < 0 || row >= nrows_ 
                    || col < 0 || col >= ncols_) {
                    throw std::runtime_error("out of bounds");
                }
                return data[row][col];
            }
            // setter
            inline double& operator()(long row, long col) {
                if (row < 0 || row >= nrows_ 
                    || col < 0 || col >= ncols_) {
                    throw std::runtime_error("out of bounds");
                }
                return data[row][col];
            }


            long get_nrows() {
                return nrows_;
            }
            long get_ncols() {
                return ncols_;
            }


            void add_constant(double value) {
                for (long row=0; row<nrows_; row++) {
                    for (long col=0; col<ncols_; col++) {
                        data[row][col] += value;
                    }
                }
            }
            void add_gaussian_noise(MtRng::MtRng64 &rng, double sigma) {
                for (long row=0; row<nrows_; row++) {
                    for (long col=0; col<ncols_; col++) {

                        double r = rng.get_normal();
                        data[row][col] += r*sigma;
                    }
                }
            }

            void zero() {
                std::memset(data, 0, MAXSIZE*MAXSIZE);
            }
            void clear() {
                zero();
                nrows_=0;
                ncols_=0;
            }

            // if larger the data might be junk
            void resize(long nrows, long ncols) {
                if (nrows < 0
                        || nrows > MAXSIZE
                        || ncols < 0
                        || ncols > MAXSIZE) {
                    throw std::runtime_error("dimensions too big");
                }

                nrows_=nrows;
                ncols_=ncols;
            }

            void show() {
                for (long row=0; row<nrows_; row++) {
                    for (long col=0; col<ncols_; col++) {
                        printf("%12.5g ", data[row][col]);
                    }
                    printf("\n");
                }
            }
        private:

            long nrows_;
            long ncols_;

    };

} // namespace image

#endif
