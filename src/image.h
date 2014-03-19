/*

   A very simple image class.  It just has a fixed sized
   buffer, determined at compile time.  The dimensions of
   the image represent a view into the memory.

*/
#ifndef _SIMPLE_IMAGE_HEADER_GUARD
#define _SIMPLE_IMAGE_HEADER_GUARD

#include <string>
#include <sstream>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include "mtrng.h"

namespace image {

    static const long MAXSIZE=256;

    class Image {

        public:

            // this is public, but be careful
            double data[MAXSIZE][MAXSIZE];

            Image() {
                resize(0,0);
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


            long get_nrows() const {
                return nrows_;
            }
            long get_ncols() const {
                return ncols_;
            }
            long get_size() const {
                return size_;
            }



            void add_scalar(double value) {
                for (long row=0; row<nrows_; row++) {
                    for (long col=0; col<ncols_; col++) {
                        data[row][col] += value;
                    }
                }
            }
            void add_gaussian_noise(mtrng::MtRng64 &rng, double sigma) {
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
                size_=0;
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
                size_ = nrows*ncols;
            }

            void print() {
                for (long row=0; row<nrows_; row++) {
                    for (long col=0; col<ncols_; col++) {
                        std::printf("%12.5g ", data[row][col]);
                    }
                    std::printf("\n");
                }
            }
            void write(std::string fname) {

                std::printf("writing temporary image to: %s\n", fname.c_str());

                FILE *fobj=std::fopen(fname.c_str(),"w");

                std::fprintf(fobj,"%lu %lu\n", nrows_, ncols_);
                for (long row=0; row<nrows_; row++) {
                    for (long col=0; col<ncols_; col++) {
                        std::fprintf(fobj,"%.16g ", data[row][col]);
                    }
                    std::fprintf(fobj,"\n");
                }

                std::fclose(fobj);
            }


            void show(std::string fname)
            {
                std::stringstream cmd;

                write(fname);

                cmd << "image-view -m "<<fname;
                printf("%s\n", cmd.str().c_str() );

                int ret=std::system( cmd.str().c_str() );
                std::printf("ret: %d\n", ret);

                // clear
                cmd.str("");
                cmd << "rm " << fname;

                printf("%s\n", cmd.str().c_str() );
                ret=std::system( cmd.str().c_str() );

                std::printf("ret: %d\n", ret);
            }

        private:

            long nrows_;
            long ncols_;
            long size_;

    };

} // namespace image

#endif
