/*

   A very simple image class.  It just has a fixed sized
   buffer, determined at compile time.  The dimensions of
   the image represent a view into the memory.

*/
#ifndef _SIMPLE_IMAGE_HEADER_GUARD
#define _SIMPLE_IMAGE_HEADER_GUARD

#include <cstring>
#include <stdexcept>

namespace simple_image {

    static const long MAXSIZE=256;

    class SimpleImage {

        public:

            // this is public, but be careful
            double data[MAXSIZE][MAXSIZE];

            SimpleImage() {
                _nrows=0;
                _ncols=0;
            }

            SimpleImage(long nrows, long ncols) {
                resize(nrows, ncols);
                zero();
            }


            // getter
            inline double operator()(long row, long col) const {
                if (row < 0 || row >= _nrows 
                    || col < 0 || col >= _ncols) {
                    throw std::runtime_error("out of bounds");
                }
                return data[row][col];
            }
            // setter
            inline double& operator()(long row, long col) {
                if (row < 0 || row >= _nrows 
                    || col < 0 || col >= _ncols) {
                    throw std::runtime_error("out of bounds");
                }
                return data[row][col];
            }


            long get_nrows() {
                return _nrows;
            }
            long get_ncols() {
                return _ncols;
            }


            void add_constant(double value) {
                for (long row=0; row<_nrows; row++) {
                    for (long col=0; col<_ncols; col++) {
                        data[row][col] += value;
                    }
                }
            }

            void zero() {
                std::memset(data, 0, MAXSIZE*MAXSIZE);
            }
            void clear() {
                zero();
                _nrows=0;
                _ncols=0;
            }

            // if larger the data might be junk
            void resize(long nrows, long ncols) {
                if (nrows < 0
                        || nrows > MAXSIZE
                        || ncols < 0
                        || ncols > MAXSIZE) {
                    throw std::runtime_error("dimensions too big");
                }

                _nrows=nrows;
                _ncols=ncols;
            }

        private:

            long _nrows;
            long _ncols;

    };

} // namespace simple_image

#endif
