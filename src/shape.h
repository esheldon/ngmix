/*
    Struct type representing a shape
*/
#ifndef _SHAPE_HEADER_GUARD
#define _SHAPE_HEADER_GUARD

#include <stdexcept>
#include <cmath>
#include <cstdio>
#include <string>
#include <iostream>

namespace shape {

    struct Shape {

        public:

            // the data is public
            double g1;
            double g2;

            Shape() {g1=0; g2=0;}

            Shape(double g1in, double g2in) {
                set_g(g1in, g2in);
            }

            void set_g(double g1in, double g2in) {
                double gsq = g1in*g1in + g2in*g2in;
                if (gsq >= 1.0) {
                    throw std::out_of_range ("g >= 1");
                }
                g1=g1in;
                g2=g2in;
            }

            // shear the shape
            void shear(Shape s) {
                shear(s.g1, s.g2);
            }

            void shear(double s1, double s2) {
                double A = 1. + g1*s1 + g2*s2;
                double B = g2*s1 - g1*s2;
                double denom_inv = 1./(A*A + B*B);

                double g1o = A*(g1 + s1) + B*(g2 + s2);
                double g2o = A*(g2 + s2) - B*(g1 + s1);

                g1o *= denom_inv;
                g2o *= denom_inv;

                set_g(g1o, g2o);
            }

            void rotate(double theta_radians)
            {
                double twotheta = 2*theta_radians;

                // note trig is always done double
                double cos2angle = std::cos(twotheta);
                double sin2angle = std::sin(twotheta);
                double g1rot =  g1*cos2angle + g2*sin2angle;
                double g2rot = -g1*sin2angle + g2*cos2angle;

                set_g(g1rot, g2rot);
            }

            double dgs_by_dgo_jacob(const Shape& shear) {

                double ssq = shear.g1*shear.g1 + shear.g2*shear.g2;
                double num = (ssq - 1)*(ssq - 1);
                double denom=(1 
                         + 2*g1*shear.g1
                         + 2*g2*shear.g2
                         + g1*g1*ssq
                         + g2*g2*ssq);
                denom *= denom;

                return num/denom;

            }

            void show() {
                std::printf("%.16g %.16g\n", g1, g2);
            }
    };




} // namespace shape

#endif
