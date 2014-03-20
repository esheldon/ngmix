/*
    Struct type representing a shape
*/
#ifndef _SHAPE_HEADER_GUARD
#define _SHAPE_HEADER_GUARD

#include <stdexcept>
#include <cstdio>
#include <string>
#include <iostream>

// must use math.h to get atanh!
//#include <cmath>
#include <math.h>

namespace NGMix {

    struct Shape {

        public:

            double g1;
            double g2;
            double e1;
            double e2;
            double eta1;
            double eta2;

            Shape() {
                set_g(0.0, 0.0);
            }

            Shape(double g1in, double g2in) {
                set_g(g1in, g2in);
            }

            void set_g(double g1in, double g2in) {

                double g = sqrt(g1in*g1in + g2in*g2in);
                if (g >= 1.) {
                    throw std::out_of_range ("g >= 1");
                }

                double eta = 2*atanh(g);
                double e = tanh(eta);

                if (e >= 1.) {
                    throw std::out_of_range ("e >= 1");
                }

                double cos2theta = g1/g;
                double sin2theta = g2/g;

                this->g1=g1in;
                this->g2=g2in;
                this->e1=e*cos2theta;
                this->e2=e*sin2theta;
                this->eta1=eta*cos2theta;
                this->eta2=eta*sin2theta;
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
                double cos2angle = cos(twotheta);
                double sin2angle = sin(twotheta);
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




} // namespace Shape

#endif
