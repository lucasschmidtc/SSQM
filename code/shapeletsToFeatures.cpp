#include <Rcpp.h>

using namespace Rcpp;

#define EPS	1E-9

// Z-normalizes the array.
void znormalization(double *a, int length) {
	double mean = 0.0;
	for (int i = 0; i < length; i++) {
		mean += a[i];
	}
	mean /= length;
	
	double sd = 0.0;
	for (int i = 0; i < length; i++) {
		sd += (a[i] - mean) * (a[i] - mean);
	}
	sd = sqrt(sd / (length - 1.0));
	
	for (int i = 0; i < length; i++) {
		a[i] = (a[i] - mean) / (sd + EPS);
	}
}

// Computes the euclidean distance.
double euclideanDistance(double *a, double *b, int length) {
	double dist = 0.0;
	for (int i = 0; i < length; i++) {
		dist += (a[i] - b[i]) * (a[i] - b[i]);
	}
	return sqrt(dist);
}

// [[Rcpp::export]]
NumericMatrix shapeletsToFeaturesCpp(IntegerMatrix shapelets,
									 NumericMatrix input) {
	
	NumericMatrix output(input.nrow(), shapelets.nrow());
	
	for (int i = 0; i < shapelets.nrow(); i++) {
		int length = shapelets(i, 2);
		double *shapelet = (double *) calloc(length, sizeof(double));
		for (int j = 0; j < length; j++) {
			shapelet[j] = input(shapelets(i, 0) - 1, j + shapelets(i, 1) - 1);
		}
		znormalization(shapelet, length);
		
		double *subsequence = (double *) calloc(length, sizeof(double));
		for (int j = 0; j < input.nrow(); j++) {
			double minDistance = 0.0;
			for (int k = 0; k <= input.ncol() - length; k++) {
				for (int l = 0; l < length; l++) {
					subsequence[l] = input(j, k + l);
				}
				znormalization(subsequence, length);
				
				double distance = euclideanDistance(shapelet, subsequence, length);
				if (k == 0 || distance < minDistance) {
					minDistance = distance;
				}
			}
			
			output(j, i) = minDistance;
		}
		free(subsequence);
		free(shapelet);
	}
	
	return output;
}