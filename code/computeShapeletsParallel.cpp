#include <Rcpp.h>
// [[Rcpp::depends(RcppParallel)]]
#include <RcppParallel.h>

#define EPS	1E-9

using namespace RcppParallel;

typedef struct _distToShapelet {
	int classOf;
	double distance;
} distToShapelet;

int compare(const void *x, const void *y) {
	distToShapelet *a = (distToShapelet *)x;
	distToShapelet *b = (distToShapelet *)y;
	
	if (a->distance < b->distance) {
		return -1;
	}
	return 1;
}

struct ShapeletsWorker : public Worker {
	const RMatrix<double> input;
	const RVector<int> classes;
	RMatrix<double> output;
	int ncol, nrow;
	int nClasses;
	int min, max;
	int flag;
	int *classesCount;
	double entropy;
	int *indexes;
	
	// Main constructor
	ShapeletsWorker(const Rcpp::NumericMatrix input, 
			   		const Rcpp::IntegerVector classes,
			   		Rcpp::NumericMatrix output,
			   		int ncol,
			   		int nrow,
					int nClasses,
					int min,
					int max,
					int flag) : 
			   		input(input), 
			   		classes(classes),
			   		output(output), 
			   		ncol(ncol),
			   		nrow(nrow),
					nClasses(nClasses),
					min(min),
					max(max),
					flag(flag) {
						if (flag & 1 || flag & 2) {
							countClasses();
						}
						if (flag & 1) {
							initialEntropy();
						}
						computeIndexes();
					}
					
	// Constructor used by the sub-class that only computes the shapelets
	// of a given list.
	ShapeletsWorker(const Rcpp::NumericMatrix input,
					const Rcpp::IntegerVector classes,
					Rcpp::NumericMatrix output,
					int ncol,
					int nrow,
					int nClasses,
					int min,
					int flag) :
					input(input),
					classes(classes),
					output(output),
					ncol(ncol),
					nrow(nrow),
					nClasses(nClasses),
					min(min),
					flag(flag) {
						if (flag & 1 || flag & 2) {
							countClasses();
						}
						if (flag & 1) {
							initialEntropy();
						}
					}
	
	void countClasses() {
		// Count the amount of instances of each class on the input.
		classesCount = (int *) calloc(nClasses + 1, sizeof(int));
		for (int i = 0; i < nrow; i++) {
			classesCount[classes[i]]++;
		}
	}
	
	void initialEntropy() {
		// Computes the initial entropy of the data set (input).
		entropy = 0.0;
		for (int i = 1; i <= nClasses; i++) {
			double proportion = classesCount[i] / (double)nrow;
			entropy -= (proportion) * log2(proportion + EPS);
		}
	}

	void computeIndexes() {
		// indexes indicates the rows on the output matrix that each lenght of
		// shapelet takes - this is used to avoid critical regions.
		int count = 0;
		indexes = (int *) calloc(max, sizeof(int));
		for (int i = min; i < max; i++) {
			indexes[i] = count;
			count += (ncol - i + 1) * nrow;
		}
	}
					   
	// Computes the in-class transitions quality measure
	int inClassTransitions(distToShapelet *distances) {
		int count = 0;
		for (int i = 1; i < nrow; i++) {
			if (distances[i - 1].classOf == distances[i].classOf) {
				count++;
			}
		}
		
		return count;
	}

	// Computes the f-statistic quality measure
	double fStatistic(distToShapelet *distances) {
		double *means = (double *) calloc(nClasses + 1, sizeof(double));
		for (int i = 0; i < nrow; i++) {
			means[0] += distances[i].distance;
			means[distances[i].classOf] += distances[i].distance;
		}
		
		for (int i = 1; i <= nClasses; i++) {
			means[i] /= (double)classesCount[i];
		}
		means[0] /= (double)nrow;
		
		double numerator = 0.0;
		for (int i = 1; i <= nClasses; i++) {
			numerator += (means[i] - means[0]) * (means[i] - means[0]);
		}
		numerator /= (nClasses - 1.0);
		
		double denominator = 0.0;
		for (int i = 0; i < nrow; i++) {
			double tmp = distances[i].distance - means[distances[i].classOf];
			denominator += tmp * tmp;
		}
		denominator /= (double)(nrow - nClasses);
		
		free(means);
		return numerator / (denominator + EPS);
	}
	
	// Computes the information-gain quality measure
	double informationGain(distToShapelet *distances) {
		double informationGain = 0.0;
		double count = 0;
		double leftEntropy, rightEntropy;
		int *currClassesCount = (int *) calloc(nClasses + 1, sizeof(int));
		
		// tests all possible splits (splitting before the first element operator
		// after the last one yields an information gain of 0).
		for (int split = 0; split < nrow - 1; split++) {
			count++;
			currClassesCount[distances[split].classOf]++;
			
			leftEntropy = rightEntropy = 0.0;
			for (int i = 1; i <= nClasses; i++) {
				double proportion = currClassesCount[i] / count;
				leftEntropy -= proportion * log2(proportion + EPS);
				
				proportion = (classesCount[i] - currClassesCount[i]) / (double)(nrow - count);
				rightEntropy -= proportion * log2(proportion + EPS);
			}
			double tmp = entropy;
			tmp -= (count / nrow) * leftEntropy + ((nrow - count) / nrow) * rightEntropy;
			
			if (tmp > informationGain) {
				informationGain = tmp;
			}
		}
		
		free(currClassesCount);
		return informationGain;
	}

	// Z-normalizes the array.
	void znormalization(double *a, int length) {
		double mean = 0.0;
		for (int i = 0; i < length; i++) {
			mean += a[i];
		}
		mean /= length;
		
		double std = 0.0;
		for (int i = 0; i < length; i++) {
			std += (a[i] - mean) * (a[i] - mean);
		}
		std = sqrt(std / (length - 1.0) + EPS);
		
		for (int i = 0; i < length; i++) {
			a[i] = (a[i] - mean) / std;
		}
	}
	
	// Computes the distance of a shapelet to the time series by sliding the
	// shapelet along the time series.
	double distFromShapeletToTimeSeries(double *shapelet,
										double *subsequence, 
										int row, 
										int length) {
		int timeSeriesLength = ncol;
		
		// the sum of X and X^2 are used to speed up the computation of the mean and
		// standard deviation required for the z-normalization.
		double *sumX = (double *)calloc(timeSeriesLength + 1, sizeof(double));
		double *sumX2 = (double *)calloc(timeSeriesLength + 1, sizeof(double));
		for (int i = 0; i < timeSeriesLength; i++) {
			sumX[i + 1] = sumX[i] + input(row, i);
			sumX2[i + 1] = sumX2[i] + input(row, i) * input(row, i); 
		}
		
		double minDistance = -1.0;
		for (int i = 0; i <= timeSeriesLength - length; i++) {
			for (int j = 0; j < length; j++) {
				subsequence[j] = input(row, i + j);
			}
			
			double diffSumX = sumX[i + length] - sumX[i];
			double mean = diffSumX / (double)length;
			double var = ((sumX2[i + length] - sumX2[i]) - diffSumX * diffSumX / (double)length) / (length - 1.0);
			double std = sqrt(var + EPS);

			// euclidean distance
			double distance = 0.0;
			for (int j = 0; j < length; j++) {
				double tmp = shapelet[j] - (subsequence[j] - mean) / std;
				distance += tmp * tmp;
			}
			distance = sqrt(distance);
												
			if (minDistance < 0.0 || distance < minDistance) {
				minDistance = distance;
			}
		}
		
		free(sumX);
		free(sumX2);
		
		return minDistance;
	}
	
	// Computes the shapelets of the given length [begin, end).
	void operator()(std::size_t begin, std::size_t end) {
		double *shapelet = (double *) calloc(end - 1, sizeof(double));
		double *subsequence = (double *) calloc(end - 1, sizeof(double));
		distToShapelet *distances = (distToShapelet *) calloc(nrow, sizeof(distToShapelet));
		
		int offset = indexes[begin];
		for (int length = begin; length < end; length++) {
			for (int row = 0; row < nrow; row++) {
				for (int col = 0; col <= ncol - length; col++) {
					for (int i = 0; i < length; i++) {
						shapelet[i] = input(row, col + i);
					}
					znormalization(shapelet, length);
					
					for (int i = 0; i < nrow; i++) {
						distances[i].classOf = classes[i];
						distances[i].distance = distFromShapeletToTimeSeries(shapelet,
																			 subsequence,
																			 i,
																			 length);
					}
					if (flag & 1 || flag & 4) {
						qsort(distances, nrow, sizeof(distToShapelet), compare);
					}
					
					output(offset, 0) = row + 1;
					output(offset, 1) = col + 1;
					output(offset, 2) = col + length;
					output(offset, 3) = length;
					output(offset, 4) = classes[row];
					output(offset, 5) = flag & 1 ? informationGain(distances) : 0.0;
					output(offset, 6) = flag & 2 ? fStatistic(distances) : 0.0;
					output(offset, 7) = flag & 4 ? inClassTransitions(distances) : 0.0;
					offset++;
				}
			}
		}
		
		free(shapelet);
		free(subsequence);
		free(distances);
	}
};

struct ShapeletsWorkerPartial : public ShapeletsWorker {
	const RVector<int> sampled;

	ShapeletsWorkerPartial(const Rcpp::NumericMatrix input,
							const Rcpp::IntegerVector classes,
							Rcpp::NumericMatrix output,
							int ncol,
							int nrow,
							int nClasses,
							int min,
							const Rcpp::IntegerVector sampled,
							int flag) :
							sampled(sampled),
							ShapeletsWorker(input,
											classes,
											output,
											ncol,
											nrow,
											nClasses,
											min,
											flag) {}
	
	// Computes the shapelets of the given length [begin, end).
	void operator()(std::size_t begin, std::size_t end) {
		double *shapelet = (double *) calloc(ncol, sizeof(double));
		double *subsequence = (double *) calloc(ncol, sizeof(double));
		distToShapelet *distances = (distToShapelet *) calloc(nrow, sizeof(distToShapelet));
	
		// computes the shapelets given a list of the ones that were sampled.
		for (int offset = (int)begin; offset < (int)end; offset++) {
			int idx = sampled[offset];
			int accum = 0;
		
			// determines the length of shapelet given its index.
			int length = min;
			while (accum + (ncol - length + 1) * nrow <= idx) {
				accum += (ncol - length + 1) * nrow;
				length++;
			}
		
			// determines from which time series (row) the shapelet was
			// extracted given its index.
			int row = 0;
			while (accum + (ncol - length + 1) <= idx) {
				accum += (ncol - length + 1);
				row++;
			}
		
			// determines the position within the time series that the
			// shapelet was extracted given its index.
			int col = 0;
			while (accum + col + 1 <= idx) {
				col++;
			}
		
			// extracts the shapelet.
			for (int i = 0; i < length; i++) {
				shapelet[i] = input(row, col + i);
			}
			znormalization(shapelet, length);
		
			for (int i = 0; i < nrow; i++) {
				distances[i].classOf = classes[i];
				distances[i].distance = distFromShapeletToTimeSeries(shapelet,
																	 subsequence,
																	 i,
																	 length);
			}
			if (flag & 1 || flag & 4) {
				qsort(distances, nrow, sizeof(distToShapelet), compare);
			}
		
			output(offset, 0) = row + 1;
			output(offset, 1) = col + 1;
			output(offset, 2) = col + length;
			output(offset, 3) = length;
			output(offset, 4) = classes[row];
			output(offset, 5) = flag & 1 ? informationGain(distances) : 0.0;
			output(offset, 6) = flag & 2 ? fStatistic(distances) : 0.0;
			output(offset, 7) = flag & 4 ? inClassTransitions(distances) : 0.0;
		}
	
		free(shapelet);
		free(subsequence);
		free(distances);
	}
};

// [[Rcpp::export]]
Rcpp::DataFrame computeShapeletsParallelCpp(Rcpp::NumericMatrix input, 
											 Rcpp::IntegerVector classes,
											 int nClasses,
											 int min,
											 int max,
											 int flag) {
	
	// max is incremented because the program works on the range [min, max)
	max++;
	
	// count the amount of shapelets to be computed.
	int count = 0;
	for (int i = min; i < max; i++) {
		count += (input.ncol() - i + 1) * input.nrow();
	}
	
	/*	This matrix stores the output of the computation of the shapelets. It
	*	has the total amount of shapelets rows and 8 columns:
	*		- the time series row/id from which the shapelet was extracted;
	*		- its beginning position at the original time series;
	*		- its ending position at the original time series;
	*		- its length;
	*		- its class (the class of the original time series);
	*		- its information gain quality measure;
	*		- its f-statistic quality measure;
	*		- its in-class-transitions quality measure.
	*/	
	Rcpp::NumericMatrix output(count, 8);
	
	// Computes the shapelets.
	ShapeletsWorker shapeletsWorker(input, 
									classes, 
									output,
									input.ncol(),
									input.nrow(),
									nClasses,
									min,
									max,
									flag);

	parallelFor(min, max, shapeletsWorker);
	
	// Returns the results in a data frame.
	Rcpp::DataFrame df =
		Rcpp::DataFrame::create(Rcpp::Named("time.series.id") = output(Rcpp::_ , 0),
								Rcpp::Named("begin") = output(Rcpp::_ , 1),
								Rcpp::Named("end") = output(Rcpp::_ , 2),
								Rcpp::Named("length") = output(Rcpp::_ , 3),
								Rcpp::Named("class") = output(Rcpp::_ , 4),
								Rcpp::Named("information.gain") = output(Rcpp::_ , 5),
								Rcpp::Named("f.statistic") = output(Rcpp::_ , 6),
								Rcpp::Named("in.class.transitions") = output(Rcpp::_ , 7));
	
	return df;
}

// [[Rcpp::export]]
Rcpp::DataFrame computeShapeletsParallelPartialCpp(Rcpp::NumericMatrix input,
													Rcpp::IntegerVector classes,
													Rcpp::IntegerVector sampled,
													int nClasses,
													int min,
													int flag) {
	
	/*	This matrix stores the output of the computation of the shapelets. It
	*	has the amount of sampled shapelets rows and 8 columns:
	*		- the time series row/id from which the shapelet was extracted;
	*		- its beginning position at the original time series;
	*		- its ending position at the original time series;
	*		- its length;
	*		- its class (the class of the original time series);
	*		- its information gain quality measure;
	*		- its f-statistic quality measure;
	*		- its in-class-transitions quality measure.
	*/	
	Rcpp::NumericMatrix output(sampled.length(), 8);
	
	ShapeletsWorkerPartial shapeletsWorkerPartial(input, 
													classes, 
													output,
													input.ncol(),
													input.nrow(),
													nClasses,
													min,
													sampled,
													flag);

	parallelFor(0, sampled.length(), shapeletsWorkerPartial);
	
	// Returns the results in a data frame.
	Rcpp::DataFrame df =
		Rcpp::DataFrame::create(Rcpp::Named("time.series.id") = output(Rcpp::_ , 0),
								Rcpp::Named("begin") = output(Rcpp::_ , 1),
								Rcpp::Named("end") = output(Rcpp::_ , 2),
								Rcpp::Named("length") = output(Rcpp::_ , 3),
								Rcpp::Named("class") = output(Rcpp::_ , 4),
								Rcpp::Named("information.gain") = output(Rcpp::_ , 5),
								Rcpp::Named("f.statistic") = output(Rcpp::_ , 6),
								Rcpp::Named("in.class.transitions") = output(Rcpp::_ , 7));
	
	return df;
}