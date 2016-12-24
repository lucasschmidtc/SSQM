require("Rcpp")
require("RcppParallel")
sourceCpp("./code/computeShapeletsParallel.cpp")
sourceCpp("./code/shapeletsToFeatures.cpp")

# Given a data set (data) this function computes the given proportion
# of shapelets (p) qualities (quality) between the length range
# of min and max. All of the computation is performed in parallel by
# the use of the RcppParallel package. Details of the parameters:
#	data: a data set that contains the time series and its classes
#			labels.
#	p: the proportion of shapelets to be computed. This value must be
#		between 0 and 1. If p is less than one, the shapelets are
#		choosen randomly.
#	min, max: the length range of the shapelets. If any of them are
#				NULL then they are set to 3 and the length of the
#				time series.
#	quality: determines which shapelets qualities measurements are to
#				be computed. It can be: information.gain, f.statistic,
#				in.class.transitions and all.
# The output of this function is a data frame containing the information
# related to the shapelets. 
computeShapelets <- function(data,
								p = 1, 
								min = NULL, 
								max = NULL, 
								quality = "all") {
											
	train <- as.matrix(data[ , -which(names(data) %in% c("id", "class"))])
	classLevels <- levels(data$class)
	cl <- as.integer(data$class)
	nClasses <- max(cl)
	
	flag <- 0
	if ("all" %in% quality) {
		flag <- 7
	}
	else {
		if ("information.gain" %in% quality) {
			flag <- flag + 1
		}
		if ("f.statistic" %in% quality) {
			flag <- flag + 2
		}
		if ("in.class.transitions" %in% quality) {
			flag <- flag + 4
		}
	}
	
	if (is.null(min) || is.null(max)) {
		min <- 3
		max <- dim(train)[2]
	}
	
	if (p == 1) {
		df <- computeShapeletsParallelCpp(train, cl, nClasses, min, max, flag)
	}
	else {
		total.shapelets <- sum(dim(train)[2] - (min:max) + 1) * dim(train)[1]
		sampled <- sample(1:total.shapelets, 
							size = total.shapelets * p, 
							replace = FALSE)
							
		df <- computeShapeletsParallelPartialCpp(train, 
													cl, 
													sampled - 1, 
													nClasses, 
													min, 
													flag)
	}

	df$time.series.id <- as.integer(df$time.series.id)
	df$begin <- as.integer(df$begin)
	df$end <- as.integer(df$end)
	df$length <- as.integer(df$length)
	df$class <- factor(classLevels[as.integer(df$class)], levels = classLevels)
	df
}

# Given a set of shapelets, selects the k best that have no overlapping
# according to the given quality measurement. Details of the parameters:
#	shapelets: a data frame of shapelets.
#	k: the amount of shapelets to be selected. Note that there is no
#		guarantee that k are going to be selected (due to the no
#		overlapping restriction).
#	quality: according to what quality measurement should the shapelets
#				be selected.
# The output of this function are the k best shapelets according to a
# quality measurement that have no overlapping.
shapeletsSelection <- function(shapelets, k, quality) {
	ans <- data.frame()
	
	if (quality %in% c("information.gain", "f.statistic", "in.class.transitions")) {
		quality <- which(names(shapelets) == quality)
		shapelets <- shapelets[order(shapelets[ , quality], decreasing = TRUE), ]

		for (i in 1:dim(shapelets)[1]) {
			if (k == 0) break
			if (any(shapelets$time.series.id[i] == ans$time.series.id & !((shapelets$begin[i] > ans$end) | (shapelets$end[i] < ans$begin))) == FALSE) {
				ans <- rbind(ans, shapelets[i, ])
				k <- k - 1
			}
		}
	}
	else {
		warning("Invalid quality measurement.")
	}
	
	ans
}

# Given a set of shapelets, returns the shapelet representation of the train
# and test set. Details of the parameters:
#	shapelets: a data frame of shapelets.
#	train: the train set.
#	test: the test set.
# The output of this function is a data frame of the concatenation of the
# shapelet representation of both the train and test set.
shapeletsToFeatures <- function(shapelets, train, test) {
	shapeletsMatrix <- as.matrix(shapelets[ , c("time.series.id", "begin", "length")])
	
	set <- rbind(train, test)
	set <- as.matrix(set[ , -which(names(set) %in% c("id", "class"))])
	
	featuresMatrix <- shapeletsToFeaturesCpp(shapeletsMatrix, set)
	
	df <- data.frame(featuresMatrix)
	df$class <- as.factor(c(train$class, test$class))
	df$set <- rep(c("train", "test"), times = c(dim(train)[1], dim(test)[1]))
	
	df
}