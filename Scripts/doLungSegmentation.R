library( ANTsR )
library( ANTsRNet )
library( keras )

args <- commandArgs( trailingOnly = TRUE )

if( length( args ) != 3 )
  {
  helpMessage <- paste0( "Usage:  Rscript doLungSegmentation.R",
    " inputFile outputFilePrefix reorientationTemplate\n" )
  stop( helpMessage )
  } else {
  inputFileName <- args[1]
  outputFilePrefix <- args[2]
  reorientTemplateFileName <- args[3]
  }

classes <- c( "Background", "LeftLung", "RightLung", "Trachea" )
numberOfClassificationLabels <- length( classes )

imageMods <- c( "CT" )
channelSize <- length( imageMods )

cat( "Reading reorientation template", reorientTemplateFileName )
startTime <- Sys.time()
reorientTemplate <- antsImageRead( reorientTemplateFileName )
endTime <- Sys.time()
elapsedTime <- endTime - startTime
cat( "  (elapsed time:", elapsedTime, "seconds)\n" )

resampledImageSize <- dim( reorientTemplate )

unetModel <- createUnetModel3D( c( resampledImageSize, channelSize ),
  numberOfOutputs = numberOfClassificationLabels,
  numberOfLayers = 4, numberOfFiltersAtBaseLayer = 8, dropoutRate = 0.0,
  convolutionKernelSize = c( 3, 3, 3 ), deconvolutionKernelSize = c( 2, 2, 2 ),
  weightDecay = 1e-5 )


cat( "Loading weights file" )
startTime <- Sys.time()
weightsFileName <- paste0( getwd(), "/humanLungSegmentationWeights.h5" )
if( ! file.exists( weightsFileName ) )
  {
  weightsFileName <- getPretrainedNetwork( "ctHumanLung", weightsFileName )
  }
unetModel$load_weights( weightsFileName )
endTime <- Sys.time()
elapsedTime <- endTime - startTime
cat( "  (elapsed time:", elapsedTime, "seconds)\n" )

# Process input

startTimeTotal <- Sys.time()

cat( "Reading ", inputFileName )
startTime <- Sys.time()
image <- antsImageRead( inputFileName, dimension = 3 )
endTime <- Sys.time()
elapsedTime <- endTime - startTime
cat( "  (elapsed time:", elapsedTime, "seconds)\n" )

cat( "Normalizing to template" )
startTime <- Sys.time()
reorientTemplateOnes <- antsImageClone( reorientTemplate ) ^ 0
centerOfMassTemplate <- getCenterOfMass( reorientTemplateOnes )
imageOnes <- antsImageClone( image ) ^ 0
centerOfMassImage <- getCenterOfMass( imageOnes )
xfrm <- createAntsrTransform( type = "Euler3DTransform",
  center = centerOfMassTemplate,
  translation = centerOfMassImage - centerOfMassTemplate )
warpedImage <- applyAntsrTransformToImage( xfrm, image, reorientTemplate )
endTime <- Sys.time()
elapsedTime <- endTime - startTime
cat( "  (elapsed time:", elapsedTime, "seconds)\n" )

warpedArray <- as.array( warpedImage )
warpedArray[which( warpedArray < -1000 )] <- -1000
warpedArray <- ( warpedArray - mean( warpedArray ) ) / sd( warpedArray )

batchX <- array( data = warpedArray,
  dim = c( 1, resampledImageSize, channelSize ) )
batchX <- ( batchX - mean( batchX ) ) / sd( batchX )

cat( "Prediction and decoding" )
startTime <- Sys.time()
predictedData <- unetModel %>% predict( batchX, verbose = 0 )
probabilityImagesArray <- decodeUnet( predictedData, reorientTemplate )
endTime <- Sys.time()
elapsedTime <- endTime - startTime
cat( " (elapsed time:", elapsedTime, "seconds)\n" )


cat( "Renormalize to native space" )
startTime <- Sys.time()

probabilityImages <- list()
mask <- antsImageClone( image ) * 0
for( i in seq_len( numberOfClassificationLabels - 1 ) )
  {
  probabilityImageTmp <- probabilityImagesArray[[1]][[i+1]]
  probabilityImages[[i]] <- applyAntsrTransformToImage( invertAntsrTransform( xfrm ),
    probabilityImageTmp, image )
  mask <- mask + probabilityImages[[i]]
  }
mask <- thresholdImage( mask, 0.5, 10, 1, 0 )

endTime <- Sys.time()
elapsedTime <- endTime - startTime
cat( "  (elapsed time:", elapsedTime, "seconds)\n" )

cat( "Writing", outputFilePrefix )
startTime <- Sys.time()

probabilityImageFiles <- c()
for( i in seq_len( numberOfClassificationLabels - 1 ) )
  {
  probabilityImageFiles[i] <- paste0( outputFilePrefix, classes[i+1], ".nii.gz" )
  antsImageWrite( probabilityImages[[i]], probabilityImageFiles[i] )
  }

probabilityImagesMatrix <- imagesToMatrix( probabilityImageFiles, mask )
segmentationVector <- apply( probabilityImagesMatrix, FUN = which.max, MARGIN = 2 )
segmentationImage <- makeImage( mask, segmentationVector )
antsImageWrite( segmentationImage, paste0( outputFilePrefix, "Segmentation.nii.gz" ) )

endTime <- Sys.time()
elapsedTime <- endTime - startTime
cat( "  (elapsed time:", elapsedTime, "seconds)\n" )

endTimeTotal <- Sys.time()
elapsedTimeTotal <- endTimeTotal - startTimeTotal
cat( "\nTotal elapsed time:", elapsedTimeTotal, "seconds\n\n" )
