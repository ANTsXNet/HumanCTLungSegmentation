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
  outputFileName <- args[2]
  reorientTemplateFileName <- args[3]
  }

classes <- c( "background", "leftLung", "rightLung", "trachea" )
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
load_model_weights_hdf5( unetModel, filepath = weightsFileName )
endTime <- Sys.time()
elapsedTime <- endTime - startTime
cat( "  (elapsed time:", elapsedTime, "seconds)\n" )

unetModel %>% compile( loss = loss_multilabel_dice_coefficient_error,
  optimizer = optimizer_adam( lr = 0.0001 ),
  metrics = c( multilabel_dice_coefficient ) )

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

cat( "Renormalize to native space and write to disk." )
startTime <- Sys.time()
probabilityImage <- applyAntsrTransformToImage( invertAntsrTransform( xfrm ),
  probabilityImagesArray[[1]][[2]], image )
antsImageWrite( probabilityImage, paste0( outputFileName, "Probability1.nii.gz" ) )
probabilityImage <- applyAntsrTransformToImage( invertAntsrTransform( xfrm ),
  probabilityImagesArray[[1]][[3]], image )
antsImageWrite( probabilityImage, paste0( outputFileName, "Probability2.nii.gz" ) )
probabilityImage <- applyAntsrTransformToImage( invertAntsrTransform( xfrm ),
  probabilityImagesArray[[1]][[4]], image )
antsImageWrite( probabilityImage, paste0( outputFileName, "Probability3.nii.gz" ) )
endTime <- Sys.time()
elapsedTime <- endTime - startTime
cat( "  (elapsed time:", elapsedTime, "seconds)\n" )

endTimeTotal <- Sys.time()
elapsedTimeTotal <- endTimeTotal - startTimeTotal
cat( "\nTotal elapsed time:", elapsedTimeTotal, "seconds\n\n" )
