library( ANTsR )
library( ANTsRNet )
library( keras )
library( tensorflow )

keras::backend()$clear_session()

Sys.setenv( "CUDA_VISIBLE_DEVICES" = "6,7" )

classes <- c( "background", "leftLung", "rightLung", "trachea" )
numberOfClassificationLabels <- length( classes )

imageMods <- c( "CT" )
channelSize <- length( imageMods )
batchSize <- 8L
segmentationLabels <- c( 0, 1, 2, 3 )

baseDirectory <- '/home/nick/Data/HumanLung/'
scriptsDirectory <- paste0( baseDirectory, 'Scripts/' )
source( paste0( scriptsDirectory, 'unetBatchGenerator.R' ) )

templateDirectory <- paste0( baseDirectory, 'Template/' )
reorientTemplateDirectory <- templateDirectory
reorientTemplate <- antsImageRead( paste0( reorientTemplateDirectory, "T_template0_resampled.nii.gz" ) )

dataDirectories <- c()
dataDirectories <- append( dataDirectories, paste0( baseDirectory, "ResampledVolumes/" ) )

lungImageFiles <- c()
for( i in seq_len( length( dataDirectories ) ) )
  {
  imageFiles <- list.files( path = dataDirectories[i],
    pattern = "1.*.nii.gz", full.names = TRUE, recursive = TRUE )
  lungImageFiles <- append( lungImageFiles, imageFiles )
  }

trainingImageFiles <- list()
trainingSegmentationFiles <- list()
trainingMaskFiles <- list()
trainingTransforms <- list()

missingFiles <- c()

cat( "Loading data...\n" )
pb <- txtProgressBar( min = 0, max = length( lungImageFiles ), style = 3 )

count <- 1
for( i in seq_len( length( lungImageFiles ) ) )
  {
  setTxtProgressBar( pb, i )

  subjectId <- basename( lungImageFiles[i] )
  subjectDirectory <- dirname( lungImageFiles[i] )
  subjectId <- sub( ".nii.gz", '', subjectId )

  lungMaskFile <- paste0( subjectDirectory, "/../ResampledSegmentations/", subjectId, ".nii.gz" )
  lungImageFile <- lungImageFiles[i]

  fwdtransforms <- c()
  invtransforms <- c()
  reorientTransform <- ''

  xfrmPrefix <- paste0( "Tx", subjectId )
  xfrmFiles <- list.files( templateDirectory, pattern = xfrmPrefix, full.names = TRUE )

  fwdtransforms[1] <- xfrmFiles[1]                    # FALSE
  fwdtransforms[2] <- xfrmFiles[3]                    # FALSE

  invtransforms[1] <- xfrmFiles[2]                    # FALSE
  invtransforms[2] <- xfrmFiles[1]                    # TRUE

  missingFile <- FALSE

  if( ! file.exists( lungMaskFile ) )
    {
    missingFile <- TRUE
    }

  for( j in seq_len( length( fwdtransforms ) ) )
    {
    if( !file.exists( invtransforms[j] ) || !file.exists( fwdtransforms[j] ) )
      {
      missingFile <- TRUE
      break
      # stop( paste( "Transform file does not exist.\n" ) )
      }
    }

  if( missingFile )
    {
    missingFiles <- append( missingFiles, subjectDirectory )
    } else {
    trainingTransforms[[count]] <- list(
      fwdtransforms = fwdtransforms, invtransforms = invtransforms )

    trainingImageFiles[[count]] <- lungImageFile
    trainingMaskFiles[[count]] <- lungMaskFile
    count <- count + 1
    }
  }
cat( "\n" )


###
#
# Create the Unet model
#

resampledImageSize <- dim( reorientTemplate )

# See this thread:  https://github.com/rstudio/tensorflow/issues/272

# with( tf$device( "/cpu:0" ), {
unetModel <- createUnetModel3D( c( resampledImageSize, channelSize ),
  numberOfOutputs = numberOfClassificationLabels,
  numberOfLayers = 4, numberOfFiltersAtBaseLayer = 8, dropoutRate = 0.0,
  convolutionKernelSize = c( 3, 3, 3 ), deconvolutionKernelSize = c( 2, 2, 2 ),
  weightDecay = 1e-5 )
  # } )

weightsFile <- paste0( scriptsDirectory, "/humanLungSegmentationWeights.h5" )
if( file.exists( weightsFile ) )
  {
  load_model_weights_hdf5( unetModel, weightsFile )
  }
parallel_unetModel <- unetModel # multi_gpu_model( unetModel, gpus = 4 )

# parallel_unetModel %>% compile( loss = loss_multilabel_dice_coefficient_error,
#   optimizer = optimizer_adam( lr = 0.0001 ),
#   metrics = c( multilabel_dice_coefficient ) )

parallel_unetModel %>% compile( loss = "categorical_crossentropy",
  optimizer = optimizer_adam( lr = 0.00001 ),
  metrics = c( "acc" ) )


###
#
# Set up the training generator
#

# Split trainingData into "training" and "validation" componets for
# training the model.

numberOfData <- length( trainingImageFiles )
sampleIndices <- sample( numberOfData )

cat( "Total number of data = ", numberOfData, "\n" )

validationSplit <- floor( 0.8 * numberOfData )
trainingIndices <- sampleIndices[1:validationSplit]
numberOfTrainingData <- length( trainingIndices )
validationIndices <- sampleIndices[( validationSplit + 1 ):numberOfData]
numberOfValidationData <- length( validationIndices )

###
#
# Run training
#

track <- unetModel %>% fit_generator(
  generator = unetImageBatchGenerator( batchSize = batchSize,
                                       segmentationLabels = segmentationLabels,
                                       doRandomHistogramMatching = FALSE,
                                       reorientImage = reorientTemplate,
                                       sourceImageList = trainingImageFiles[trainingIndices],
                                       segmentationList = trainingMaskFiles[trainingIndices],
                                       sourceTransformList = trainingTransforms[trainingIndices],
                                       referenceImageList = trainingMaskFiles[trainingIndices],
                                       referenceTransformList = trainingTransforms[trainingIndices],
                                       outputFile = paste0( scriptsDirectory, "trainingData.csv" )
                                     ),
  steps_per_epoch = 32L,
  epochs = 75,
  validation_data = unetImageBatchGenerator( batchSize = batchSize,
                                       segmentationLabels = segmentationLabels,
                                       doRandomHistogramMatching = FALSE,
                                       reorientImage = reorientTemplate,
                                       sourceImageList = trainingImageFiles[validationIndices],
                                       segmentationList = trainingMaskFiles[validationIndices],
                                       sourceTransformList = trainingTransforms[validationIndices],
                                       referenceImageList = trainingMaskFiles[validationIndices],
                                       referenceTransformList = trainingTransforms[validationIndices],
                                       outputFile = paste0( scriptsDirectory, "validationData.csv" )
                                     ),
  validation_steps = 16,
  callbacks = list(
    callback_model_checkpoint( weightsFile,
      monitor = 'val_loss', save_best_only = TRUE, save_weights_only = TRUE,
      verbose = 1, mode = 'auto', period = 1 ),
     callback_reduce_lr_on_plateau( monitor = 'val_loss', factor = 0.5,
       verbose = 1, patience = 10, mode = 'auto' ),
     callback_early_stopping( monitor = 'val_loss', min_delta = 0.001,
       patience = 20 )
  )
)

save_model_weights_hdf5( unetModel, weightsFile )

