### House Price prediction@Kaggle: https://www.kaggle.com/c/house-prices-advanced-regression-techniques
# Problem: How do home features add up to its price tag?

# 0. clear the workspace
rm(list=ls())
# 1. Read in the data
library(readr)
train<-read.csv("data/train.csv")
test<-read.csv("data/test.csv")
dim(train) # 1460 81
dim(test) # 1460 80

# Data Preprocessing steps
# Recoding variables with NA values that are actually data points as highlighted by
# reference: http://stackoverflow.com/questions/19379081/how-to-replace-na-values-in-a-table-for-selected-columns-data-frame-data-tab

# 2. Recoding the train data
train[c("Alley")][is.na(train[c("Alley")])] <- "NoAlleyAccess"
train[c("BsmtQual")][is.na(train[c("BsmtQual")])] <- "NoBasement"
train[c("BsmtCond")][is.na(train[c("BsmtCond")])] <- "NoBasement"
train[c("BsmtExposure")][is.na(train[c("BsmtExposure")])] <- "NoBasement"
train[c("BsmtFinType1")][is.na(train[c("BsmtFinType1")])] <- "NoBasement"
train[c("BsmtFinType2")][is.na(train[c("BsmtFinType2")])] <- "NoBasement"
train[c("GarageType")][is.na(train[c("GarageType")])] <- "NoGarage"
train[c("GarageFinish")][is.na(train[c("GarageFinish")])] <- "NoGarage"
train[c("GarageQual")][is.na(train[c("GarageQual")])] <- "NoGarage"
train[c("GarageCond")][is.na(train[c("GarageCond")])] <- "NoGarage"
train[c("PoolQC")][is.na(train[c("PoolQC")])] <- "NoPool"
train[c("Fence")][is.na(train[c("Fence")])] <- "NoFence"
train[c("MiscFeature")][is.na(train[c("MiscFeature")])] <- "None"
train[c("FireplaceQu")][is.na(train[c("FireplaceQu")])] <- "NoFireplace"

# 2.a Renaming the incorrectly labelled identifiers
train$firstFloorSqft<-train$`1stFlrSF` # incorrect named identifier
train$secndFloorSqft<-train$`2ndFlrSF` # incorrect named identifier
train$`1stFlrSF`<-NULL
train$`2ndFlrSF`<-NULL
sum(is.na(train)) # 357 missing
#colSums(is.na(train)) # all categorical NA recoded accordingly as they were not missing according to the data dictionary

# 3. Recoding the test data
test[c("Alley")][is.na(test[c("Alley")])] <- "NoAlleyAccess"
test[c("BsmtQual")][is.na(test[c("BsmtQual")])] <- "NoBasement"
test[c("BsmtCond")][is.na(test[c("BsmtCond")])] <- "NoBasement"
test[c("BsmtExposure")][is.na(test[c("BsmtExposure")])] <- "NoBasement"
test[c("BsmtFinType1")][is.na(test[c("BsmtFinType1")])] <- "NoBasement"
test[c("BsmtFinType2")][is.na(test[c("BsmtFinType2")])] <- "NoBasement"
test[c("GarageType")][is.na(test[c("GarageType")])] <- "NoGarage"
test[c("GarageFinish")][is.na(test[c("GarageFinish")])] <- "NoGarage"
test[c("GarageCond")][is.na(test[c("GarageCond")])] <- "NoGarage"
test[c("GarageQual")][is.na(test[c("GarageQual")])] <- "NoGarage"
test[c("PoolQC")][is.na(test[c("PoolQC")])] <- "NoPool"
test[c("Fence")][is.na(test[c("Fence")])] <- "NoFence"
test[c("MiscFeature")][is.na(test[c("MiscFeature")])] <- "None"
test[c("FireplaceQu")][is.na(test[c("FireplaceQu")])] <- "NoFireplace"
test$firstFloorSqft<-test$`1stFlrSF` # incorrect named identifier
test$secndFloorSqft<-test$`2ndFlrSF` # incorrect named identifier
test$`1stFlrSF`<-NULL
test$`2ndFlrSF`<-NULL
sum(is.na(test)) # 358 missing values

#4. Missing value treatment
## Train data
set.seed(1234)
library(mice)
tempData <- mice(train,m=5,maxit=50,meth='pmm',seed=500)
## Note on the parameters. m=5 refers to the number of imputed datasets. Five is the default value.
## meth='pmm' refers to the imputation method. In this case we are using predictive mean matching as imputation method. Other imputation methods can be used, type methods(mice) for a list of the available imputation methods.
train.complete<-complete(tempData,1) # where 1 referes to the first dataset out of five generated above

## Test data
tempData <- mice(test,m=5,maxit=50,meth='pmm',seed=500)
test.complete<-complete(tempData,1) # where 1 referes to the first dataset out of five generated above

#5. Check for missing data after preprocessing in both train and test dataset
sum(is.na(train.complete)) #9
sum(is.na(test.complete)) #28

#6. Check for zero variance predictors in train data
library(caret) # load the caret library for it has the nearZeroVar()
nzv_cols <- nearZeroVar(train.complete)
names(train.complete[nzv_cols])
if(length(nzv_cols) > 0) 
  train.complete <- train.complete[, -nzv_cols] # 1460 60
dim(train.complete)
#7. Check for zero variance predictors in test data
nzv_cols <- nearZeroVar(test.complete)
names(test[nzv_cols])
if(length(nzv_cols) > 0) 
  test.complete <- test.complete[, -nzv_cols] # 1459 61
dim(test.complete)
## Missing data treatment: There are two types of missing data.a) MCAR (Missing Completetly At Random) & (b) MNAR (Missing Not At Random)
## Usually, MCAR is the desirable scenario in case of missing data. For this analysis I will assume that MCAR is at play. 
## Assuming data is MCAR, too much missing data can be a problem too. Usually a safe maximum threshold is 5% of the total for large datasets. If missing data for a certain feature or sample is more than 5% then you probably should leave that feature or sample out. We therefore check for features (columns) and samples (rows) where more than 5% of the data is missing using a simple function

# 8. Visualizing the missing data pattern using the VIM package
library(VIM)
aggr_plot <- aggr(train.complete, col=c('navyblue','red'), numbers=TRUE, sortVars=TRUE, labels=names(train.complete), cex.axis=.7, gap=3, ylab=c("Histogram of missing data","Pattern"))
aggr_plot <- aggr(test.complete, col=c('navyblue','red'), numbers=TRUE, sortVars=TRUE, labels=names(test.complete), cex.axis=.7, gap=3, ylab=c("Histogram of missing data","Pattern"))
## Some great references
# http://stackoverflow.com/questions/4862178/remove-rows-with-nas-missing-values-in-data-frame
# http://stackoverflow.com/questions/4605206/drop-data-frame-columns-by-name


# 9. DUMMY CODING FOR CHARACTER VARIABLES in TRAIN DATASET
# http://stats.stackexchange.com/questions/94010/understanding-dummy-manual-or-automated-variable-creation-in-glm
# http://www.ats.ucla.edu/stat/r/library/contrast_coding.htm
# https://www.r-bloggers.com/quickly-create-dummy-variables-in-a-data-frame/
str(train)
train.complete$housezone[train.complete$MSZoning %in% c("FV")] <- 4
train.complete$housezone[train.complete$MSZoning %in% c("RL")] <- 3
train.complete$housezone[train.complete$MSZoning %in% c("RH","RM")] <- 2
train.complete$housezone[train.complete$MSZoning %in% c("C (all)")] <- 1
train.complete$houselotshape[train.complete$LotShape =="Reg"]<-1
train.complete$houselotshape[train.complete$LotShape != "Reg"]<-0
train.complete$houselotconfg[train.complete$LotConfig %in% c("Inside")] <- 3
train.complete$houselotconfg[train.complete$LotConfig %in% c("Corner")] <- 2
train.complete$houselotconfg[train.complete$LotConfig %in% c("CulDSac","FR2","FR3")] <- 1
train.complete$houseneighbrhd<-ifelse(train.complete$Neighborhood=="Blmngtn",1,
                             ifelse(train.complete$Neighborhood=="Blueste",2,
                                    ifelse(train.complete$Neighborhood=="BrDale",3,
                                           ifelse(train.complete$Neighborhood=="BrkSide",4,
                                                  ifelse(train.complete$Neighborhood=="ClearCr",5,
                                                         ifelse(train.complete$Neighborhood=="CollgCr",6,
                                                                ifelse(train.complete$Neighborhood=="Crawfor",7,
                                                                       ifelse(train.complete$Neighborhood=="Edwards",8,
                                                                              ifelse(train.complete$Neighborhood=="Gilbert",9,
                                                                                     ifelse(train.complete$Neighborhood=="IDOTRR",10,
                                                                                            ifelse(train.complete$Neighborhood=="MeadowV",11,
                                                                                                   ifelse(train.complete$Neighborhood=="Mitchel",12,
                                                                                                          ifelse(train.complete$Neighborhood=="NAmes",13,
                                                                                                                 ifelse(train.complete$Neighborhood=="NoRidge",14,
                                                                                                                        ifelse(train.complete$Neighborhood=="NPkVill",15,
                                                                                                                               ifelse(train.complete$Neighborhood=="NridgHt",16,
                                                                                                                                      ifelse(train.complete$Neighborhood=="NWAmes",17,
                                                                                                                                             ifelse(train.complete$Neighborhood=="OldTown",18,
                                                                                                                                                    ifelse(train.complete$Neighborhood=="Sawyer",19,
                                                                                                                                                           ifelse(train.complete$Neighborhood=="SawyerW",20,
                                                                                                                                                                  ifelse(train.complete$Neighborhood=="Somerst",21,
                                                                                                                                                                         ifelse(train.complete$Neighborhood=="StoneBr",22,
                                                                                                                                                                                ifelse(train.complete$Neighborhood=="SWISU",23,
                                                                                                                                                                                       ifelse(train.complete$Neighborhood=="Timber",24,
                                                                                                                                                                                              ifelse(train.complete$Neighborhood=="Veenker",25,-1)))))))))))))))))))))))))

train.complete$housenearto<- ifelse(train.complete$Condition1=="Artery",1,
                            ifelse(train.complete$Condition1=="Feedr",2,
                                   ifelse(train.complete$Condition1=="Norm",3,
                                          ifelse(train.complete$Condition1=="PosA",4,
                                                 ifelse(train.complete$Condition1=="PosN",5,
                                                        ifelse(train.complete$Condition1=="RRAe",6,
                                                               ifelse(train.complete$Condition1=="RRAn",7,
                                                                      ifelse(train.complete$Condition1=="RRNe",8,
                                                                             ifelse(train.complete$Condition1=="RRNn",9,-1)))))))))
train.complete$housebldgtype <- ifelse(train.complete$BldgType=="1Fam",1,
                           ifelse(train.complete$BldgType=="2fmCon",2,
                                  ifelse(train.complete$BldgType=="Duplex",3,
                                         ifelse(train.complete$BldgType=="Twnhs",4,
                                                ifelse(train.complete$BldgType=="TwnhsE",5,-1)))))

train.complete$housetyletype<-ifelse(train.complete$HouseStyle=="1.5Fin",1,
                           ifelse(train.complete$HouseStyle=="1.5Unf",2,
                                  ifelse(train.complete$HouseStyle=="1Story",3,
                                         ifelse(train.complete$HouseStyle=="2.5Fin",4,
                                                ifelse(train.complete$HouseStyle=="2.5Unf",5,
                                                       ifelse(train.complete$HouseStyle=="2Story",6,
                                                              ifelse(train.complete$HouseStyle=="SFoyer",7,
                                                                     ifelse(train.complete$HouseStyle=="SLvl",8,-1))))))))


train.complete$houserooftype<-ifelse(train.complete$RoofStyle=="Flat",1,
                          ifelse(train.complete$RoofStyle=="Gable",2,
                                 ifelse(train.complete$RoofStyle=="Gambrel",3,
                                        ifelse(train.complete$RoofStyle=="Hip",4,
                                               ifelse(train.complete$RoofStyle=="Mansard",5,
                                                      ifelse(train.complete$RoofStyle=="Shed",6,-1))))))


train.complete$housecovertype.1<-ifelse(train.complete$Exterior1st=="AsbShng",1,
                            ifelse(train.complete$Exterior1st=="AsphShn",2,
                                   ifelse(train.complete$Exterior1st=="BrkComm",3,
                                          ifelse(train.complete$Exterior1st=="BrkFace",4,
                                                 ifelse(train.complete$Exterior1st=="CBlock",5,
                                                        ifelse(train.complete$Exterior1st=="CemntBd",6,
                                                               ifelse(train.complete$Exterior1st=="HdBoard",7,
                                                                      ifelse(train.complete$Exterior1st=="ImStucc",8,
                                                                             ifelse(train.complete$Exterior1st=="MetalSd",9,
                                                                                    ifelse(train.complete$Exterior1st=="Plywood",10,
                                                                                           ifelse(train.complete$Exterior1st=="Stone",11,
                                                                                                  ifelse(train.complete$Exterior1st=="Stucco",12,
                                                                                                         ifelse(train.complete$Exterior1st=="VinylSd",13,
                                                                                                                ifelse(train.complete$Exterior1st=="Wd Sdng",14,
                                                                                                                       ifelse(train.complete$Exterior1st=="WdShing",15,-1)))))))))))))))

train.complete$housecovertype.2<-ifelse(train.complete$Exterior2nd=="AsbShng",1,
                            ifelse(train.complete$Exterior2nd=="AsphShn",2,
                                   ifelse(train.complete$Exterior2nd=="Brk Cmn",3,
                                          ifelse(train.complete$Exterior2nd=="BrkFace",4,
                                                 ifelse(train.complete$Exterior2nd=="CBlock",5,
                                                        ifelse(train.complete$Exterior2nd=="CmentBd",6,
                                                               ifelse(train.complete$Exterior2nd=="HdBoard",7,
                                                                      ifelse(train.complete$Exterior2nd=="ImStucc",8,
                                                                             ifelse(train.complete$Exterior2nd=="MetalSd",9,
                                                                                    ifelse(train.complete$Exterior2nd=="Other",10,
                                                                                           ifelse(train.complete$Exterior2nd=="Plywood",11,
                                                                                                  ifelse(train.complete$Exterior2nd=="Stone",12,
                                                                                                         ifelse(train.complete$Exterior2nd=="Stucco",13,
                                                                                                                ifelse(train.complete$Exterior2nd=="VinylSd",14,
                                                                                                                       ifelse(train.complete$Exterior2nd=="Wd Sdng",15,
                                                                                                                              ifelse(train.complete$Exterior2nd=="Wd Shng",16,-1))))))))))))))))
train.complete$housemasonrytype<- ifelse(train.complete$MasVnrType=="BrkCmn",1,
                            ifelse(train.complete$MasVnrType=="BrkFace",2,
                                   ifelse(train.complete$MasVnrType=="character",3,
                                          ifelse(train.complete$MasVnrType=="None",4,
                                                 ifelse(train.complete$MasVnrType=="Stone",5,-1)))))

train.complete$housematqual<-ifelse(train.complete$ExterQual=="Ex",1,
                          ifelse(train.complete$ExterQual=="Fa",2,
                                 ifelse(train.complete$ExterQual=="Gd",3,
                                        ifelse(train.complete$ExterQual=="TA",4,-1))))

train.complete$housematcond<-ifelse(train.complete$ExterCond=="Ex",1,
                          ifelse(train.complete$ExterCond=="Fa",2,
                                 ifelse(train.complete$ExterCond=="Gd",3,
                                        ifelse(train.complete$ExterCond=="TA",4,
                                               ifelse(train.complete$ExterCond=="Po",5,-1)))))

train.complete$housefoundtype<-ifelse(train.complete$Foundation=="BrkTil",1,
                           ifelse(train.complete$Foundation=="CBlock",2,
                                  ifelse(train.complete$Foundation=="PConc",3,
                                         ifelse(train.complete$Foundation=="Slab",4,
                                                ifelse(train.complete$Foundation=="Stone",5,
                                                       ifelse(train.complete$Foundation=="Wood",6,-1))))))

train.complete$housebsmtheight<-ifelse(train.complete$BsmtQual=="NoBasement",5,
                         ifelse(train.complete$BsmtQual=="TA",1,
                                ifelse(train.complete$BsmtQual=="Gd",2,
                                       ifelse(train.complete$BsmtQual=="Ex",3,
                                              ifelse(train.complete$BsmtQual=="Fa",4,-1)))))

train.complete$housebsmtexpose<-ifelse(train.complete$BsmtExposure=="Av",1,
                             ifelse(train.complete$BsmtExposure=="NoBasement",2,
                                    ifelse(train.complete$BsmtExposure=="Gd",3,
                                           ifelse(train.complete$BsmtExposure=="Mn",4,
                                                  ifelse(train.complete$BsmtExposure=="No",5,-1)))))

train.complete$housebsmtrating<-ifelse(train.complete$BsmtFinType1=="ALQ",1,
                             ifelse(train.complete$BsmtFinType1=="BLQ",2,
                                    ifelse(train.complete$BsmtFinType1=="GLQ",3,
                                           ifelse(train.complete$BsmtFinType1=="LwQ",4,
                                                  ifelse(train.complete$BsmtFinType1=="Rec",5,
                                                         ifelse(train.complete$BsmtFinType1=="Unf",6,
                                                                ifelse(train.complete$BsmtFinType1=="NoBasement",7,-1)))))))

train.complete$househeatqual<-ifelse(train.complete$HeatingQC=="Ex",1,
                          ifelse(train.complete$HeatingQC=="Fa",2,
                                 ifelse(train.complete$HeatingQC=="Gd",3,
                                        ifelse(train.complete$HeatingQC=="TA",4,
                                               ifelse(train.complete$HeatingQC=="Po",5,-1)))))

train.complete$housecentrac<-ifelse(train.complete$CentralAir=="Y",1,0) # 1= Yes, central ac 0= No, central ac

train.complete$houselectric<-ifelse(train.complete$Electrical=="character",0,
                           ifelse(train.complete$Electrical=="FuseA",1,
                                  ifelse(train.complete$Electrical=="FuseF",2,
                                         ifelse(train.complete$Electrical=="FuseP",3,
                                                ifelse(train.complete$Electrical=="Mix",4,
                                                       ifelse(train.complete$Electrical=="SBrkr",5,-1))))))

train.complete$housekitchqual<-ifelse(train.complete$KitchenQual=="Ex",1,
                            ifelse(train.complete$KitchenQual=="Fa",2,
                                   ifelse(train.complete$KitchenQual=="Gd",3,
                                          ifelse(train.complete$KitchenQual=="TA",4,-1))))

train.complete$housefireplcqual<-ifelse(train.complete$FireplaceQu=="NoFireplace",6,
                            ifelse(train.complete$FireplaceQu=="Ex",1,
                                   ifelse(train.complete$FireplaceQu=="Fa",2,
                                          ifelse(train.complete$FireplaceQu=="Gd",3,
                                                 ifelse(train.complete$FireplaceQu=="TA",4,
                                                        ifelse(train.complete$FireplaceQu=="Po",5,-1))))))

train.complete$housegarageloc<-ifelse(train.complete$GarageType=="2Types",1,
                           ifelse(train.complete$GarageType=="Attchd",2,
                                  ifelse(train.complete$GarageType=="Basment",3,
                                         ifelse(train.complete$GarageType=="BuiltIn",4,
                                                ifelse(train.complete$GarageType=="CarPort",5,
                                                       ifelse(train.complete$GarageType=="Detchd",6,
                                                              ifelse(train.complete$GarageType=="NoGarage",7,-1)))))))

train.complete$housegarageinterior<-ifelse(train.complete$GarageFinish=="Fin",1,
                             ifelse(train.complete$GarageFinish=="RFn",2,
                                    ifelse(train.complete$GarageFinish=="Unf",3,
                                           ifelse(train.complete$GarageFinish=="NoGarage",4,-1))))

train.complete$housegaragequal<-ifelse(train.complete$GarageQual=="Ex",1,
                               ifelse(train.complete$GarageQual=="Gd",2,
                                      ifelse(train.complete$GarageQual=="Fa",3,
                                             ifelse(train.complete$GarageQual=="Po",4,
                                                    ifelse(train.complete$GarageQual=="TA",5,
                                                           ifelse(train.complete$GarageQual=="NoGarage",6,-1))))))

train.complete$housegaragecond<-ifelse(train.complete$GarageCond=="Ex",1,
                              ifelse(train.complete$GarageCond=="Gd",2,
                                     ifelse(train.complete$GarageCond=="Fa",3,
                                            ifelse(train.complete$GarageCond=="Po",4,
                                                   ifelse(train.complete$GarageCond=="TA",5,
                                                          ifelse(train.complete$GarageCond=="NoGarage",6,-1))))))

train.complete$housedrivewaycond[train.complete$PavedDrive %in% c("P")] <- 3
train.complete$housedrivewaycond[train.complete$PavedDrive %in% c("N")] <- 2
train.complete$housedrivewaycond[train.complete$PavedDrive %in% c("Y")] <- 1

train.complete$housefencequal[train.complete$Fence %in% c("GdPrv","GdWo")] <- 3
train.complete$housefencequal[train.complete$Fence %in% c("MnPrv","MnWw")] <- 2
train.complete$housefencequal[train.complete$Fence %in% c("NoFence")] <- 1

train.complete$housesaletype[train.complete$SaleType %in% c("COD")] <- 5
train.complete$housesaletype[train.complete$SaleType %in% c("CWD","WD")] <- 4
train.complete$housesaletype[train.complete$SaleType %in% c("Con","ConLD","ConLI","ConLw")] <- 3
train.complete$housesaletype[train.complete$SaleType %in% c("Oth")] <- 2
train.complete$housesaletype[train.complete$SaleType %in% c("New")] <- 1

train.complete$housesalecond[train.complete$SaleCondition %in% c("Normal")] <- 5
train.complete$housesalecond[train.complete$SaleCondition %in% c("Abnorml")] <- 4
train.complete$housesalecond[train.complete$SaleCondition %in% c("AdjLand","Alloca")] <- 3
train.complete$housesalecond[train.complete$SaleCondition %in% c("Family")] <- 2
train.complete$housesalecond[train.complete$SaleCondition %in% c("Partial")] <- 1

# 10.  Drop all the character variables
train.complete$MSZoning<-NULL
train.complete$LotShape<-NULL
train.complete$LotConfig<-NULL
train.complete$Neighborhood<-NULL
train.complete$Condition1<-NULL
train.complete$BldgType<-NULL
train.complete$HouseStyle<-NULL
train.complete$RoofStyle<-NULL
train.complete$Exterior1st<-NULL
train.complete$Exterior2nd<-NULL
train.complete$Foundation<-NULL
train.complete$BsmtQual<-NULL
train.complete$BsmtExposure<-NULL
train.complete$BsmtFinType1<-NULL
train.complete$HeatingQC<-NULL
train.complete$CentralAir<-NULL
train.complete$Electrical<-NULL
train.complete$KitchenQual<-NULL
train.complete$FireplaceQu<-NULL
train.complete$GarageType<-NULL
train.complete$GarageFinish<-NULL
train.complete$GarageQual<-NULL
train.complete$GarageCond<-NULL
train.complete$PavedDrive<-NULL
train.complete$Fence<-NULL
train.complete$SaleType<-NULL
train.complete$SaleCondition<-NULL
train.complete$MasVnrType<-NULL
train.complete$ExterQual<-NULL
train.complete$ExterCond<-NULL
train.complete$LandContour<-NULL

str(train.complete)
sum(is.na(train.complete))
colSums(is.na(train.complete))
# 11. Fix some NA's in train.complete
train.complete$houselectric[is.na(train.complete$houselectric)]<- -1
train.complete$housemasonrytype[is.na(train.complete$housemasonrytype)]<- -1
sum(is.na(train.complete)) # No missing values is train.complete

# 12. DUMMY CODING FOR CHARACTER VARIABLES in TEST DATA
test.complete$housezone[test.complete$MSZoning %in% c("FV")] <- 4
test.complete$housezone[test.complete$MSZoning %in% c("RL")] <- 3
test.complete$housezone[test.complete$MSZoning %in% c("RH","RM")] <- 2
test.complete$housezone[test.complete$MSZoning %in% c("C (all)")] <- 1

test.complete$houselotshape[test.complete$LotShape =="Reg"]<-1
test.complete$houselotshape[test.complete$LotShape != "Reg"]<-0

test.complete$houselotconfg[test.complete$LotConfig %in% c("Inside")] <- 3
test.complete$houselotconfg[test.complete$LotConfig %in% c("Corner")] <- 2
test.complete$houselotconfg[test.complete$LotConfig %in% c("CulDSac","FR2","FR3")] <- 1

test.complete$houseneighbrhd<-ifelse(test.complete$Neighborhood=="Blmngtn",1,
                                     ifelse(test.complete$Neighborhood=="Blueste",2,
                                            ifelse(test.complete$Neighborhood=="BrDale",3,
                                                   ifelse(test.complete$Neighborhood=="BrkSide",4,
                                                          ifelse(test.complete$Neighborhood=="ClearCr",5,
                                                                 ifelse(test.complete$Neighborhood=="CollgCr",6,
                                                                        ifelse(test.complete$Neighborhood=="Crawfor",7,
                                                                               ifelse(test.complete$Neighborhood=="Edwards",8,
                                                                                      ifelse(test.complete$Neighborhood=="Gilbert",9,
                                                                                             ifelse(test.complete$Neighborhood=="IDOTRR",10,
                                                                                                    ifelse(test.complete$Neighborhood=="MeadowV",11,
                                                                                                           ifelse(test.complete$Neighborhood=="Mitchel",12,
                                                                                                                  ifelse(test.complete$Neighborhood=="NAmes",13,
                                                                                                                         ifelse(test.complete$Neighborhood=="NoRidge",14,
                                                                                                                                ifelse(test.complete$Neighborhood=="NPkVill",15,
                                                                                                                                       ifelse(test.complete$Neighborhood=="NridgHt",16,
                                                                                                                                              ifelse(test.complete$Neighborhood=="NWAmes",17,
                                                                                                                                                     ifelse(test.complete$Neighborhood=="OldTown",18,
                                                                                                                                                            ifelse(test.complete$Neighborhood=="Sawyer",19,
                                                                                                                                                                   ifelse(test.complete$Neighborhood=="SawyerW",20,
                                                                                                                                                                          ifelse(test.complete$Neighborhood=="Somerst",21,
                                                                                                                                                                                 ifelse(test.complete$Neighborhood=="StoneBr",22,
                                                                                                                                                                                        ifelse(test.complete$Neighborhood=="SWISU",23,
                                                                                                                                                                                               ifelse(test.complete$Neighborhood=="Timber",24,
                                                                                                                                                                                                      ifelse(test.complete$Neighborhood=="Veenker",25,-1)))))))))))))))))))))))))

test.complete$housenearto<- ifelse(test.complete$Condition1=="Artery",1,
                                   ifelse(test.complete$Condition1=="Feedr",2,
                                          ifelse(test.complete$Condition1=="Norm",3,
                                                 ifelse(test.complete$Condition1=="PosA",4,
                                                        ifelse(test.complete$Condition1=="PosN",5,
                                                               ifelse(test.complete$Condition1=="RRAe",6,
                                                                      ifelse(test.complete$Condition1=="RRAn",7,
                                                                             ifelse(test.complete$Condition1=="RRNe",8,
                                                                                    ifelse(test.complete$Condition1=="RRNn",9,-1)))))))))
test.complete$housebldgtype <- ifelse(test.complete$BldgType=="1Fam",1,
                                      ifelse(test.complete$BldgType=="2fmCon",2,
                                             ifelse(test.complete$BldgType=="Duplex",3,
                                                    ifelse(test.complete$BldgType=="Twnhs",4,
                                                           ifelse(test.complete$BldgType=="TwnhsE",5,-1)))))

test.complete$housetyletype<-ifelse(test.complete$HouseStyle=="1.5Fin",1,
                                    ifelse(test.complete$HouseStyle=="1.5Unf",2,
                                           ifelse(test.complete$HouseStyle=="1Story",3,
                                                  ifelse(test.complete$HouseStyle=="2.5Fin",4,
                                                         ifelse(test.complete$HouseStyle=="2.5Unf",5,
                                                                ifelse(test.complete$HouseStyle=="2Story",6,
                                                                       ifelse(test.complete$HouseStyle=="SFoyer",7,
                                                                              ifelse(test.complete$HouseStyle=="SLvl",8,-1))))))))

test.complete$houserooftype<-ifelse(test.complete$RoofStyle=="Flat",1,
                                    ifelse(test.complete$RoofStyle=="Gable",2,
                                           ifelse(test.complete$RoofStyle=="Gambrel",3,
                                                  ifelse(test.complete$RoofStyle=="Hip",4,
                                                         ifelse(test.complete$RoofStyle=="Mansard",5,
                                                                ifelse(test.complete$RoofStyle=="Shed",6,-1))))))
test.complete$housecovertype.1<-ifelse(test.complete$Exterior1st=="AsbShng",1,
                                       ifelse(test.complete$Exterior1st=="AsphShn",2,
                                              ifelse(test.complete$Exterior1st=="BrkComm",3,
                                                     ifelse(test.complete$Exterior1st=="BrkFace",4,
                                                            ifelse(test.complete$Exterior1st=="CBlock",5,
                                                                   ifelse(test.complete$Exterior1st=="CemntBd",6,
                                                                          ifelse(test.complete$Exterior1st=="HdBoard",7,
                                                                                 ifelse(test.complete$Exterior1st=="ImStucc",8,
                                                                                        ifelse(test.complete$Exterior1st=="MetalSd",9,
                                                                                               ifelse(test.complete$Exterior1st=="Plywood",10,
                                                                                                      ifelse(test.complete$Exterior1st=="Stone",11,
                                                                                                             ifelse(test.complete$Exterior1st=="Stucco",12,
                                                                                                                    ifelse(test.complete$Exterior1st=="VinylSd",13,
                                                                                                                           ifelse(test.complete$Exterior1st=="Wd Sdng",14,
                                                                                                                                  ifelse(test.complete$Exterior1st=="WdShing",15,-1)))))))))))))))
test.complete$housecovertype.2<-ifelse(test.complete$Exterior2nd=="AsbShng",1,
                                       ifelse(test.complete$Exterior2nd=="AsphShn",2,
                                              ifelse(test.complete$Exterior2nd=="Brk Cmn",3,
                                                     ifelse(test.complete$Exterior2nd=="BrkFace",4,
                                                            ifelse(test.complete$Exterior2nd=="CBlock",5,
                                                                   ifelse(test.complete$Exterior2nd=="CmentBd",6,
                                                                          ifelse(test.complete$Exterior2nd=="HdBoard",7,
                                                                                 ifelse(test.complete$Exterior2nd=="ImStucc",8,
                                                                                        ifelse(test.complete$Exterior2nd=="MetalSd",9,
                                                                                               ifelse(test.complete$Exterior2nd=="Other",10,
                                                                                                      ifelse(test.complete$Exterior2nd=="Plywood",11,
                                                                                                             ifelse(test.complete$Exterior2nd=="Stone",12,
                                                                                                                    ifelse(test.complete$Exterior2nd=="Stucco",13,
                                                                                                                           ifelse(test.complete$Exterior2nd=="VinylSd",14,
                                                                                                                                  ifelse(test.complete$Exterior2nd=="Wd Sdng",15,
                                                                                                                                         ifelse(test.complete$Exterior2nd=="Wd Shng",16,-1))))))))))))))))
test.complete$housemasonrytype<- ifelse(test.complete$MasVnrType=="BrkCmn",1,
                                        ifelse(test.complete$MasVnrType=="BrkFace",2,
                                               ifelse(test.complete$MasVnrType=="character",3,
                                                      ifelse(test.complete$MasVnrType=="None",4,
                                                             ifelse(test.complete$MasVnrType=="Stone",5,-1)))))
test.complete$housematqual<-ifelse(test.complete$ExterQual=="Ex",1,
                                   ifelse(test.complete$ExterQual=="Fa",2,
                                          ifelse(test.complete$ExterQual=="Gd",3,
                                                 ifelse(test.complete$ExterQual=="TA",4,-1))))
test.complete$housematcond<-ifelse(test.complete$ExterCond=="Ex",1,
                                   ifelse(test.complete$ExterCond=="Fa",2,
                                          ifelse(test.complete$ExterCond=="Gd",3,
                                                 ifelse(test.complete$ExterCond=="TA",4,
                                                        ifelse(test.complete$ExterCond=="Po",5,-1)))))
test.complete$housefoundtype<-ifelse(test.complete$Foundation=="BrkTil",1,
                                     ifelse(test.complete$Foundation=="CBlock",2,
                                            ifelse(test.complete$Foundation=="PConc",3,
                                                   ifelse(test.complete$Foundation=="Slab",4,
                                                          ifelse(test.complete$Foundation=="Stone",5,
                                                                 ifelse(test.complete$Foundation=="Wood",6,-1))))))
test.complete$housebsmtheight<-ifelse(test.complete$BsmtQual=="NoBasement",5,
                                      ifelse(test.complete$BsmtQual=="TA",1,
                                             ifelse(test.complete$BsmtQual=="Gd",2,
                                                    ifelse(test.complete$BsmtQual=="Ex",3,
                                                           ifelse(test.complete$BsmtQual=="Fa",4,-1)))))
test.complete$housebsmtexpose<-ifelse(test.complete$BsmtExposure=="Av",1,
                                      ifelse(test.complete$BsmtExposure=="NoBasement",2,
                                             ifelse(test.complete$BsmtExposure=="Gd",3,
                                                    ifelse(test.complete$BsmtExposure=="Mn",4,
                                                           ifelse(test.complete$BsmtExposure=="No",5,-1)))))
test.complete$housebsmtrating<-ifelse(test.complete$BsmtFinType1=="ALQ",1,
                                      ifelse(test.complete$BsmtFinType1=="BLQ",2,
                                             ifelse(test.complete$BsmtFinType1=="GLQ",3,
                                                    ifelse(test.complete$BsmtFinType1=="LwQ",4,
                                                           ifelse(test.complete$BsmtFinType1=="Rec",5,
                                                                  ifelse(test.complete$BsmtFinType1=="Unf",6,
                                                                         ifelse(test.complete$BsmtFinType1=="NoBasement",7,-1)))))))
test.complete$househeatqual<-ifelse(test.complete$HeatingQC=="Ex",1,
                                    ifelse(test.complete$HeatingQC=="Fa",2,
                                           ifelse(test.complete$HeatingQC=="Gd",3,
                                                  ifelse(test.complete$HeatingQC=="TA",4,
                                                         ifelse(test.complete$HeatingQC=="Po",5,-1)))))

test.complete$housecentrac<-ifelse(test.complete$CentralAir=="Y",1,0) # 1= Yes, central ac 0= No, central ac

test.complete$houselectric<-ifelse(test.complete$Electrical=="character",0,
                                   ifelse(test.complete$Electrical=="FuseA",1,
                                          ifelse(test.complete$Electrical=="FuseF",2,
                                                 ifelse(test.complete$Electrical=="FuseP",3,
                                                        ifelse(test.complete$Electrical=="Mix",4,
                                                               ifelse(test.complete$Electrical=="SBrkr",5,-1))))))
test.complete$housekitchqual<-ifelse(test.complete$KitchenQual=="Ex",1,
                                     ifelse(test.complete$KitchenQual=="Fa",2,
                                            ifelse(test.complete$KitchenQual=="Gd",3,
                                                   ifelse(test.complete$KitchenQual=="TA",4,-1))))

test.complete$housefireplcqual<-ifelse(test.complete$FireplaceQu=="NoFireplace",6,
                                       ifelse(test.complete$FireplaceQu=="Ex",1,
                                              ifelse(test.complete$FireplaceQu=="Fa",2,
                                                     ifelse(test.complete$FireplaceQu=="Gd",3,
                                                            ifelse(test.complete$FireplaceQu=="TA",4,
                                                                   ifelse(test.complete$FireplaceQu=="Po",5,-1))))))

test.complete$housegarageloc<-ifelse(test.complete$GarageType=="2Types",1,
                                     ifelse(test.complete$GarageType=="Attchd",2,
                                            ifelse(test.complete$GarageType=="Basment",3,
                                                   ifelse(test.complete$GarageType=="BuiltIn",4,
                                                          ifelse(test.complete$GarageType=="CarPort",5,
                                                                 ifelse(test.complete$GarageType=="Detchd",6,
                                                                        ifelse(test.complete$GarageType=="NoGarage",7,-1)))))))
test.complete$housegarageinterior<-ifelse(test.complete$GarageFinish=="Fin",1,
                                          ifelse(test.complete$GarageFinish=="RFn",2,
                                                 ifelse(test.complete$GarageFinish=="Unf",3,
                                                        ifelse(test.complete$GarageFinish=="NoGarage",4,-1))))
test.complete$housegaragequal<-ifelse(test.complete$GarageQual=="Ex",1,
                                      ifelse(test.complete$GarageQual=="Gd",2,
                                             ifelse(test.complete$GarageQual=="Fa",3,
                                                    ifelse(test.complete$GarageQual=="Po",4,
                                                           ifelse(test.complete$GarageQual=="TA",5,
                                                                  ifelse(test.complete$GarageQual=="NoGarage",6,-1))))))
test.complete$housegaragecond<-ifelse(test.complete$GarageCond=="Ex",1,
                                      ifelse(test.complete$GarageCond=="Gd",2,
                                             ifelse(test.complete$GarageCond=="Fa",3,
                                                    ifelse(test.complete$GarageCond=="Po",4,
                                                           ifelse(test.complete$GarageCond=="TA",5,
                                                                  ifelse(test.complete$GarageCond=="NoGarage",6,-1))))))
test.complete$housedrivewaycond[test.complete$PavedDrive %in% c("P")] <- 3
test.complete$housedrivewaycond[test.complete$PavedDrive %in% c("N")] <- 2
test.complete$housedrivewaycond[test.complete$PavedDrive %in% c("Y")] <- 1

test.complete$housefencequal[test.complete$Fence %in% c("GdPrv","GdWo")] <- 3
test.complete$housefencequal[test.complete$Fence %in% c("MnPrv","MnWw")] <- 2
test.complete$housefencequal[test.complete$Fence %in% c("NoFence")] <- 1

test.complete$housesaletype[test.complete$SaleType %in% c("COD")] <- 5
test.complete$housesaletype[test.complete$SaleType %in% c("CWD","WD")] <- 4
test.complete$housesaletype[test.complete$SaleType %in% c("Con","ConLD","ConLI","ConLw")] <- 3
test.complete$housesaletype[test.complete$SaleType %in% c("Oth")] <- 2
test.complete$housesaletype[test.complete$SaleType %in% c("New")] <- 1

test.complete$housesalecond[test.complete$SaleCondition %in% c("Normal")] <- 5
test.complete$housesalecond[test.complete$SaleCondition %in% c("Abnorml")] <- 4
test.complete$housesalecond[test.complete$SaleCondition %in% c("AdjLand","Alloca")] <- 3
test.complete$housesalecond[test.complete$SaleCondition %in% c("Family")] <- 2
test.complete$housesalecond[test.complete$SaleCondition %in% c("Partial")] <- 1

#13. Drop all the character variables in Test data
test.complete$MSZoning<-NULL
test.complete$LotShape<-NULL
test.complete$LotConfig<-NULL
test.complete$Neighborhood<-NULL
test.complete$Condition1<-NULL
test.complete$BldgType<-NULL
test.complete$HouseStyle<-NULL
test.complete$RoofStyle<-NULL
test.complete$Exterior1st<-NULL
test.complete$Exterior2nd<-NULL
test.complete$Foundation<-NULL
test.complete$BsmtQual<-NULL
test.complete$BsmtExposure<-NULL
test.complete$BsmtFinType1<-NULL
test.complete$HeatingQC<-NULL
test.complete$CentralAir<-NULL
test.complete$Electrical<-NULL
test.complete$KitchenQual<-NULL
test.complete$FireplaceQu<-NULL
test.complete$GarageType<-NULL
test.complete$GarageFinish<-NULL
test.complete$GarageQual<-NULL
test.complete$GarageCond<-NULL
test.complete$PavedDrive<-NULL
test.complete$Fence<-NULL
test.complete$SaleType<-NULL
test.complete$SaleCondition<-NULL
test.complete$MasVnrType<-NULL
test.complete$ExterQual<-NULL
test.complete$ExterCond<-NULL
test.complete$LandContour<-NULL

# 14. Fix some NA's in test.complete data
str(test.complete)
sum(is.na(test.complete))
colSums(is.na(test.complete))
# coding the NA's as -1
test.complete$housekitchqual[is.na(test.complete$housekitchqual)]<- -1
test.complete$housesaletype[is.na(test.complete$housesaletype)]<- -1
test.complete$housecovertype.1[is.na(test.complete$housecovertype.1)]<- -1
test.complete$housecovertype.2[is.na(test.complete$housecovertype.2)]<- -1
test.complete$housemasonrytype[is.na(test.complete$housemasonrytype)]<- -1
test.complete$housezone[is.na(test.complete$housezone)]<- -1
sum(is.na(test.complete))

