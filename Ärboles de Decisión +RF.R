# Importar librerías
library(caret)
library(pROC)
library(rpart)

# Importar la base de datos
data <- read.csv("C:/Users/afroo/OneDrive/Escritorio/IA/thyroid_edit_equal.csv")

# Inspección y preprocesamiento de datos
str(data)
summary(data)
colSums(is.na(data) | data == "")

# Transformación de variables
data$on_thyroxine <- factor(data$on_thyroxine, levels = c("f", "t"), labels = c(0, 1))
data$query_on_thyroxine <- factor(data$query_on_thyroxine, levels = c("f", "t"), labels = c(0, 1))
data$on_antithyroid_medication <- factor(data$on_antithyroid_medication, levels = c("f", "t"), labels = c(0, 1))
data$sick <- factor(data$sick, levels = c("f", "t"), labels = c(0, 1))
data$pregnant <- factor(data$pregnant, levels = c("f", "t"), labels = c(0, 1))
data$thyroid_surgery <- factor(data$thyroid_surgery, levels = c("f", "t"), labels = c(0, 1))
data$I131_treatment <- factor(data$I131_treatment, levels = c("f", "t"), labels = c(0, 1))
data$query_hypothyroid <- factor(data$query_hypothyroid, levels = c("f", "t"), labels = c(0, 1))
data$query_hyperthyroid <- factor(data$query_hyperthyroid, levels = c("f", "t"), labels = c(0, 1))
data$lithium <- factor(data$lithium, levels = c("f", "t"), labels = c(0, 1))
data$goitre <- factor(data$goitre, levels = c("f", "t"), labels = c(0, 1))
data$tumor <- factor(data$tumor, levels = c("f", "t"), labels = c(0, 1))
data$hypopituitary <- factor(data$hypopituitary, levels = c("f", "t"), labels = c(0, 1))
data$psych <- factor(data$psych, levels = c("f", "t"), labels = c(0, 1))
data$TSH_measured <- factor(data$TSH_measured, levels = c("f", "t"), labels = c(0, 1))
data$T3_measured <- factor(data$T3_measured, levels = c("f", "t"), labels = c(0, 1))
data$TT4_measured <- factor(data$TT4_measured, levels = c("f", "t"), labels = c(0, 1))
data$FTI_measured <- factor(data$FTI_measured, levels = c("f", "t"), labels = c(0, 1))
data$TBG_measured <- factor(data$TBG_measured, levels = c("f", "t"), labels = c(0, 1))
data$referral_source <- factor(data$referral_source, levels = c("other", "STMW", "SVHC", "SVHD", "SVI"), labels = c(0, 1, 2, 3, 4))
data$Class <- factor(data$Class, levels = c("negative", "sick"), labels = c(0, 1))
data$T4U_measured <- factor(data$T4U_measured, levels = c("f", "t"), labels = c(0, 1))
str(data)
colSums(is.na(data) | data == "")

# Dividir en conjuntos de entrenamiento y prueba
set.seed(123)
dat_partition <- createDataPartition(data$Class, p = 0.8, list = FALSE)
traindata <- data[dat_partition, ]
testdata <- data[-dat_partition, ]

# Configuración de parámetros para validación cruzada
ctr <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final"
)

# Renombrar niveles de la variable objetivo
levels(traindata$Class) <- c("negative", "sick")
levels(testdata$Class) <- c("negative", "sick")

# Entrenamiento del modelo de Árbol de Decisión
set.seed(888)
tree_model <- train(
  Class ~ ., 
  data = traindata, 
  method = "rpart", 
  trControl = ctr, 
  metric = "ROC",
  tuneLength = 10
)

# Evaluación en cada pliegue de entrenamiento
predictions <- tree_model$pred
resamples <- unique(predictions$Resample)
auc_train <- numeric(length(resamples))

cat("\nResultados por pliegue de entrenamiento:\n")
par(mfrow = c(2, 3))

for (i in seq_along(resamples)) {
  fold_pred <- predictions[predictions$Resample == resamples[i], ]
  roc_curve <- roc(response = fold_pred$obs, predictor = fold_pred$sick, levels = rev(levels(fold_pred$obs)))
  auc_train[i] <- auc(roc_curve)
  plot(roc_curve, col = "green", main = paste("FOLD ROC ~ TESTING SET", resamples[i], ")", auc_train[i]))
  cat("Pliegue:", resamples[i], "- AUC:", auc_train[i], "\n")
}

# Evaluación en el conjunto de prueba
test_predictions <- predict(tree_model, testdata, type = "prob")
roc_test <- roc(response = testdata$Class, predictor = test_predictions$sick, levels = rev(levels(testdata$Class)))
auc_test <- auc(roc_test)

# Resultados finales
cat("\nAUC en el conjunto de prueba:", auc_test, "\n")
plot(roc_test, col = "green", main = paste("ROC ~ TEST SET (AUC:", round(auc_test, 3), ")"))

# Predicciones en el conjunto de prueba
tree_pred <- predict(tree_model, testdata)

# Matriz de confusión para evaluar el rendimiento
cm_tree <- confusionMatrix(tree_pred, testdata$Class)
print(cm_tree)
