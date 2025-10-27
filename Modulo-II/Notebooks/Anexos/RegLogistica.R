# Bibliotecas ================================================================
pacman::p_load(ggplot2, tidyverse, caret, pROC, broom)
set.seed(1989) # Taylor's Version

# Función regresión logística =================================================
regresion_logistica <- function(formula, data, prop = 0.66, p_value = 0.05){
  
  # División train-test ------------------------------------------------------
  if(prop <= 0 | prop >= 1){
    stop("La proporción debe estar entre 0 y 1")
  }
  
  indices <- 1:nrow(data)
  ind_train <- sample(indices, ceiling(prop * length(indices)))
  ind_test <- setdiff(indices, ind_train)
  
  train_set <- data[ind_train,]
  test_set  <- data[ind_test,]
  
  # Ajuste del modelo --------------------------------------------------------
  logit_model <- glm(formula, data = train_set, family = binomial)
  print(summary(logit_model))
  
  #Obtenemos los intervalos de confianza
  conf_int <- confint(logit_model, level = 1 - p_value)
  
  #Creamos la tabla de coeficientes
  coef_tab <- broom::tidy(logit_model) %>%
    mutate(Lower = conf_int[,1],
           Upper = conf_int[,2],
           is_signf = ifelse(p.value < p_value, TRUE, FALSE))
  
  # Predicciones -------------------------------------------------------------
  train_set$prob <- predict(logit_model, newdata = train_set, type = "response")
  test_set$prob  <- predict(logit_model, newdata = test_set, type = "response")
  
  # Clasificación binaria (umbral = 0.5) -------------------------------------
  train_set$pred <- ifelse(train_set$prob >= 0.5, 1, 0)
  test_set$pred  <- ifelse(test_set$prob >= 0.5, 1, 0)
  
  # Métricas de rendimiento --------------------------------------------------
  conf_train <- caret::confusionMatrix(as.factor(train_set$pred), as.factor(train_set[[all.vars(formula)[1]]]))
  conf_test  <- caret::confusionMatrix(as.factor(test_set$pred),  as.factor(test_set[[all.vars(formula)[1]]]))
  
  
  perf_train <- data.frame(
    Dataset = "Train",
    Accuracy = conf_train$overall["Accuracy"],
    Sensitivity = conf_train$byClass["Sensitivity"],
    Specificity = conf_train$byClass["Specificity"]
  )
  
  perf_test <- data.frame(
    Dataset = "Test",
    Accuracy = conf_test$overall["Accuracy"],
    Sensitivity = conf_test$byClass["Sensitivity"],
    Specificity = conf_test$byClass["Specificity"]
  )
  
  performance <- bind_rows(perf_train, perf_test)
  print(performance)
  
  # Curva ROC ----------------------------------------------------------------
  roc_plot <- function(prob, obs){
    roc_obj <- pROC::roc(obs, prob)
    ggroc(roc_obj, color = "#0072B2", size = 1.2) +
      labs(title = sprintf("Curva ROC (AUC = %.3f)", roc_obj$auc),
           x = "1 - Especificidad", y = "Sensibilidad") +
      theme_minimal()
  }
  
  plot_roc <- roc_plot(test_set$prob, test_set[[all.vars(formula)[1]]])
  print(plot_roc)
  
  # Resultados ---------------------------------------------------------------
  result_list <- list(
    modelo = logit_model,
    coef_tab = coef_tab,
    train_set = train_set,
    test_set = test_set,
    performance = performance,
    confusion_train = conf_train,
    confusion_test = conf_test,
    roc_plot = plot_roc
  )
  names(result_list) <- c("modelo", "coeficientes", "train_set","test_set","performance",
                          "confusion_train", "confusion_test","roc_plot")
  return(result_list)
}

# Modelo =======================================================================

#Datos
datos = readr::read_csv("C:/Users/raulb/Desktop/Diplomado/Modulo2/Datos/Rice_Cammeo_Osmancik.csv") %>%
  mutate(Class = ifelse(Class == "Cammeo",0,1)) %>%
  mutate(Class = as.factor(Class))

#Modelo
logistic_model <- regresion_logistica(Class ~ ., datos, prop = 0.70)

#Coeficientes
logistic_model$coeficientes

#Performance
logistic_model$performance

#Matrices de confusion
logistic_model$confusion_train
logistic_model$confusion_test

