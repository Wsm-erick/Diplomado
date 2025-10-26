#===============================================================================
# Bibliotecas
#===============================================================================
pacman::p_load(rstanarm, ggplot2, tidyverse, gridExtra,
               lmtest, nortest, bayesplot, posterior)

set.seed(1989) #Taylor's Version

#===============================================================================
# Función de Regresión Lineal Bayesiana Mejorada
#===============================================================================
regresion_lineal_bayes <- function(formula, data, prop = 0.66, iter = 2000, chains = 10){
  
  # -------------------------------
  # División en entrenamiento y prueba
  # -------------------------------
  if(prop <= 0 | prop >= 1){
    stop("La proporción debe estar entre 0 y 1")
  }
  
  indices <- 1:nrow(data)
  ind_train <- sample(indices, ceiling(prop * length(indices)))
  ind_test <- setdiff(indices, ind_train)
  
  train_set <- data[ind_train,]
  test_set  <- data[ind_test,]
  
  # -------------------------------
  # Ajuste del modelo bayesiano
  # -------------------------------
  bmodel <- stan_glm(formula,
                     data = train_set,
                     family = gaussian(),
                     chains = chains,
                     iter = iter,
                     refresh = 0)
  
  # -------------------------------
  # Distribución posterior de coeficientes
  # -------------------------------
  draws <- as.data.frame(as.matrix(bmodel))
  coef_names <- names(coef(bmodel))
  draws_coef <- draws %>% select(all_of(coef_names))
  
  coef_plot <- draws_coef %>%
    pivot_longer(cols = everything(), names_to = "Coeficiente", values_to = "Valor") %>%
    ggplot(aes(x = Valor, fill = Coeficiente)) +
    geom_density(alpha = 0.5) +
    facet_wrap(~Coeficiente, scales = "free", ncol = 2) +
    theme_minimal() +
    theme(legend.position = "none") +
    labs(title = "Distribución posterior de los coeficientes", x = "", y = "Densidad")
  
  print(coef_plot)
  
  # -------------------------------
  # Convergencia (Traceplots)
  # -------------------------------
  trace_plot <- bayesplot::mcmc_trace(as.array(bmodel),
                                      pars = coef_names,
                                      facet_args = list(ncol = 2, strip.position = "left")) +
    ggtitle("Convergencia de cadenas MCMC (traceplots)")
  print(trace_plot)
  
  # -------------------------------
  # Distribución del R2 posterior
  # -------------------------------
  # Generamos predicciones por cada muestra posterior
  y_hat_draws <- posterior_predict(bmodel)
  y_obs <- bmodel$y
  
  # Calculamos R² para cada muestra posterior
  r2_samples <- apply(y_hat_draws, 1, function(yhat) {
    1 - sum((y_obs - yhat)^2) / sum((y_obs - mean(y_obs))^2)
  })
  
  r2_df <- data.frame(R2 = r2_samples)
  
  r2_plot <- ggplot(r2_df, aes(x = R2)) +
    geom_density(fill = "#000059", alpha = 0.5) +
    theme_minimal() +
    labs(title = "Distribución posterior del R²", x = "R²", y = "Densidad")
  
  print(r2_plot)
  
  # -------------------------------
  # Análisis de residuales
  # -------------------------------
  residuales <- residuals(bmodel)
  ajustados  <- fitted(bmodel)
  observados <- bmodel$model[,1]
  x <- 1:length(residuales)
  df_rf <- data.frame(x, residuales, ajustados, observados)
  
  # --- Pruebas ---
  ad <- ad.test(residuales)$p.value
  cvm <- cvm.test(residuales)$p.value
  lillie <- lillie.test(residuales)$p.value
  bp <- bptest(lm(ajustados ~ residuales))$p.value
  dw <- dwtest(lm(observados ~ ajustados))$p.value
  
  tests_tab <- tibble(
    Tests = c("Anderson-Darling", "Cramer-von Mises", "Lilliefors",
              "Breusch-Pagan", "Durbin-Watson"),
    p_value = c(ad, cvm, lillie, bp, dw)
  )
  
  print(tests_tab)
  
  # -------------------------------
  # Predicciones (medias posteriores)
  # -------------------------------
  pred_tab <- posterior_predict(bmodel, newdata = test_set)
  pred_mean <- apply(pred_tab, 2, mean)
  test_results <- data.frame(
    Observado = test_set[[all.vars(formula)[1]]],
    Predicho = pred_mean
  )
  
  # -------------------------------
  # Resultado
  # -------------------------------
  bayes_list <- list(
    modelo = bmodel,
    train_set = train_set,
    test_set = test_set,
    residuales = df_rf[,2:4],
    tests = tests_tab,
    predicciones = test_results,
    r2_posterior = r2_df
  )
  
  return(bayes_list)
}

#===============================================================================
# Ejemplo de uso
#===============================================================================
#Modelo 
reg_bayes1 <- regresion_lineal_bayes(Sepal.Length ~ ., data = iris)

#Imprimimos el modelo
reg_bayes1$modelo

#Predicciones
reg_bayes1$predicciones

#Test
reg_bayes1$tests
