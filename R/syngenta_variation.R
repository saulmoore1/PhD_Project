#!/usr/bin/env R
rm(list=ls())
graphics.off()

# Script to perform linear mixed models on Syngenta screening feature summary 
# results (for each feature) that account for variation in observed behaviour
# with respect to the duration that the worms spent in M9 buffer 
# (time of day effects, starvation effects)

# TODO: Investigate results of January screen: effect of duration/plateID on N2 
#       (column-effects?) # how strong is effect of plateID/columnID

library(tools)
library(lme4) # lmer
library(jtools)
library(sjPlot) # plot_model
library(tidyr)
library(comprehenr)
library(data.table)
library(ggplot2) # ggplot
library(broom) # tidy, glance
library(stargazer) # stargazer
library(RColorBrewer)
#library(MCMCglmm)

project_root_dir <- '/Users/sm5911/Documents/Syngenta_Analysis/'
setwd(project_root_dir)

screening_month <- '2020_Jan' # '2019_Dec'
bluelight_conditions <- c("prestim", "bluelight", "poststim")
drug_types <- c("DMSO","NoCompound")
p_value_threshold <- 0.05

results_dir <- paste0(project_root_dir, "results/", screening_month, "/")
control_results_path <- paste0(project_root_dir, "data/", screening_month, "/control_results.csv")
top256feats_path <- paste0(project_root_dir, "data/top256_tierpsy_no_blob_no_eigen_only_abs_no_norm.csv")

# Functions
createDir <- function(savepath){
  # Check if directory exists for save path provided, and create it if not.
  if (!file.exists(dirname(savepath))){
    dir.create(dirname(savepath), recursive=TRUE)
  }
}

# Reed results: combined metadata + feature summaries
df <- read.csv(control_results_path, header=TRUE, stringsAsFactors=FALSE)
#str(df); dim(df)

# # Subset results for selected worm strains (OPTIONAL)
df <- subset(df, df$worm_strain %in% c("N2","CB4856"))

# Convert worm strain + bluelight to factors
strains <- unique(df$worm_strain) # get strain list
strains <- c("N2", strains[-which(strains=="N2")]) # set N2 as first factor level
df$worm_strain <- factor(df$worm_strain, levels=strains)
df$bluelight <- factor(df$bluelight, levels=c('prestim','bluelight','poststim'))
df$well_name <- factor(df$well_name, level=sort(unique(df$well_name)))

# Read Tierpsy Top256 feature list
top256 <- read.csv(top256feats_path, header=TRUE, stringsAsFactors=FALSE)
top256 <- top256$X1
# Remove 'path_curvature' related features
top256 <- top256[!grepl(paste0("path_curvature", collapse = "|"), top256)]
# Append some intuitive features: 'speed_90th'
top256 <- append(top256, "speed_90th")

# Standardise continuous explanatory variables, so that estimated coefficients
# are on same scale:
# explanatory_vars <- c("","")
# for(i in explanatory_vars){
#   df[[i]] <- scale(df[[i]], center = TRUE, scale = TRUE)
# }

# Subset by drug (treatment) and plot/model results for each feature
PRINT = FALSE
model_params_list <- list()
model_stats_list <- list()
i <- 1
for (drug in drug_types) {
  drug_df <- subset(df, df$drug_type==drug)
  for (feature in top256) {
    feat_df <- drug_df %>% drop_na(feature)
    if (PRINT){print(paste0("Dropped ", as.character(dim(drug_df)[1] - dim(feat_df)[1]), 
                   " missing values for ", feature))}
    
    # Histogram of feature (response variable distribution should be Gaussian)
    hist_plot_path <- paste0(results_dir, drug, "/histograms/", feature, 
                             "_histogram.png")
    createDir(hist_plot_path)
    png(hist_plot_path)
    g <- ggplot(feat_df, aes(.data[[feature]], fill = worm_strain)) + 
         geom_density(alpha = 0.2) + 
         ggtitle(paste("Histogram of", feature))
    print(g)
    dev.off()
    
    # Boxplot of feature by worm strain (fixed effect variable)
    box_plot_path <- paste0(results_dir, drug, "/boxplots/", feature, 
                            "_boxplot_worm_strain.png")
    createDir(box_plot_path)
    png(box_plot_path)
    boxplot(feat_df[[feature]] ~ feat_df$worm_strain, pch=19, cex=0.7,
            xlab="", ylab=feature, las=2)
    dev.off()
    
    # Boxplot of feature by bluelight condition (fixed effect variable)
    box_plot_path <- paste0(results_dir, drug, "/boxplots/", feature, 
                            "_boxplot_bluelight.png")
    png(box_plot_path)
    g <- ggplot(feat_df, aes(.data[[feature]], bluelight, fill = bluelight)) + 
         geom_boxplot() + 
         ggtitle(paste("Effect of bluelight stimulus on:", feature))
    print(g)
    dev.off()
    
    # Plot of feature by worm strain for each bluelight condition
    facet_plot_path <- paste0(results_dir, drug, "/facetplots/", feature,
                            "_facetplot_bluelight.png")
    createDir(facet_plot_path)
    png(facet_plot_path)
    g <- ggplot(aes(worm_strain, .data[[feature]], colour = worm_strain), 
                data=feat_df) + 
      geom_point() + facet_wrap(~ date_yyyymmdd) +
      xlab("worm strain") + ylab(feature) + 
      theme(axis.text.x = element_text(angle = 90))
    print(g)
    dev.off()
    
    # g <- ggplot(feat_df, aes(fill=duration_M9_seconds, y=.data[[feature]], x=worm_strain)) +
    #       geom_bar(position="dodge", stat="identity") + theme_minimal()
    # print(g)
    
    # Scatterplot of feature by duration in M9 (random effect variable)
    scatter_plot_path <- paste0(results_dir, drug, "/scatterplots/", feature, 
                                "_scatterplot.png")
    createDir(scatter_plot_path)
    png(scatter_plot_path)
    g <- ggplot(feat_df, aes(x = duration_M9_seconds, y = .data[[feature]])) +
         geom_point(aes(colour = worm_strain), size = 2) +
         geom_smooth(method = "lm", formula='y~x') +
         labs(y=feature, x="Duration in M9 buffer (s)")
    print(g)
    dev.off()
    
    # In summary, CB4856 appears to move faster, but varies more in speed. 
    # Bluelight and poststim look to have greater speed on average than prestim.
    # There also looks to be a slight positive trend with duration in M9.
    # Let's perform a linear regression to test these effects.
    
    # Linear Regression (taking a 'bottom-up' approach)
    # Basic model: Ignoring bluelight + random variation due to duration in M9
    lm0 <- lm(feat_df[[feature]] ~ worm_strain, data=feat_df)
    if (PRINT){print(summary(lm0))}
    
    # Including interaction between worm strain and bluelight condition
    if (length(unique(feat_df$worm_strain)) <= 5){
      lm1 <- lm(feat_df[[feature]] ~ worm_strain * bluelight, data=feat_df)
      if (PRINT){print(summary(lm1))}
    } else {
      lm1 <- lm(feat_df[[feature]] ~ worm_strain + bluelight, data=feat_df)
      if (PRINT){print(summary(lm1))
                 print("WARNING: Dropped interaction effect of worm strain with bluelight (too many levels)")}
    }
    # Speed 90th:
    # Is there an association between the feature and worm strain? 
    #   Yes, CB4856 is nearly twice the speed of N2 (***)
    # Is there a difference between bluelight conditions?
    #   Yes, bluelight + poststim are significantly different from prestim (***)
    # Is there a difference between worm stains in their responses to bluelight? 
    #   Yes, CB4856 sped up significantly more than N2 response to bluelight (**)
    #        After bluelight, CB4856 decreased speed significantly more than N2 (***)
    # Worm strain explains ~25% variation in speed 90th. Together, worm strain 
    # and bluelight condition explain 34%.
    
    # Model comparison 
    anova(lm0, lm1) 
    # Does adding bluelight explain significantly more variation?
    # Yes. Update our model to lm1.
    
    # Plot + save linear model diagnostics
    lm_plot_path <- paste0(results_dir, drug, "/lm/", feature, 
                           "_model_diagnostics.png")
    createDir(lm_plot_path)
    png(lm_plot_path)
    par(mfrow=c(2,2))
    plot(lm1)
    dev.off()
    # Normal Q-Q: shows some dispersion at the extremes, but this is often the 
    #             case, it doesn't look too bad.
    # Residuals vs leverage: looks good, only few outliers with high leverage.
    
    # Tidy model results into dataframe using broom and store in list
    lm_params <- tidy(lm1)
    lm_params$model <- 'lm'
    lm_params$drug_type <- drug
    lm_params$feature <- feature
    model_params_list[[i]] <- lm_params
    # head(augment(lm1)) # Extract per object information about fitted values/residuals
    lm_stats <- glance(lm1)
    lm_stats$model <- 'lm'
    lm_stats$drug_type <- drug
    lm_stats$feature <- feature
    model_stats_list[[i]] <- lm_stats
    i <- i + 1 # increment index for storing results in list
    
    # Within-group linear models using dplyr + broom
    # feat_df %>% group_by(duration_M9_seconds) %>% do(tidy(lm(feature ~ worm_strain, .)))
    
    # Construct coefficient plots using tidy method
    lm_plot_path <- paste0(results_dir, drug, "/lm/", feature, 
                           "_coefficient_plot.png")
    png(lm_plot_path)
    td <- tidy(lm1, conf.int=TRUE)
    g <- ggplot(td, aes(estimate, term, color=term)) +
         geom_point() +
         geom_errorbarh(aes(xmin = conf.low, xmax = conf.high)) + 
         geom_vline(xintercept=0)
    print(g)
    dev.off()
    
    # Add fitted values to dataframe
    feat_df$lm1_predicted <- predict(lm1)
      
    # Random effects should be categorical variables, with at least 5 levels 
    #   e.g.'Sex' should be added as a fixed effect (2 levels)
    # (https://ourcodingclub.github.io/tutorials/mixed-models/#FERE)
    
    # 2 params(intercept,slope),2 worm strains,3 bluelight conditions,16 durations=192 models! 
    # That's a lot of regression analyses, and would hugely decrease the 
    # sample size per model and also increase the chances of Type I error 
    # (fasely rejecting the null) through multiple comparisons.
    # Instead, we can run a linear mixed effects model to use all the data,
    # and account for variation due to duration spent in M9 by adding it as a 
    # random effect.
    
    # Linear Mixed Model (with random intercept)
    #   worm strain + bluelight added as fixed effects (+ interaction effect)
    #   duration in M9 as a random effect
    if (length(unique(feat_df$worm_strain)) <= 5){
      lmm <- lmer(formula=paste0(feature, "~ worm_strain * bluelight",
                                 "+ (1 | duration_M9_seconds)"), data=feat_df)
      if (PRINT){print(summary(lmm), correlation=TRUE)}
    } else {
      lmm <- lmer(formula=paste0(feature, "~ worm_strain + bluelight",
                                 "+ (1 | duration_M9_seconds)"), data=feat_df)
      if (PRINT){print(summary(lmm), correlation=TRUE)}       
    }
    
    # Model comparison
    # We cannot compare LMM to ordinary LM using anova, but we can use AIC
    if (AIC(lmm) < AIC(lm1)) {if (PRINT) {print(paste("Duration has an effect on", feature))}}
    # Does adding duration in M9 as a random effect explain more variation?
    #    Yes. Adding duration in M9 explains more variance and yields a 
    #    lower AIC = better model. Update our model to lmm.

    # Tidy model results into dataframe using broom
    lm_params <- tidy(lmm)
    lm_params$model <- 'lmm' # Standard errors look small
    lm_params$drug_type <- drug
    lm_params$feature <- feature
    model_params_list[[i]] <- lm_params
    
    # head(augment(lm1)) # Extract per object information about fitted values/residuals
    lm_stats <- glance(lmm)
    lm_stats$model <- 'lmm'
    lm_stats$drug_type <- drug
    lm_stats$feature <- feature
    
    # NB: No p-values given, although they can be inferred post-hoc. 
    # With large sample sizes, p-values based on the likelihood ratio are 
    # generally considered okay.
    
    # Extract the explained variances
    varDurationM9 <- as.numeric(VarCorr(lmm)) # random effect variance
    varResidual <- attr(VarCorr(lmm),"sc")^2 # residual variance
    # Use an ordinary linear model to get variance explained by fixed effects
    varFixed <- var(predict(lm(paste0(feature, "~ worm_strain * bluelight"),
                               data=feat_df))) 
    lm_stats$propVarFixed <- varFixed / (varFixed + varResidual)
    lm_stats$propVarDurationM9 <- varDurationM9 / (varDurationM9 + varResidual)
    model_stats_list[[i]] <- lm_stats
    i <- i + 1 # increment index for storing results in list
    
    # Plot + save residuals vs fitted diagnostics for model
    lmm_plot_path <- paste0(results_dir, drug, "/lmm/", feature, 
                            "_residuals_vs_fitted.png")
    createDir(lmm_plot_path)
    png(lmm_plot_path)
    p <- plot(lmm, ask=FALSE) # no patterns evident = good
    print(p)
    dev.off()
    
    # Plot + save Q-Q plot
    lmm_plot_path <- paste0(results_dir, drug, "/lmm/", feature, 
                            "_normal_QQ_plot.png")
    png(lmm_plot_path)
    qqnorm(resid(lmm))
    qqline(resid(lmm))  # points fall mostly nicely on the Q-Q line = good
    dev.off()
    
    # Construct coefficient plots using tidy method
    lmm_plot_path <- paste0(results_dir, drug, "/lmm/", feature, 
                            "_coefficient_plot.png")
    png(lmm_plot_path)
    td <- tidy(lmm, conf.int=TRUE)
    g <- ggplot(td, aes(estimate, term, color=term)) +
      geom_point() +
      geom_errorbarh(aes(xmin = conf.low, xmax = conf.high)) + 
      geom_vline(xintercept=0) + theme_minimal()
    print(g)
    dev.off()
    # Looks like all effects are still significant after having taken into 
    # account the variation due to duration in M9
    
    # Random effects plot for mixed model
    lmm_plot_path <- paste0(results_dir, drug, "/lmm/", feature, 
                            "_random_effects.png")
    png(lmm_plot_path)
    re.effects <- plot_model(lmm, type = "re", show.values = TRUE)
    print(re.effects)
    dev.off()
    # Not too much structure, looks good
    # TODO: Look into correlation in random effects (non-linear?)
    
    # Add fitted values to dataframe
    feat_df$lmm_predicted <- predict(lmm)
    
    # Plot model fit (scatterplot by duration in M9, with predicted values)
    lmm_plot_path <- paste0(results_dir, drug, "/lmm/", feature, "_lmm_fit.png")
    png(lmm_plot_path)
    g <- ggplot(feat_df, aes(x = duration_M9_seconds, y = .data[[feature]])) +
      geom_point(aes(colour = worm_strain), size = 2) +
      geom_smooth(method = "lm", formula='y~x') +
      geom_line(aes(y = lm1_predicted), size=1, colour='red') +
      geom_line(aes(y = lmm_predicted), size=1, colour='blue') +
      # geom_ribbon(aes(x = duration_M9_seconds, ymin = lmm_predicted - std_error, 
      #             ymax = lmm_predicted + std_error), 
      #             fill = "lightgrey", alpha = 0.5) +  # error band
      ggtitle("Linear model (red) and mixed model (blue) predictions") +
      theme_minimal()
    print(g)
    dev.off()

    # Since we used different worms at the different imaging timepoints
    # throughout the day, we do not need to worry about CROSSED random effects
    # or pseudoreplication due to repeated measurements (athough lme4 handles 
    # partially/fully crossed factors pretty well)
     
    # Linear Mixed Model (nested by imaging plate id)
    # However, our data may be HIERARCHICALLY NESTED, since results for a given 
    # plate may be more similar to each other. So we can create a nested sample
    # within each group for duration M9 to account for the plates that were
    # recorded in that group.
    #feat_df <- within(feat_df, sample <- factor(duration_M9_seconds:imaging_plate_id))
    if (length(unique(feat_df$worm_strain)) <= 5){
      lmm2 <- lmer(formula=paste0(feature, "~ worm_strain * bluelight + ",
                   "(1 | duration_M9_seconds / imaging_plate_id)"), 
                   data=feat_df)
      if (PRINT){print(summary(lmm2), correlation=TRUE)}
    } else {
      lmm2 <- lmer(formula=paste0(feature, "~ worm_strain + bluelight + ",
                                  "(1 | duration_M9_seconds / imaging_plate_id)"), 
                   data=feat_df)
      if (PRINT){print(summary(lmm2), correlation=TRUE)}
    }
    # Is there an association between feature and worm strain/bluelight, after 
    # accounting for duration in M9, AND allowing for different baselines 
    # within each duration for different imaging plates?
    
    # Extract the explained variances
    varNestedPlateID <- as.numeric(VarCorr(lmm2))[1]
    varDurationM9 <- as.numeric(VarCorr(lmm2))[2]
    varResidual <- attr(VarCorr(lmm2),"sc")^2
    
    # Proportion of total variance ecxplained by random effect duration in M9
    propVar_M9 <- varDurationM9 / (varDurationM9 + varResidual)
    propVar_PlateID <- varNestedPlateID / (varNestedPlateID + varResidual)
    # print(paste0("Duration in M9 explains ", round(propVarExplained * 100, 1), 
    #              "% of the total variance in ", feature))
    
    # Tidy model results into dataframe using broom
    lm_params <- tidy(lmm2)
    lm_params$model <- 'lmm2'
    lm_params$drug_type <- drug
    lm_params$feature <- feature
    model_params_list[[i]] <- lm_params
    
    # Extract per object information about fitted values/residuals
    #head(augment(lm1)) 
    lm_stats <- glance(lmm2)
    lm_stats$model <- 'lmm2'
    lm_stats$drug_type <- drug
    lm_stats$feature <- feature

    # Extract the explained variances
    varNestedPlateID <- as.numeric(VarCorr(lmm2))[1] # random effect variances
    varDurationM9 <- as.numeric(VarCorr(lmm2))[2]
    varResidual <- attr(VarCorr(lmm2),"sc")^2 # residual variance
    varFixed <- var(predict(lm(paste0(feature, "~ worm_strain * bluelight"),
                               data=feat_df))) 
    lm_stats$propVarFixed <- varFixed / (varFixed + varResidual)
    lm_stats$propVarDurationM9 <- varDurationM9 / (varDurationM9 + 
                                                   varNestedPlateID + 
                                                   varResidual)
    lm_stats$propVarNestedPlateID <- varNestedPlateID / (varDurationM9 + 
                                                         varNestedPlateID + 
                                                         varResidual)
    model_stats_list[[i]] <- lm_stats
    i <- i + 1 # increment index for storing results in list
    
    # Plot + save residuals vs fitted diagnostics for model
    lmm2_plot_path <- paste0(results_dir, drug, "/lmm2/", feature,
                             "_residuals_vs_fitted.png")
    createDir(lmm2_plot_path)
    png(lmm2_plot_path)
    p <- plot(lmm2, ask=FALSE) # no patterns evident = good
    print(p)
    dev.off()
    
    # Plot + save Q-Q plot
    lmm2_plot_path <- paste0(results_dir, drug, "/lmm2/", feature,
                             "_normal_QQ_plot.png")
    png(lmm2_plot_path)
    qqnorm(resid(lmm2))
    qqline(resid(lmm2)) # points fall nicely onto the line = good!
    dev.off()
    
    # Compare nested model with non-nested model
    anova(lmm, lmm2)
    # NB: Even though you use ML to compare models, you should report parameter 
    # estimates from your final “best” REML model, as ML may underestimate 
    # variance of the random effects. Also, with large sample sizes, p-values 
    # based on the likelihood ratio are generally considered okay.

    # So a nested model with spearate intercepts for imaging plate ID explains
    # more of the observed variance. Update our model to lmm2.
    
    feat_df$lmm2_predicted <- predict(lmm2)
    
    # To visualise what is going on, let plot the duration, with a separate 
    # intercept fitted for each imaging plate.
    lmm2_plot_path <- paste0(results_dir, drug, "/lmm2/", feature, 
                             "_lmm_fit.png")
    png(lmm2_plot_path)
    g <- ggplot(feat_df, aes(x = duration_M9_seconds, y = .data[[feature]])) +
      geom_point(aes(colour = worm_strain), size = 2) +
      geom_smooth(method = "lm", formula='y~x') +
      geom_line(aes(y = lm1_predicted), size=1, colour='red') +
      geom_line(aes(y = lmm_predicted), size=1, colour='blue') +
      geom_line(aes(y = lmm2_predicted), size=1, colour='green') +
      # geom_ribbon(aes(x = duration_M9_seconds, ymin = lmm_predicted - std_error, 
      #             ymax = lmm_predicted + std_error), 
      #             fill = "lightgrey", alpha = 0.5) +  # error band
      ggtitle("Linear model (red) and mixed model (blue) predictions") +
      theme_minimal()
    print(g)
    dev.off()
    
    if (screening_month == "2019_Dec"){
      
      # Linear Mixed Model (with random intercept + random slope for each 
      # imaging plate within each duration)
      lmm3 <- lmer(formula=paste0(feature, "~ worm_strain * bluelight +",
                                  "(1 + worm_strain |",
                                  "duration_M9_seconds / imaging_plate_id)"),
                                  data=feat_df) 
      if (PRINT){print(summary(lmm3))}
    
    
      # PERHAPS OVERKILL, but essentially asking:
      # Is there an association between feature and worm strain, after accounting 
      # for different durations in M9 with different baselines within each 
      # duration for each imaging plate AND allowing for different relationships
      # between each imaging plate within each duration?
      
      # Model comparison
      anova(lmm2, lmm3)
      # lmm3 has a lower AIC, and also BIC, by only estimating 4 more parameters.
      # Update our model to lmm3.
      
      if (PRINT) {stargazer(lmm3, type = "text",
                  digits = 3,
                  star.cutoffs = c(0.05, 0.01, 0.001),
                  digit.separator = "")}
      
      
      # Tidy model results into dataframe using broom
      lm_params <- tidy(lmm3)
      lm_params$model <- 'lmm3'
      lm_params$drug_type <- drug
      lm_params$feature <- feature
      model_params_list[[i]] <- lm_params
      
      # # Extract per object information about fitted values/residuals
      # No longer easy to extract the explained variances as there are many vars
      lm_stats <- glance(lmm3)
      lm_stats$model <- 'lmm3'
      lm_stats$drug_type <- drug
      lm_stats$feature <- feature
      model_stats_list[[i]] <- lm_stats
      i <- i + 1 # increment index for storing results in list
      
      # Plot + save residuals vs fitted diagnostics for model
      lmm3_plot_path <- paste0(results_dir, drug, "/lmm3/", feature,
                               "_residuals_vs_fitted.png")
      createDir(lmm3_plot_path)
      png(lmm3_plot_path)
      p <- plot(lmm3, ask=FALSE) # no patterns evident = good
      print(p)
      dev.off()
      
      # Plot + save Q-Q plot
      lmm3_plot_path <- paste0(results_dir, drug, "/lmm3/", feature,
                               "_normal_QQ_plot.png")
      png(lmm3_plot_path)
      qqnorm(resid(lmm3))
      qqline(resid(lmm3))  # points fall nicely onto the line = good!
      dev.off()
      
      feat_df$lmm3_predicted <- predict(lmm3)
      
      # Plot model fit
      lmm3_plot_path <- paste0(results_dir, drug, "/lmm3/", feature, 
                              "_lmm_fit.png")
      png(lmm3_plot_path)
      g <- ggplot(feat_df, aes(x = duration_M9_seconds, y = .data[[feature]])) +
        geom_point(aes(colour = worm_strain), size = 2) +
        geom_smooth(method = "lm", formula='y~x') +
        geom_line(aes(y = lm1_predicted), size=1, colour='red') +
        geom_line(aes(y = lmm_predicted), size=1, colour='blue') +
        geom_line(aes(y = lmm2_predicted), size=1, colour='green') +
        geom_line(aes(y = lmm3_predicted), size=1, colour='purple') +
        # geom_ribbon(aes(x = duration_M9_seconds, ymin = lmm_predicted - std_error, 
        #             ymax = lmm_predicted + std_error), 
        #             fill = "lightgrey", alpha = 0.5) +  # error band
        ggtitle("Linear model (red) and mixed model (blue) predictions") +
        theme_minimal()
      print(g)
      dev.off()
      
      # # Linear Mixed Model (constraining the random effects to be uncorrelated)
      # lmm4 <- lmer(formula=paste0(feature, "~ worm_strain * bluelight +",
      #                             "(1 | worm_strain) +",
      #                             "(0 + worm_strain | duration_M9_seconds / imaging_plate_id)"),
      #                             data=feat_df)
      # print(summary(lmm4))
      # anova(lmm3, lmm4)
      # # This does not decrease the AIC by that much.
    }
  }
}

# Concatenate model parameter/statistics dataframes from list and save as CSV
params_df <- dplyr::bind_rows(model_params_list)
param_path <- paste0(results_dir, "/model_parameters.csv")
write.csv(params_df, param_path, row.names=FALSE)

stats_df <- dplyr::bind_rows(model_stats_list)
stats_path <- paste0(results_dir, "/model_statistics.csv")
write.csv(stats_df, stats_path, row.names=FALSE)

#####################
# Reporting results
#####################
# 1. Don't report p-values. They are crap!
# 2. Report the fixed effects estimates. These represent the best-guess average 
#    effects in the population. Make a quantitative statement: 
#    "Overall the effect is strong/weak/negligible". 
# 3. Report the confidence limits. Make statements on uncertainty: 
#    "The appears to be a strong effect, but CIs are large, so confidence is low"
#    "The effect is practically zero and we can say so with good certainty"
# 4. Report how variable the effect is between individuals by the random effects 
#    standard deviations: 
#    "On average a strong effect, but considerable variation between subjects"
# 5. Present in the following table structure:
#    parameter name | beta | lower-95 | upper-95 | random effect (SD)

for(drug in drug_types){
  for(model in c('lm')){ #'lmm','lmm2','lmm3'
    stats_lm <- stats_df[which(stats_df$model==model & stats_df$drug_type==drug),]
    percent_sigFeats_lm <- round(table(stats_lm$p.value < p_value_threshold) / 
                                   length(stats_lm$p.value) * 100, 1)
    cat("\n\nDrug:", drug, "\nModel:", model, "\nFalse True\n", percent_sigFeats_lm)
  }
}

# stats_df[grepl(paste0(top256, collapse="|"), stats_df$feature),]

# for(col in c("bluelight","drug_type","imaging_plate_drug_concentration",
#              "instrument_name","imaging_run_number","robot_run_number",
#              "stock_plate_id","worm_strain","duration_M9_seconds",
#              "starvation_level")){
#   cat(paste0("\n", col,":\n"))
#   print(unique(df[[col]]))
# }

# # For plotting statistics of resampling methods
# ggplot(lm_params, aes(statistic, estimate)) + geom_line(color = "red") +
#   geom_ribbon(aes(ymin=(estimate - std.error), ymax=(estimate + std.error)), 
#               alpha=.2) +
#   geom_vline(xintercept = lm_stats$lambda.min) +
#   geom_vline(xintercept = lm_stats$lambda.1se, lty = 2)

