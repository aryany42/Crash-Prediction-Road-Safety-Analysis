# Crash-Prediction-Road-Safety-Analysis
The purpose of this project is to examine environmental, temporal, road-level, and behavioural predictors of crash severity in Montgomery County, Virginia. I used a four-phase analytical approach: exploratory data analysis and visualization, one-way ANOVA and multiple linear regression, ordinal logistic regression (proportional odds model), and supplementary spatial analysis conducted using kernel density estimation.

Road safety and crash severity is a major challenge facing public health policy makers due to its prevalence, costs, and major personal impact to the people involved. Predicting the severity of crashes based on different factors thus provides us with a unique insight into what leads to crashes, and the steps we can take to prevent them. I chose a localised dataset of Montgomery County to present an exclusive analytical context: a mid-sized Virginia county anchored by a major research university (Virginia Tech), with high traffic volume corridors serving both commuter and student populations.

## Dataset
The dataset for this study was obtained from the Virginia DMV/VDOT via the Virginia Open Data Portal (https://data.virginia.gov/dataset/crash-data). The final dataset, post-processing, contains 20, 162 crash records localised to Montgomery County, VA, each with GPS coordinates, crash date and time, severity classification, weather, lighting, road surface conditions, collision type, and binary driver behaviour flags (under the influence of alcohol, speeding, distractions involved).

## Key Results 
Key Results: Alcohol-involved crashes had nearly three times the odds of higher severity (OR = 2.91, p < 0.001). Surprisingly, night-time driving was associated with reduced severity (OR = 0.807, p < 0.001). Spatial analysis revealed that crashes are strongly clustered and not randomly distributed.

## Conclusions
Crash severity is driven by a combination of behavioral and environmental conditions. Crash locations are strongly clustered around the US-460/I-81 traffic corridor, the VT campus perimeter and Radford city limits.
