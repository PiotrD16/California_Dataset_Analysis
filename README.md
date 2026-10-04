## California Housing analysis program

This program is made for learning purposes and discovering various techniques of data preparation, exploration and analyzing. It also covers some selected topics of Machine Learning algorithms worth attention.

Here is some inital data received by running scripts:
---
### Correlation Matrix (correlations > 0.5)

| Variable 1 | Variable 2 | Correlation |
| :--- | :--- | :---: |
| total_bedrooms | households | 0.979728 |
| total_rooms | total_bedrooms | 0.930380 |
| total_rooms *(ciąg dalszy)* | households | 0.918484 |
| population | households | 0.907222 |
| total_bedrooms | population | 0.877747 |
| total_rooms | population | 0.857126 |
| median_income | median_house_value | 0.688075 |
| longitude | latitude | -0.924664 |

Based on those data we can calculate some additional data to our dataset:

- rooms_per_family = total_rooms / households
- bedrooms_coefficient = total_rooms / total_bedrooms
- people_per_household = population / households

After trying these new features, correlation matrix looks like this: